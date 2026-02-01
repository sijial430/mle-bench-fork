import argparse
import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import docker

from agents.registry import Agent
from agents.registry import registry as agent_registry
from agents.run import run_in_container
from agents.utils import apply_cli_overrides
from environment.defaults import DEFAULT_CONTAINER_CONFIG_PATH
from mlebench.data import is_dataset_prepared
from mlebench.registry import Competition, registry
from mlebench.utils import create_run_dir, get_logger, get_runs_dir, get_timestamp

logger = get_logger(__name__)


# The @dataclass(frozen=True) decorator marks Task as a dataclass whose fields are immutable (frozen),
# meaning once an instance is created, its attribute values cannot be changed. This is useful for
# safety and correctness in concurrent or asynchronous settings, where tasks should be passed around
# without risk of accidental modification. However, you could remove 'frozen=True' if immutability is unnecessary.
@dataclass(frozen=True)
class Task:
    run_id: str
    seed: int
    image: str
    path_to_run_group: Path
    path_to_run: Path
    agent: Agent
    competition: Competition
    container_config: dict[str, Any]


async def worker(
    idx: int,
    queue: asyncio.Queue[Task],
    client: docker.DockerClient,
    tasks_outputs: dict[str, dict[str, Any]],
) -> None:
    while True:
        task = await queue.get()

        # Create logger for the run
        run_logger = get_logger(str(task.path_to_run))
        log_file_handler = logging.FileHandler(task.path_to_run / "run.log")
        log_file_handler.setFormatter(
            logging.getLogger().handlers[0].formatter
        )  # match the formatting we have
        run_logger.addHandler(log_file_handler)
        run_logger.propagate = False

        run_logger.info(
            f"[Worker {idx}] Running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
        )

        task_output = {}
        try:
            await asyncio.to_thread(
                run_in_container,
                client=client,
                competition=task.competition,
                agent=task.agent,
                image=task.agent.name,
                container_config=task.container_config,
                retain_container=args.retain,
                run_dir=task.path_to_run,
                logger=run_logger,
            )
            task_output["success"] = True

            run_logger.info(
                f"[Worker {idx}] Finished running seed {task.seed} for {task.competition.id} and agent {task.agent.name}"
            )
        except Exception as e:
            stack_trace = traceback.format_exc()
            run_logger.error(type(e))
            run_logger.error(stack_trace)
            run_logger.error(
                f"Run failed for seed {task.seed}, agent {task.agent.id} and competition "
                f"{task.competition.id}"
            )
            task_output["success"] = False
        finally:
            tasks_outputs[task.run_id] = task_output
            queue.task_done()


async def main(args):
    if args.runtime == "apptainer":
        from agents import apptainer_client
        client = apptainer_client.from_env()
    else:
        client = docker.from_env()
        
    global registry
    registry = registry.set_data_dir(Path(args.data_dir))

    agent = agent_registry.get_agent(args.agent_id)
    merged_kwargs, merged_env_vars = apply_cli_overrides(agent, args.kwargs, args.env_vars)
    agent = replace(agent, kwargs=merged_kwargs, env_vars=merged_env_vars)
    if args.kwargs:
        logger.info(f"Overridden kwargs: {args.kwargs}")
    if args.env_vars:
        logger.info(f"Overridden env_vars: {args.env_vars}")

    if agent.privileged and not (
        os.environ.get("I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS", "False").lower()
        in ("true", "1", "t")
    ):
        raise ValueError(
            "Agent requires running in a privileged container, but the environment variable `I_ACCEPT_RUNNING_PRIVILEGED_CONTAINERS` is not set to `True`! "
            "Carefully consider if you wish to run this agent before continuing. See agents/README.md for more details."
        )

    run_group = f"{get_timestamp()}_run-group_{agent.name}"

    # Load competition ids and check all are prepared
    with open(args.competition_set, "r") as f:
        competition_ids = [line.strip() for line in f.read().splitlines() if line.strip()]
    for competition_id in competition_ids:
        competition = registry.get_competition(competition_id)
        if not is_dataset_prepared(competition):
            raise ValueError(
                f"Dataset for competition `{competition.id}` is not prepared! "
                f"Please run `mlebench prepare -c {competition.id}` to prepare the dataset."
            )

    with open(args.container_config, "r") as f:
        container_config = json.load(f)

    # Create tasks for each (competition * seed)
    logger.info(f"Launching run group: {run_group}")
    tasks = []
    for seed in range(args.n_seeds):
        for competition_id in competition_ids:
            competition = registry.get_competition(competition_id)
            run_dir = create_run_dir(competition.id, agent.id, run_group)
            run_id = run_dir.stem
            task = Task(
                run_id=run_id,
                seed=seed,
                image=agent.name,
                agent=agent,
                competition=competition,
                path_to_run_group=run_dir.parent,
                path_to_run=run_dir,
                container_config=container_config,
            )
            tasks.append(task)

    logger.info(f"Creating {args.n_workers} workers to serve {len(tasks)} tasks...")

    # Create queue of tasks, and assign workers to run them
    queue = asyncio.Queue()
    for task in tasks:
        queue.put_nowait(task)
    workers = []
    tasks_outputs = {}
    for idx in range(args.n_workers):
        w = asyncio.create_task(worker(idx, queue, client, tasks_outputs))
        workers.append(w)

    # Wait for all tasks to be completed and collect results
    started_at = time.monotonic()
    await queue.join()
    time_taken = time.monotonic() - started_at

    for w in workers:
        w.cancel()  # Cancel all workers now that the queue is empty

    await asyncio.gather(*workers, return_exceptions=True)

    # Generate metadata.json
    metadata = {
        "run_group": run_group,
        "created_at": get_timestamp(),
        "runs": tasks_outputs,
    }
    run_group_dir = get_runs_dir() / run_group
    with open(run_group_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=False, default=str)
    logger.info(f"{args.n_workers} workers ran for {time_taken:.2f} seconds in total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent on a set of competitions in a Docker container."
    )
    parser.add_argument(
        "--agent-id",
        help="Agent ID of the agent to run.",
        type=str,
    )
    parser.add_argument(
        "--competition-set",
        type=str,
        required=True,
        help="Path to a text file with a single competition ID on each line",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers to run in parallel",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=False,
        default=1,
        help="Number of seeds to run for each competition",
    )
    parser.add_argument(
        "--container-config",
        help="Path to a JSON file with an environment configuration; these args will be passed to `docker.from_env().containers.create`",
        type=str,
        required=False,
        default=DEFAULT_CONTAINER_CONFIG_PATH,
    )
    parser.add_argument(
        "--retain",
        help="Whether to retain the container after the run instead of removing it.",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the directory where all assets associated with the run are stored.",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        help="Path to the directory containing the competition data.",
        type=str,
        required=False,
        default=registry.get_data_dir(),
    )
    parser.add_argument(
        "--runtime",
        help="Container runtime to use: 'docker' (default) or 'apptainer'.",
        type=str,
        choices=["docker", "apptainer"],
        default="docker",
    )
    parser.add_argument(
        "--kwargs",
        help="Additional kwargs to pass to the agent (overrides config.yaml). Format: key=value key2=value2",
        type=str,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--env-vars",
        help="Additional env vars to pass to the container (overrides config.yaml). Format: KEY=value KEY2=value2",
        type=str,
        nargs="*",
        default=[],
    )
    args = parser.parse_args()
    logger = get_logger(__name__)

    asyncio.run(main(args))
