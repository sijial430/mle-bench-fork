import logging
import time
from pathlib import Path

import docker
from docker.models.containers import Container
from dotenv import dotenv_values

from agents.registry import Agent
from environment.utils import (
    create_competition_container,
    extract_from_container,
    extract_from_container_sysbox,
)
from mlebench.registry import Competition
from mlebench.utils import purple

CONSTANTS = dotenv_values(Path(__file__).parent.resolve() / ".shared_env")


def save_output(container: Container, save_dir: Path, container_config: dict) -> Path:
    """
    Extracts the submission, logs, and code directories from the container

    and saves them to the specified directory.

    Args:
        container: The Docker container.
        save_dir: The directory where the output file will be saved.
        container_config: The container configuration.
    Returns:
        Path to the output directory.
    """
    if "runtime" in container_config and container_config["runtime"] == "sysbox-runc":
        extraction_fn = extract_from_container_sysbox
    else:
        extraction_fn = extract_from_container

    for dir_type in ["SUBMISSION_DIR", "LOGS_DIR", "CODE_DIR"]:
        container_dir = CONSTANTS[dir_type]
        extraction_fn(container, container_dir, save_dir)

    return save_dir


def execute_agent(container: Container, agent: Agent, logger: logging.Logger):
    """
    Initiates the agent via its start script inside the container.
    """
    cmd = ["bash", f"{CONSTANTS['AGENT_DIR']}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]

    logger.info("Running agent...")
    exit_code, output = container.exec_run(cmd, stream=True, user="nonroot")

    for chunk in output:
        logger.info(f"[Container] {chunk.decode('utf-8').strip()}")


def clean_up(container: Container, logger: logging.Logger, retain: bool = False) -> bool:
    """
    Stops and removes the container.

    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Cleaning up container: {container.name}")
    try:
        container.stop()
        if not retain:
            container.remove()
        logger.info(f"Container {container.name} stopped and removed.")
        return True
    except Exception as e:
        logger.error(
            f"Error cleaning up: {e}. You may wish to manually check the status of the {container.name} container."
        )
        return False


def run_in_container(
    client: docker.DockerClient,
    competition: Competition,
    agent: Agent,
    image: str,
    container_config: dict,
    retain_container: bool,
    run_dir: Path,
    logger: logging.Logger,
) -> Path:
    """
    Runs environment containing the competition and agent for a set maximum amount of time.

    Args:
        client: Docker or Apptainer client.
        competition: The competition to run.
        agent: The agent to run.
        image: The Docker image to use. Assumes the image is built.
        container_config: Configuration for the Docker container.
        retain_container: Whether to retain the container after the run instead of removing it.
        run_dir: Path to the directory where all assets associated with the run are stored.
        logger: Logger for the run.

    Returns:
        Path to the output file.
    """
    # Adjust volume paths for Apptainer if necessary, or trust the shim
    # Get the agent's directory (parent of start.sh)
    agent_dir = agent.start.parent.resolve().as_posix()

    # Get the mlebench source directory (for agents that need it like aira-dojo)
    mlebench_dir = Path(__file__).parent.parent / "mlebench"

    volumes_config = {
        competition.public_dir.resolve().as_posix(): {
            "bind": "/home/data",
            "mode": "ro",
        },
        competition.private_dir.resolve().as_posix(): {
            "bind": f"/private/data/{competition.id}/prepared/private/",
            "mode": "ro",
        },
        agent_dir: {
            "bind": "/home/agent",
            "mode": "ro",
        },
    }

    # Mount mlebench source for aira-dojo agents (superimage doesn't have it installed)
    if agent.id.startswith("aira-dojo") and mlebench_dir.exists():
        volumes_config[mlebench_dir.resolve().as_posix()] = {
            "bind": "/home/mlebench",
            "mode": "ro",
        }

    container = create_competition_container(
        client=client,
        competition=competition,
        container_config=container_config,
        volumes_config=volumes_config,
        env_vars={
            "COMPETITION_ID": competition.id,
            **agent.env_vars,
        },
        container_image=image,
        privileged=agent.privileged,
    )

    logger.info(purple(f"Run started: {run_dir}"))
    try:
        time_start = time.monotonic()
        
        # Docker client requires explicit start, Apptainer shim starts on creation
        # We can check if it has a 'start' method and call it if so (Docker)
        # or if it's already running (Apptainer shim might auto-start)
        # But consistent with the shim design, let's assume create_competition_container
        # calls client.containers.create which might return a started or stopped container.
        # The shim's 'create' starts the instance.
        # Docker's 'create' does NOT start.
        
        try:
            container.start()
        except:
            # If start fails or doesn't exist (shim might not need it if already started)
            # Check if it's the Apptainer shim
             if not hasattr(container, "status") or container.status != "created":
                 pass # Assume it's running or doesn't support start
        
        # Skip grading server check for agents that have their own validation (e.g., aira-dojo)
        skip_grading_server = agent.id.startswith("aira-dojo")
        if not skip_grading_server:
            exit_code, _ = container.exec_run(
                'timeout 60s sh -c "while ! curl -s http://localhost:5000/health > /dev/null; do sleep 1; done"'
            )
            if exit_code != 0:
                raise RuntimeError(
                    "The grading server failed to start within 60 seconds. This is likely due to an error in `entrypoint.sh`; check the logs."
                )
        else:
            logger.info("Skipping grading server check for aira-dojo agent")
        execute_agent(container, agent, logger)
        save_output(container, run_dir, container_config)
        time_end = time.monotonic()
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")
        return run_dir
    except Exception as e:
        raise e
    finally:
        clean_up(container, logger, retain_container)
