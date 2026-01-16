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


def startup_heartbeat(container: Container, agent: Agent, logger: logging.Logger, timeout: int = 30) -> bool:
    """
    Performs startup heartbeat checks to detect silent failures early.

    This runs a series of quick validation commands to ensure the container
    environment is properly set up before the main agent execution.

    Args:
        container: The Docker/Apptainer container.
        agent: The agent being executed.
        logger: Logger for the run.
        timeout: Timeout in seconds for each check.

    Returns:
        True if all checks pass, False otherwise.
    """
    logger.info("[HEARTBEAT] Starting container validation checks...")

    checks = [
        # Basic shell responsiveness
        ("echo 'HEARTBEAT: Container responsive'", "Container responsive"),
        # Check agent directory exists
        (f"test -d {CONSTANTS['AGENT_DIR']} && echo 'HEARTBEAT: Agent directory exists'", "Agent directory"),
        # Check start script exists
        (f"test -f {CONSTANTS['AGENT_DIR']}/start.sh && echo 'HEARTBEAT: Start script exists'", "Start script"),
        # Check data directory is mounted
        ("test -d /home/data && echo 'HEARTBEAT: Data directory mounted'", "Data directory"),
    ]

    # Some agents (e.g., ml-master) don't have nonroot user in their image
    exec_user = None if agent.id.startswith("ml-master") else "nonroot"

    all_passed = True
    for cmd, check_name in checks:
        try:
            exit_code, output = container.exec_run(
                f"timeout {timeout}s bash -c \"{cmd}\"",
                user=exec_user
            )
            output_str = output.decode('utf-8').strip() if output else ""

            if exit_code == 0 and "HEARTBEAT:" in output_str:
                logger.info(f"[HEARTBEAT] OK: {check_name}")
            else:
                logger.warning(f"[HEARTBEAT] FAILED: {check_name} (exit={exit_code}, output={output_str})")
                all_passed = False
        except Exception as e:
            logger.error(f"[HEARTBEAT] ERROR: {check_name} - {str(e)}")
            all_passed = False

    if all_passed:
        logger.info("[HEARTBEAT] All startup checks passed")
    else:
        logger.warning("[HEARTBEAT] Some startup checks failed - agent may encounter issues")

    return all_passed


def execute_agent(container: Container, agent: Agent, logger: logging.Logger):
    """
    Initiates the agent via its start script inside the container.
    Captures both stdout and stderr to prevent silent failures.
    """
    cmd = ["bash", f"{CONSTANTS['AGENT_DIR']}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]

    logger.info("[HEARTBEAT] Agent execution starting...")
    logger.info(f"[HEARTBEAT] Command: {' '.join(cmd)}")

    # Some agents (e.g., ml-master) don't have nonroot user in their image
    logger.info(f"[HEARTBEAT] Before exec_user OK")
    exec_user = None if agent.id.startswith("ml-master") else "nonroot"
    logger.info(f"[HEARTBEAT] After exec_user OK, exec_user: {exec_user}")
    
    # Execute with both stdout and stderr captured (demux=False merges them)
    # stream=True allows us to see output in real-time
    exit_code, output = container.exec_run(
        cmd,
        stream=True,
        user=exec_user,
        demux=False,  # Merge stdout and stderr into single stream
    )
    logger.info(f"[HEARTBEAT] After exec_run OK, exit_code: {exit_code}, output: {output}")
    
    # Track if we received any output (silence detection)
    output_received = False
    last_output_time = time.monotonic()

    for chunk in output:
        output_received = True
        last_output_time = time.monotonic()
        decoded = chunk.decode('utf-8').strip()
        if decoded:
            logger.info(f"[Container] {decoded}")

    # Log completion status
    elapsed_since_output = time.monotonic() - last_output_time
    if not output_received:
        logger.warning("[HEARTBEAT] WARNING: No output received from agent - possible silent failure")

    logger.info(f"[HEARTBEAT] Agent execution finished (exit_code={exit_code})")


def save_logs_on_failure(container: Container, run_dir: Path, logger: logging.Logger) -> None:
    """
    Attempts to save logs from the container when a failure occurs.
    This helps with debugging issues like grading server startup failures.

    Args:
        container: The Docker container.
        run_dir: The directory where logs will be saved.
        logger: Logger for the run.
    """
    try:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Try to capture the entrypoint.log from the container
        exit_code, output = container.exec_run("cat /home/logs/entrypoint.log")
        if exit_code == 0 and output:
            entrypoint_log_path = logs_dir / "entrypoint.log"
            entrypoint_log_path.write_bytes(output)
            logger.info(f"[DEBUG] Saved entrypoint.log to {entrypoint_log_path}")
            # Also log the contents for immediate visibility
            logger.info("[DEBUG] entrypoint.log contents:")
            for line in output.decode('utf-8', errors='replace').split('\n')[-50:]:  # Last 50 lines
                logger.info(f"  {line}")

        # Try to get any conda/pip errors by checking if grading server can be imported
        exit_code, output = container.exec_run(
            '/opt/conda/bin/conda run -n mleb python -c "from mlebench.grade import validate_submission; from mlebench.registry import registry; print(\'OK\')"'
        )
        if exit_code != 0:
            logger.error(f"[DEBUG] Grading server import test failed: {output.decode('utf-8', errors='replace')}")

        # Check if Flask is available
        exit_code, output = container.exec_run(
            '/opt/conda/bin/conda run -n mleb python -c "import flask; print(flask.__version__)"'
        )
        if exit_code != 0:
            logger.error(f"[DEBUG] Flask import failed: {output.decode('utf-8', errors='replace')}")

    except Exception as e:
        logger.warning(f"[DEBUG] Could not save failure logs: {e}")


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
            "mode": "rw",
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
        bypass_entrypoint=agent.bypass_entrypoint,
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
        except Exception:
            # If start fails or doesn't exist (shim might not need it if already started)
            # Check if it's the Apptainer shim
            if not hasattr(container, "status") or container.status != "created":
                pass  # Assume it's running or doesn't support start
        
        # Perform startup heartbeat checks to detect silent failures early
        startup_heartbeat(container, agent, logger)

        # Skip grading server check for agents that have their own validation
        # - aira-dojo: has its own validation
        # - ml-master: has its own grading server on port 5001 (started by start.sh)
        skip_grading_server = agent.id.startswith("aira-dojo") or agent.id.startswith("ml-master")
        if not skip_grading_server:
            logger.info("[HEARTBEAT] Waiting for grading server...")
            exit_code, _ = container.exec_run(
                'timeout 120s sh -c "while ! curl -s http://localhost:5000/health > /dev/null; do sleep 1; done"'
            )
            if exit_code != 0:
                raise RuntimeError(
                    "The grading server failed to start within 120 seconds. This is likely due to an error in `entrypoint.sh`; check the logs."
                )
            logger.info("[HEARTBEAT] Grading server is ready")
        else:
            logger.info(f"[HEARTBEAT] Skipping grading server check for {agent.id} agent")

        execute_agent(container, agent, logger)
        save_output(container, run_dir, container_config)
        time_end = time.monotonic()
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")
        return run_dir
    except Exception as e:
        # Save logs before cleanup to help with debugging
        logger.error(f"[DEBUG] Run failed with error: {e}")
        save_logs_on_failure(container, run_dir, logger)
        raise e
    finally:
        clean_up(container, logger, retain_container)
