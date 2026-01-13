import logging
import subprocess
import time
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Mimic Docker's ExecResult named tuple for compatibility
ExecResult = namedtuple("ExecResult", ["exit_code", "output"])

class ApptainerContainer:
    def __init__(self, instance_name: str, image: str, environment: Optional[Dict[str, str]] = None):
        self.name = instance_name
        self.image = image
        self.status = "created"
        self.environment = environment or {}

    def stop(self, timeout: int = 10):
        """Stop the apptainer instance"""
        cmd = ["apptainer", "instance", "stop", self.name]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self.status = "exited"
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop instance {self.name}: {e.stderr.decode()}")

    def remove(self, force: bool = False):
        """Apptainer instances are removed on stop, but we might need to cleanup artifacts if any"""
        pass

    def exec_run(
        self,
        cmd: Union[str, List[str]],
        stdout: bool = True,
        stderr: bool = True,
        stream: bool = False,
        detach: bool = False,
        user: str = "",
        workdir: str = "",
    ) -> Any:
        """
        Execute a command inside the running instance using `apptainer exec instance://...`
        """
        base_cmd = ["apptainer", "exec"]
        if workdir:
            base_cmd.extend(["--pwd", workdir])

        # Pass environment variables
        for k, v in self.environment.items():
            base_cmd.extend(["--env", f"{k}={v}"])
        
        base_cmd.append(f"instance://{self.name}")
        
        if isinstance(cmd, str):
            # If it's a shell string, wrap it in sh -c
            command = ["/bin/bash", "-c", cmd]
        else:
            command = cmd

        full_cmd = base_cmd + command
        
        logger.debug(f"Executing in container: {' '.join(full_cmd)}")

        if detach:
            # Run in background and return immediately
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE if stdout else subprocess.DEVNULL,
                stderr=subprocess.PIPE if stderr else subprocess.DEVNULL,
            )
            return ExecResult(exit_code=0, output=b"Detached process started")

        if stream:
            # Generator for streaming output
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr into stdout for simple streaming
                bufsize=1,
                universal_newlines=False
            )
            # Create a generator wrapper to yield bytes from stdout
            def output_generator():
                while True:
                    chunk = process.stdout.read(4096)
                    if not chunk:
                        break
                    yield chunk
            return ExecResult(exit_code=None, output=output_generator())

        # Blocking run
        result = subprocess.run(
            full_cmd,
            capture_output=True
        )
        return ExecResult(exit_code=result.returncode, output=result.stdout)

    def get_archive(self, path: str):
        """
        Simulate get_archive. Since apptainer shares the filesystem (or we can mount),
        we can usually just access the files if they are in bind mounts.
        However, if the file is internal to the overlay, we need to 'cat' it out.
        """
        # This is a simplified implementation that just cats the file
        # Real docker returns a tar stream.
        # For MLE-bench, this is used in `extract_from_container`.
        # We might need to implement a tar wrapper here.
        
        cmd = ["apptainer", "exec", f"instance://{self.name}", "tar", "cf", "-", path]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process.stdout, {}

class ApptainerClient:
    def __init__(self):
        self.containers = self.ContainerCollection(self)

    class ContainerCollection:
        def __init__(self, client):
            self.client = client

        def create(
            self,
            image: str,
            command: Optional[Union[str, List[str]]] = None,
            detach: bool = False,
            environment: Optional[Dict[str, str]] = None,
            volumes: Optional[Dict[str, Dict[str, str]]] = None,
            name: Optional[str] = None,
            **kwargs
        ) -> ApptainerContainer:
            """
            Start an Apptainer instance.
            """
            if not name:
                name = f"mle-bench-{uuid.uuid4().hex[:8]}"

            # Prepare arguments
            instance_args = ["apptainer", "instance", "start"]
            
            # Handle binds
            if volumes:
                bind_list = []
                for host_path, vol_config in volumes.items():
                    container_path = vol_config.get("bind", host_path)
                    mode = vol_config.get("mode", "rw")
                    bind_option = f"{host_path}:{container_path}"
                    if mode == "ro":
                        bind_option += ":ro"
                    bind_list.append(bind_option)
                
                if bind_list:
                    instance_args.extend(["--bind", ",".join(bind_list)])

            # Handle environment variables
            # With --contain, we need to pass env vars explicitly via --env flag
            if environment:
                for k, v in environment.items():
                    instance_args.extend(["--env", f"{k}={v}"])

            # Add GPU support if requested (naive check)
            if kwargs.get("device_requests"):
                instance_args.append("--nv")
            
            # Isolate container
            instance_args.append("--contain")
            instance_args.append("--ipc")
            # Enable writing to tmpfs (needed for pip install, etc.)
            instance_args.append("--writable-tmpfs")
            
            # Add image and name
            # Note: Apptainer instance start syntax: [options] <image> <instance_name> [args]
            instance_args.append(image)
            instance_args.append(name)

            logger.info(f"Starting apptainer instance: {' '.join(instance_args)}")

            subprocess.run(instance_args, check=True)

            return ApptainerContainer(name, image, environment=environment)
            
        def get(self, container_id: str):
            # Implement if needed to find existing instances
            pass

        def list(self):
            # Implement apptainer instance list
            pass

def from_env():
    return ApptainerClient()

