import subprocess
import os
import datetime
import logging
import requests
import time

class TGIServer:
    def __init__(
        self,
        volume: str = None,
        port: int = 8080,
        gpus: bool = True,
        shm_size: str = "1g",
        trust_remote_code: bool = False,
        model_id: str = None,
        path_to_local_model: str = None,
        docker_image: str = "ghcr.io/huggingface/text-generation-inference:latest",
        logdir: str = ".logs/tgi_server",
        hostname: str = "0.0.0.0",
        cuda_visible_devices: str = "0"
    ):
        """
        Initialize TGI Server Parameters.
        
        Args:
            model_id (str): Hugging Face Hub model ID.
            path_to_local_model (str): Local path to the model directory.
            volume (str): Directory to mount into the container, optional if local model is provided.
            port (int): Port to expose the server on (default 8080).
            gpus (bool): Use GPU support if available (default True).
            shm_size (str): Shared memory size (default '1g').
            trust_remote_code (bool): Allow remote custom model code (default False).
            docker_image (str): Docker image for TGI (default HF's TGI image).
            logdir (str): Directory where logs will be stored.
        """
        assert model_id or path_to_local_model, "Either model_id or path_to_local_model must be provided"
        assert not (model_id and path_to_local_model), "Only one of model_id or path_to_local_model can be provided"
        if path_to_local_model:
            assert volume is None or os.path.abspath(volume) == os.path.dirname(path_to_local_model), (
                "If local model path is provided, volume must match its base directory or be omitted."
            )
        
        self.model_id = model_id if model_id else ""
        self.volume = os.path.abspath(volume) if volume else None
        self.port = port
        self.gpus = gpus
        self.shm_size = shm_size
        self.trust_remote_code = trust_remote_code
        self.path_to_local_model = path_to_local_model if path_to_local_model else ""
        self.is_local_model = path_to_local_model is not None
        self.docker_image = docker_image
        self.hostname = hostname
        self.logdir = os.path.abspath(logdir)
        os.makedirs(self.logdir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logdir = os.path.join(self.logdir, self.timestamp)
        os.makedirs(self.logdir, exist_ok=True)
        self.container_id = None
        self.process = None
        self.cuda_visible_devices = cuda_visible_devices

        # Initialize logger
        log_filename = os.path.join(self.logdir, f"tgi_server.log")
        self.logger = logging.getLogger("TGIServer")
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(file_handler)
        self.logger.info("TGI Server initialized")

    def __enter__(self):
        """Context management: Start the server when entering the context."""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context management: Stop the server when exiting the context."""
        self.stop_server()
        if exc_type:
            self.logger.error(f"Exception occurred: {exc_val}")

    def _find_existing_container(self):
        """Find an existing container running on the specified port."""
        try:
            check_command = f"docker ps --filter 'publish={self.port}' --format '{{{{.ID}}}}'"
            result = subprocess.run(check_command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
            container_id = result.stdout.strip()
            return container_id if container_id else None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error checking for existing container: {e}")
            return None

    def start_server(self):
        """Build the Docker command and start the TGI server, logging output to a file."""
        # Check if a container is already running on the specified port
        existing_container = self._find_existing_container()
        if existing_container:
            self.logger.info(f"Found existing container {existing_container} running on port {self.port}. Attaching to it.")
            self.container_id = existing_container
            return

        # Validate inputs
        if self.is_local_model and not self.path_to_local_model:
            self.logger.error("Path to the local model is required if is_local_model=True")
            raise ValueError("Path to the local model is required if is_local_model=True")
        
        # Handle local model path with correct volume mapping
        if self.is_local_model:
            base_volume = os.path.dirname(self.path_to_local_model)
            relative_model_path = os.path.relpath(self.path_to_local_model, base_volume)
            model_option = f"--model-id /data/{relative_model_path}"
            volume_mapping = f"-v {base_volume}:/data"
        else:
            model_option = f"--model-id {self.model_id}"
            volume_mapping = f"-v {self.volume}:/data" if self.volume else ""

        # Add trust-remote-code flag if required
        trust_remote_flag = "--trust-remote-code" if self.trust_remote_code else ""

        # Add GPU flag
        gpu_flag = "--gpus all" if self.gpus else ""

        # Set CUDA_VISIBLE_DEVICES environment variable
        environ = f"--env CUDA_VISIBLE_DEVICES={self.cuda_visible_devices}"

        # Build Docker command
        docker_command = (
            f"docker run {environ} {gpu_flag} --shm-size {self.shm_size} -p {self.port}:80 --hostname {self.hostname} "
            f"{volume_mapping} {self.docker_image} "
            f"{model_option} {trust_remote_flag}"
        )
        
        self.logger.info(f"Starting TGI server with command at {datetime.datetime.now().isoformat()}:")
        self.logger.info(docker_command)

        # Run the Docker command and redirect output to log file
        try:
            log_file_path = os.path.join(self.logdir, "docker_output.log")
            with open(log_file_path, "w") as log_file:
                self.process = subprocess.Popen(
                    docker_command,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                self.logger.info(f"Docker process started with PID: {self.process.pid}")

            # Wait for server readiness by monitoring logs
            if self._wait_for_log_message("Connected", log_file_path):
                # Fetch the container ID
                inspect_command = "docker ps -q -l"
                result = subprocess.run(inspect_command, shell=True, check=True, stdout=subprocess.PIPE, text=True)
                self.container_id = result.stdout.strip()
                self.logger.info(f"TGI server started successfully. Container ID: {self.container_id}")
            else:
                self.logger.error("TGI server failed to start within the timeout period.")
                self.stop_server()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error starting TGI server. Check logs at {log_file_path}")
            self.stop_server()
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.stop_server()

    def _wait_for_log_message(self, message, log_file_path, timeout=60):
        """Wait for a specific log message to appear in the Docker output log."""
        self.logger.info(f"Waiting for log message: '{message}'")
        start_time = datetime.datetime.now()
        while (datetime.datetime.now() - start_time).seconds < timeout:
            try:
                with open(log_file_path, "r") as log_file:
                    logs = log_file.read()
                    if message in logs:
                        self.logger.info(f"Log message '{message}' found.")
                        return True
            except FileNotFoundError:
                self.logger.warning("Log file not found yet, retrying...")
            time.sleep(1)
        self.logger.error(f"Timeout reached while waiting for log message: '{message}'")
        return False

    def stop_server(self):
        """Stop the running TGI server container."""
        if not self.container_id:
            self.logger.warning("No container ID found. Cannot stop server.")
            return

        stop_command = f"docker stop {self.container_id}"
        self.logger.info(f"Stopping TGI server with command: {stop_command}")

        try:
            subprocess.run(stop_command, shell=True, check=True)
            self.logger.info("TGI server stopped successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error stopping TGI server: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")

        if self.process:
            self.process.terminate()
            self.logger.info(f"Terminated Docker process with PID: {self.process.pid}")

    def is_alive(self):
        """Check if the TGI server is running and responding."""
        if self.container_id:
            self.logger.info(f"Checking if TGI server in container {self.container_id} is alive.")
            # Use ping_test to verify responsiveness
            response = self.ping_test()
            if response is not None:
                self.logger.info("TGI server is alive and responding.")
                return True
            else:
                self.logger.warning("TGI server is running but not responding.")
                return False
        else:
            self.logger.warning("No container ID found. TGI server is not running.")
            return False

    def ping_test(self, retries=3, delay=2):
        """Run a test generation request to verify the TGI server is responsive."""
        url = f"http://{self.hostname}:{self.port}/v1/completions"
        payload = {
            "max_tokens": 32,
            "n": 2,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "prompt": "Explain neural networks."
        }
        for attempt in range(1, retries + 1):
            try:
                self.logger.info(f"Sending ping test to {url}, attempt {attempt}/{retries}")
                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                if response.status_code == 200:
                    self.logger.info("Ping test successful. Response: %s", response.json())
                    return response.json()
                else:
                    self.logger.error(f"Ping test failed with status code {response.status_code}. Response: {response.text}")
            except requests.ConnectionError as e:
                self.logger.error(f"Ping test failed. Connection error: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error during ping test: {e}")
            self.logger.info(f"Retrying ping test after {delay} seconds...")
            time.sleep(delay)
        self.logger.error("Ping test failed after all retries.")
        return None

if __name__ == "__main__":
    model_path = '<root>/models/proof-wala-codet5-small-coq-lean-4-2048/checkpoint-337500'
    if "<root>" in model_path:
        model_path = model_path.replace("<root>/", os.environ.get("ROOT", "").trim('/') + "/")
    with TGIServer(
        path_to_local_model=model_path,
        port=8080,
        gpus=True,
        logdir=".logs/tgi_server"
    ) as tgi_server:
        if tgi_server.is_alive():
            tgi_server.ping_test()