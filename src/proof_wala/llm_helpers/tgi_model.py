import json
import logging
import typing
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from model import Model, GenerationResults, GenerationResult
from tgi_server import TGIServer

class TgiModel(Model):
    """
    Wrapper for interacting with models hosted on TGI servers.
    """

    def __init__(self, name: str, log_folder: str = None, **kwargs):
        """
        Initialize TGI Model instance.

        Args:
            name (str): Name of the model hosted on TGI.
            log_folder (str): Optional folder path for logging.
            **kwargs: Additional arguments for the model or configuration.
        """
        self.port = kwargs.get("port", 8080)
        self.hostname = kwargs.get("hostname", "localhost")
        self.base_url = f"http://{self.hostname}:{self.port}"
        self.model_name = name
        self.is_chat_model = kwargs.get("is_chat_model", False)

        # Setup logging
        self.logger = logging.getLogger(f"TgiModel-{name}")
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        if log_folder:
            handler = logging.FileHandler(f"{log_folder}/tgi_model.log")
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("TgiModel initialized")

    def _send_request(self, endpoint: str, payload: dict) -> dict:
        """
        Utility method to send a request to the TGI server.

        Args:
            endpoint (str): API endpoint (e.g., "/v1/completions").
            payload (dict): Request payload.

        Returns:
            dict: Response data from the TGI server.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error("Request to %s failed: %s", url, e)
            raise

    def tokenize(self, inputs: typing.List[typing.Any], kwargs: dict = None):
        """
        Tokenize the input strings using the TGI API.

        Args:
            inputs (list[str]): List of input strings to tokenize.
            kwargs (dict): Additional tokenization parameters.

        Returns:
            list: Tokenized inputs from TGI.
        """
        kwargs = kwargs or {}
        self.logger.info("Tokenizing inputs via TGI API: %s", inputs)

        endpoint = "/v1/chat_tokenize" if self.is_chat_model else "/v1/tokenize"
        if self.is_chat_model:
            supported_params = ["max_tokens", "logprobs", "frequency_penalty", "n", 
                "presence_penalty", "response_format", "seed", "stop", "stream", "stream_options",
                "temperature", "tool_choice", "tool_prompt", "tools", "top_logprobs", "top_p"]
            payload = {
                "messages": [{"role": "user", "content": text} for text in inputs] if self.is_chat_model else None
            } | {k: v for k, v in kwargs.items() if k in supported_params}
        else:
            supported_params = ['adapter_id', 'best_of', 'decoder_input_details', 
            'details', 'do_sample', 'frequency_penalty', 'grammar', 'max_new_tokens', 'repetition_penalty', 
            'return_full_text', 'seed', 'stop', 'temperature', 'top_k', 'top_n_tokens', 
            'top_p', 'truncate', 'typical_p', 'watermark']
            payload = {
                "inputs": inputs,
                "parameters": {k: v for k, v in kwargs.items() if k in supported_params}
            }

        return self._send_request(endpoint, payload)

    def _generate_request(self, inputs: typing.Union[str, typing.List[typing.Dict]] , **kwargs) -> typing.List[str]:
        """
        Helper method to send a generation request for a single input.

        Args:
            inputs (str): The input text.
            **kwargs: Additional generation parameters.

        Returns:
            list[str]: Generated text responses.
        """
        endpoint = "/v1/chat/completions" if self.is_chat_model else "/v1/completions"

        # Construct the payload
        payload = {
            "max_tokens": kwargs.get("max_new_tokens", 50),
            "n": kwargs.get("n", 1),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.95),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "stream": False, # Disable streaming for now
        }

        if self.is_chat_model:
            assert isinstance(inputs, typing.List[typing.Dict])
            assert all("role" in msg and "content" in msg for msg in inputs)
            payload["messages"] = inputs
        else:
            assert isinstance(inputs, str)
            payload["prompt"] = inputs

        # Add optional parameters only if present
        optional_params = ["stop", "suffix", "seed"]
        for param in optional_params:
            if param in kwargs:
                payload[param] = kwargs[param]

        # Log the payload
        self.logger.debug("Sending request to %s with payload: %s", endpoint, json.dumps(payload, indent=2))

        # Send request and handle response
        try:
            response = self._send_request(endpoint, payload)
            self.logger.debug("Response from %s: %s", endpoint, json.dumps(response, indent=2))

            # Extract responses based on the model type
            if self.is_chat_model:
                return [{"generation": {choice.get("message", {})}, "finish_reason": choice.get("finish_reason", None)} for choice in response.get("choices", [])]
            else:
                return [{"generation": choice.get("text", ""), "finish_reason": choice.get("finish_reason", None)} for choice in response.get("choices", [])]

        except Exception as e:
            self.logger.error("Error during generation request: %s", e)
            return []

    def generate(self, inputs: typing.Union[str, typing.List[str], typing.List[typing.List[typing.Dict]], typing.List[typing.Dict]], **kwargs) -> GenerationResults:
        """
        Generate text completions for a batch of inputs using the TGI model.

        Args:
            inputs (Union[list[Any], Any]): Input text or list of texts.
            **kwargs: Additional generation parameters.

        Returns:
            GenerationResults: Generated text responses encapsulated in a GenerationResults object.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise ValueError("Input must be a string or list of strings")
        if not self.is_chat_model:
            assert all(isinstance(input_text, str) for input_text in inputs)
        else:
            if all(isinstance(input_text, dict) for input_text in inputs):
                inputs = [inputs]
            assert all(isinstance(input_text, list) for input_text in inputs)
            assert all(all(isinstance(msg, dict) for msg in input_text) for input_text in inputs)

        self.logger.debug("Generating text for batch of inputs: %s", inputs)
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_input = {}
            for inp_idx, inp in enumerate(inputs):
                num_return_sequences = kwargs.get("num_return_sequences", 1)
                for _ in range(num_return_sequences):
                    future_to_input[executor.submit(self._generate_request, inp, **kwargs)] = inp_idx
            responses = {}
            for future in as_completed(future_to_input):
                inp_idx = future_to_input[future]
                input_text = inputs[inp_idx]
                resp = responses.get(inp_idx, [])
                try:
                    resp.extend(future.result())
                    responses[inp_idx] = resp
                except Exception as e:
                    self.logger.error("Error processing input '%s': %s", input_text, e)
                    responses[inp_idx] = []
            results = []
            for inp_idx, generated_texts in responses.items():
                input_text = inputs[inp_idx]
                generated_text_list = [gen["generation"] for gen in generated_texts]
                finish_reasons = [gen["finish_reason"] for gen in generated_texts]
                results.append(GenerationResult(input_text=input_text, generated_text=generated_text_list, finish_reason=finish_reasons))
        self.logger.debug("Generated batch responses: %s", results)
        return GenerationResults(results=results)

if __name__ == "__main__":
    import os
    model_path = '<root>/models/proof-wala-codet5-small-coq-lean-4-2048/checkpoint-337500'
    if "<root>" in model_path:
        model_path = model_path.replace("<root>/", os.environ.get("ROOT", "").trim('/') + "/")
    import time
    with TGIServer(
        path_to_local_model=model_path,
        port=8080,
        gpus=True,
        logdir=".logs/tgi_server",
        cuda_visible_devices="0,1,2,3,4,5,6,7"
    ) as tgi_server:
        if tgi_server.is_alive():
            tgi_server.ping_test()

            model = TgiModel(
                name="proof-wala-codet5-small-coq-lean",
                hostname=tgi_server.hostname,
                port=tgi_server.port,
                is_chat_model=False
            )
            avg_time = 0
            total_generations = 0
            for run in range(10):
                # Simple test for generation
                inputs = ["a + e = a", "a + 0 = a", "a + 1 = 1 + a", "a*a + 2*a + 1 = (a + 1)*(a + 1)"]
                num_return_sequences = 32
                print("\nGeneration Test:")
                time_start = time.time()
                responses = model.generate(inputs, max_new_tokens=150, temperature=0.75, num_return_sequences=num_return_sequences, stop=["[END]"])
                end_time = time.time()
                for result in responses.results:
                    print(f"Input: {result.input_text}\nResponses: {result.generated_text}\n Finish reasons: {result.finish_reason}")
                total_generations += len(inputs) * num_return_sequences
                print(f"Time taken: {end_time - time_start:.2f} seconds")
                print(f"Total generations: {total_generations}")
                print(f"Average time per generation: {(end_time - time_start) / total_generations:.2f} seconds")
                print('=' * 80)
                avg_time = (avg_time * run + (end_time - time_start)) / (run + 1)
            print(f"Average time taken: {avg_time:.2f} seconds")
            print(f"Avg generations per second: {avg_time / total_generations:.2f}")
        else:
            print("TGI server failed to start")
