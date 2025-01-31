#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import copy
from proof_wala.llm_helpers.model import TrainingDataFormatterCallback
from proof_wala.llm_helpers.theorem_proving_training_dataset import TheoremProvingTrainingDataset
from itp_interface.tools.training_data import TrainingData
from proof_wala.parsers.copra_grammars import CoqGPTResponseDfsGrammar, CoqGPTRequestGrammar, CoqGptRequest, CoqGptResponse
from proof_wala.parsers.grammars.prompt_template_grammar import PromptGrammar

class CodeT5TrainingDataset(TheoremProvingTrainingDataset):
    response_grammar: CoqGPTResponseDfsGrammar = CoqGPTResponseDfsGrammar()
    request_grammar: CoqGPTRequestGrammar = CoqGPTRequestGrammar()
    def __init__(self, 
            training_data: TrainingData,
            characters_per_token: float = 3.6,
            max_tokens: int = 2048,
            no_steps: bool = False,
            lang_tag: typing.Optional[str] = None):
        assert characters_per_token >= 1, f"Characters per token must be at least 1, but is {characters_per_token}"
        super().__init__(training_data)
        self.response_grammar = CoqGPTResponseDfsGrammar()
        self.request_grammar = CoqGPTRequestGrammar()
        self.prompt_grammar = PromptGrammar()
        self.characters_per_token = characters_per_token
        self.max_tokens = max_tokens
        self.max_tokens_in_prompt = self.max_tokens
        self.max_chars_in_prompt = int(self.max_tokens_in_prompt * self.characters_per_token)
        self.no_steps = no_steps
        self.lang_tag = lang_tag
        if self.max_tokens_in_prompt < 0:
            raise ValueError(f"Max tokens in prompt is negative: {self.max_tokens_in_prompt}, increase max tokens or decrease characters per token")
        self.prompt_delimiter = "\n"

    def __getitem__(self, idx):
        response, request = self._get_response_request(idx)
        prompt = self.response_grammar.format_as_per_grammar(response, max_token_cnt=self.max_tokens_in_prompt, characters_per_token=self.characters_per_token)
        completion = self.request_grammar.generate_message_from_gpt_request(request)
        prompt += self.prompt_delimiter
        if self.lang_tag is not None:
            prompt = f"{self.lang_tag}\n{prompt}"
        return {
            "prompt": prompt,
            "completion": completion
        }

    def _get_response_request(self, idx):
        steps : typing.List[str] = []
        last_step = None
        last_tdf = self.training_data[idx]
        if not self.no_steps:
            assert last_tdf.proof_id is not None, f"Proof id is None for training data at index {idx}, cannot infer the previous steps in the proof"
            tdf = self.training_data[idx - 1] if idx > 0 else None
            last_step = tdf.proof_steps if tdf is not None and tdf.proof_id == last_tdf.proof_id else None
            tdf = self.training_data[idx - 2] if idx > 1 else None
            while tdf is not None and last_tdf.proof_id == tdf.proof_id:
                steps.extend(reversed(tdf.proof_steps))
                idx -= 1
                tdf = self.training_data[idx - 1] if idx > 1 else None
            steps.reverse()
        response = CoqGptResponse(
            success=True,
            steps=steps,
            last_step='\n'.join(last_step) if last_step is not None else None,
            training_data_format=last_tdf
        )
        request = CoqGptRequest(
            args=last_tdf.proof_steps
        )
        return response, request
    
    def prompt_formatter(
            response: CoqGptResponse, 
            max_tokens_in_prompt: int = None, 
            characters_per_token: float = 4,
            no_steps: bool = False,
            lang_tag: typing.Optional[str] = None) -> str:
        response_copy = copy.deepcopy(response)
        if no_steps:
            response_copy.steps = []
            response_copy.last_step = None
        prompt = CodeT5TrainingDataset.response_grammar.format_as_per_grammar(
            response_copy,
            max_token_cnt=max_tokens_in_prompt, 
            characters_per_token=characters_per_token)
        if lang_tag is not None:
            prompt = f"{lang_tag}\n{prompt}"
        return prompt

    def response_parser(request_str: str) -> CoqGptRequest:
        if request_str.endswith(CodeT5PromptTrainingDataFormatter.stopping_token):
            request_str = request_str[:-len(CodeT5PromptTrainingDataFormatter.stopping_token)]
        request, _ = CodeT5TrainingDataset.request_grammar.attempt_parsing(request_str)
        return request
    

class CodeT5PromptTrainingDataFormatter(TrainingDataFormatterCallback):
    stopping_token = "[END]"
    def __call__(self, training_data_format_examples: typing.List[typing.Dict[str, str]]) -> typing.List[str]:
        output_texts = []
        all_prompts = training_data_format_examples["prompt"]
        all_completions = training_data_format_examples["completion"]
        stopping_token = ""
        for prompt, completion in zip(all_prompts, all_completions):
            output_text = f"{prompt}{completion}{stopping_token}"
            output_texts.append(output_text)
        return output_texts
    
    def get_prompt_and_completion(self, training_data_examples: typing.List[typing.Dict[str, str]]) -> typing.List[typing.Tuple[str, str]]:
        all_prompts = training_data_examples["prompt"]
        all_completions = training_data_examples["completion"]
        prompt_and_completions = []
        if isinstance(all_prompts, str):
            all_prompts = [all_prompts]
            assert isinstance(all_completions, str)
            all_completions = [all_completions]
        assert len(all_prompts) == len(all_completions)
        for prompt, completion in zip(all_prompts, all_completions):
            # Trim everything after the stopping token from the completion
            completion = completion.split(self.get_stopping_token())[0]
            prompt_and_completion = (prompt, completion + self.get_stopping_token())
            prompt_and_completions.append(prompt_and_completion)
        return prompt_and_completions
    
    def get_stopping_token(self):
        return CodeT5PromptTrainingDataFormatter.stopping_token

if __name__ == "__main__":
    prompt_name = "copra-coq-dfs"
    data_folder = f".log/traces"
    meta_filename = "local.meta.json"
    training_data = TrainingData(data_folder, meta_filename)
    # Not removing system_prompt_file and conversation_prompt_file though not needed
    # as removing makes the function signature incompatable with the call-type in run_training.py
    with CodeT5TrainingDataset(training_data, max_tokens=4096, no_steps=True) as dataset:
        hf_dataset = dataset.get_hf_dataset()
        formatter = CodeT5PromptTrainingDataFormatter()
        formatted_dataset = formatter(hf_dataset)
        for example in hf_dataset:
            prompt = example["prompt"]
            completion = example["completion"]
            print(f"Prompt:\n{prompt}")
            print(f"Completion:\n{completion}")
            print('-'*80)
        prompt_and_completions = formatter.get_prompt_and_completion(hf_dataset)
    pass