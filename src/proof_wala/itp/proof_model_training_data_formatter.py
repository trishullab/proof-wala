#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from proof_wala.llm_helpers.model import TrainingDataFormatterCallback
from proof_wala.llm_helpers.theorem_proving_training_dataset import TheoremProvingTrainingDataset
from itp_interface.tools.training_data import TrainingData
from proof_wala.parsers.proof_model_grammars import ProofModelGrammar, ProofModelPredGrammar
from proof_wala.parsers.grammars.prompt_template_grammar import PromptGrammar

class ProofModelTrainingDataset(TheoremProvingTrainingDataset):
    def __init__(self, 
            training_data: TrainingData,
            characters_per_token: float = 3.6,
            max_tokens: int = 2048,
            state_delta: bool = False):
        assert characters_per_token >= 1, f"Characters per token must be at least 1, but is {characters_per_token}"
        super().__init__(training_data)
        self.prompt_grammar = PromptGrammar()
        self.characters_per_token = characters_per_token
        self.max_tokens = max_tokens
        self.max_tokens_in_prompt = self.max_tokens
        self.max_chars_in_prompt = int(self.max_tokens_in_prompt * self.characters_per_token)
        self.state_delta = state_delta
        self.response_grammar = ProofModelGrammar(state_delta=False) # Response grammar does not need state delta
        self.request_grammar = ProofModelPredGrammar(state_delta=self.state_delta)
        if self.max_tokens_in_prompt < 0:
            raise ValueError(f"Max tokens in prompt is negative: {self.max_tokens_in_prompt}, increase max tokens or decrease characters per token")
        self.prompt_delimiter = "\n"

    def __getitem__(self, idx):
        proof_state = self.training_data[idx]
        prompt = self.response_grammar.format_as_per_grammar(proof_state, max_token_cnt=self.max_tokens_in_prompt, characters_per_token=self.characters_per_token)
        completion = self.request_grammar.format_as_per_grammar(proof_state, max_token_cnt=self.max_tokens_in_prompt, characters_per_token=self.characters_per_token)
        prompt += self.prompt_delimiter
        return {
            "prompt": prompt,
            "completion": completion
        }
    
    # def prompt_formatter(
    #         response: CoqGptResponse, 
    #         max_tokens_in_prompt: int = None, 
    #         characters_per_token: float = 4):
    #     prompt = ProofModelTrainingDataset.response_grammar.format_as_per_grammar(
    #         response, 
    #         max_token_cnt=max_tokens_in_prompt, 
    #         characters_per_token=characters_per_token)
    #     return prompt

    # def response_parser(request_str: str) -> CoqGptRequest:
    #     if request_str.endswith(ProofModelPromptTrainingDataFormatter.stopping_token):
    #         request_str = request_str[:-len(ProofModelPromptTrainingDataFormatter.stopping_token)]
    #     request, _ = ProofModelTrainingDataset.request_grammar.attempt_parsing(request_str)
    #     return request
    

class ProofModelPromptTrainingDataFormatter(TrainingDataFormatterCallback):
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
        return ProofModelPromptTrainingDataFormatter.stopping_token

if __name__ == "__main__":
    data_folder = f".log/train"
    meta_filename = "local.meta.json"
    training_data = TrainingData(data_folder, meta_filename)
    # Not removing system_prompt_file and conversation_prompt_file though not needed
    # as removing makes the function signature incompatable with the call-type in run_training.py
    with ProofModelTrainingDataset(training_data, max_tokens=4096) as dataset:
        hf_dataset = dataset.get_hf_dataset()
        formatter = ProofModelPromptTrainingDataFormatter()
        formatted_dataset = formatter(hf_dataset)
        for example in formatted_dataset:
            print(example)
        prompt_and_completions = formatter.get_prompt_and_completion(hf_dataset)
    pass