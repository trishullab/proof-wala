#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import os
from proof_wala.llm_helpers.model import TrainingDataFormatterCallback
from proof_wala.llm_helpers.llama2_chat_format import Llama2ChatFormat
from proof_wala.llm_helpers.theorem_proving_training_dataset import TheoremProvingTrainingDataset
from itp_interface.tools.training_data import TrainingData
from proof_wala.parsers.copra_grammars import CoqGPTResponseDfsGrammar, CoqGPTRequestGrammar, CoqGptRequest, CoqGptResponse
from proof_wala.parsers.grammars.prompt_template_grammar import PromptGrammar

class CopraTrainingDataset(TheoremProvingTrainingDataset):
    def __init__(self, 
            training_data: TrainingData, 
            system_prompt_file: str, 
            conversation_prompt_file: str,
            characters_per_token: float = 3.6,
            max_tokens: int = 2048):
        assert os.path.exists(system_prompt_file), f"System prompt file {system_prompt_file} does not exist"
        assert os.path.exists(conversation_prompt_file), f"Conversation prompt file {conversation_prompt_file} does not exist"
        assert characters_per_token >= 1, f"Characters per token must be at least 1, but is {characters_per_token}"
        super().__init__(training_data)
        self.response_grammar = CoqGPTResponseDfsGrammar()
        self.request_grammar = CoqGPTRequestGrammar()
        self.system_prompt_file = system_prompt_file
        self.conversation_prompt_file = conversation_prompt_file
        self.prompt_grammar = PromptGrammar()
        self.system_message = self.prompt_grammar.get_main_message(self.system_prompt_file)
        self.characters_per_token = characters_per_token
        self.conversation_messages = self.prompt_grammar.get_conv_messages(self.conversation_prompt_file)
        self.system_message_chars_length = self._get_message_length(self.system_message)
        self.conversation_message_chars_length = self._get_message_length(self.conversation_messages)
        self.system_message_token_length = int(self.system_message_chars_length / self.characters_per_token)
        self.conversation_message_token_length = int(self.conversation_message_chars_length / self.characters_per_token)
        self.max_tokens = max_tokens
        self.max_tokens_in_prompt = self.max_tokens - self.system_message_token_length - self.conversation_message_token_length
        self.max_chars_in_prompt = int(self.max_tokens_in_prompt * self.characters_per_token)
        if self.max_tokens_in_prompt < 0:
            raise ValueError(f"Max tokens in prompt is negative: {self.max_tokens_in_prompt}, increase max tokens or decrease characters per token")
        self.llama2_chat_format = Llama2ChatFormat()

    def __getitem__(self, idx):
        response, request = self._get_response_request(idx)
        prompt = self.response_grammar.format_as_per_grammar(response, max_token_cnt=self.max_tokens_in_prompt, characters_per_token=self.characters_per_token)
        completion = self.request_grammar.generate_message_from_gpt_request(request)
        prompt_msg = self.prompt_grammar.get_main_message_from_string(prompt, role="user")
        completion_msg = self.prompt_grammar.get_main_message_from_string(completion, role="assistant")  
        messages = [self.system_message] + self.conversation_messages + [prompt_msg]
        partial_prompt, _ = self.llama2_chat_format(messages)
        completion_msg, _ = self.llama2_chat_format([completion_msg])
        return {
            "prompt": partial_prompt,
            "completion": completion_msg
        }
    
    def _get_message_length(self, message):
        # for base types like str, int, float, etc. the length is the number of characters in the string representation
        # for lists, the length is the sum of the lengths of the elements
        # for dicts, the length is the sum of the lengths of the keys and values
        # for tuples, the length is the sum of the lengths of the elements
        if isinstance(message, str):
            return len(message)
        elif isinstance(message, int):
            return len(str(message))
        elif isinstance(message, float):
            return len(str(message))
        elif isinstance(message, list):
            return sum([self._get_message_length(element) for element in message])
        elif isinstance(message, dict):
            return sum([self._get_message_length(key) + self._get_message_length(value) for key, value in message.items()])
        elif isinstance(message, tuple):
            return sum([self._get_message_length(element) for element in message])
        else:
            raise Exception(f"Unknown type {type(message)}")

    def _get_response_request(self, idx):
        last_tdf = self.training_data[idx]
        tdf = self.training_data[idx - 1] if idx > 0 else None
        last_step = tdf.proof_steps if tdf is not None and tdf.proof_id == last_tdf.proof_id else None
        tdf = self.training_data[idx - 2] if idx > 1 else None
        steps : typing.List[str] = []
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
    

class CopraPromptTrainingDataFormatter(TrainingDataFormatterCallback):
    def __call__(self, training_data_format_examples: typing.List[typing.Dict[str, str]]) -> typing.List[str]:
        output_texts = []
        all_prompts = training_data_format_examples["prompt"]
        all_completions = training_data_format_examples["completion"]
        stopping_token = "" # The stopping token is already included in the completion because LLama2ChatFormat adds it
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
        return "[END]"

if __name__ == "__main__":
    prompt_name = "copra-coq-dfs"
    system_prompt_file = f"src/proof_wala/data/prompts/system/{prompt_name}.md"
    conversation_prompt_file = f"src/proof_wala/data/prompts/conversation/{prompt_name}.md"
    data_folder = f".log/train"
    meta_filename = "local.meta.json"
    training_data = TrainingData(data_folder, meta_filename)
    with CopraTrainingDataset(training_data, system_prompt_file, conversation_prompt_file, max_tokens=4096) as dataset:
        hf_dataset = dataset.get_hf_dataset()
        formatter = CopraPromptTrainingDataFormatter()
        formatted_dataset = formatter(hf_dataset)
        for example in formatted_dataset:
            print(example)
        prompt_and_completions = formatter.get_prompt_and_completion(hf_dataset)
    pass