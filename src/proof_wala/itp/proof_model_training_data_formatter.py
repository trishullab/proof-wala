#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from proof_wala.llm_helpers.model import TrainingDataFormatterCallback
from proof_wala.llm_helpers.theorem_proving_training_dataset import TheoremProvingTrainingDataset
from itp_interface.tools.training_data import TrainingData, TrainingDataFormat
from proof_wala.parsers.proof_model_grammars import ProofModelGrammar, ProofModelPredGrammar
from proof_wala.parsers.grammars.prompt_template_grammar import PromptGrammar

class ProofModelTrainingDataset(TheoremProvingTrainingDataset):
    supported_training_data_configs = [
        'prompt__state__complete__steps_description_state',
        'prompt__state__complete__steps',
        'prompt__state__complete__steps_description',
        'prompt__state__complete__steps_description_state',
        'prompt__state_steps__complete__description',
        'prompt__state_steps__complete__state_description'
    ]
    ordering = {
        'prompt__state__complete__steps_description_state': 
            (['STATE'], ['PROOFSTEP', 'DESCRIPTION', 'STATE']),
        'prompt__state__complete__steps':
            (['STATE'], ['PROOFSTEP']),
        'prompt__state__complete__steps_description':
            (['STATE'], ['PROOFSTEP', 'DESCRIPTION']),
        'prompt__state__complete__steps_description_state':
            (['STATE'], ['PROOFSTEP', 'DESCRIPTION', 'STATE']),
        'prompt__state_steps__complete__state_description': 
            (['STATE', 'PROOFSTEP'], ['STATE', 'DESCRIPTION']),
        'prompt__state_steps__complete__description': 
            (['STATE', 'PROOFSTEP'], ['DESCRIPTION'])
    }
    def __init__(self, 
            training_data: TrainingData,
            characters_per_token: float = 3.6,
            max_tokens: int = 2048,
            state_delta: bool = False,
            max_distance_to_good: int = 10,
            training_data_gen_config: typing.List[str] = ['prompt__state__complete__steps_description_state']):
        assert len(training_data_gen_config) > 0, "Training data gen config is empty"
        assert characters_per_token >= 1, f"Characters per token must be at least 1, but is {characters_per_token}"
        super().__init__(training_data)
        self.prompt_grammar = PromptGrammar()
        self.characters_per_token = characters_per_token
        self.max_tokens = max_tokens
        self.max_tokens_in_prompt = self.max_tokens
        self.max_chars_in_prompt = int(self.max_tokens_in_prompt * self.characters_per_token)
        self.state_delta = state_delta
        self.training_data_gen_config = training_data_gen_config
        ordering_response, ordering_request = ProofModelTrainingDataset.ordering.get(training_data_gen_config[0], ProofModelTrainingDataset.supported_training_data_configs[0])
        self.response_grammar = ProofModelGrammar(state_delta=False, enable_proofstep=True, ordering=ordering_response) # Response grammar does not need state delta
        self.request_grammar = ProofModelPredGrammar(state_delta=self.state_delta, enable_proofstep=True, ordering=ordering_request)
        if self.max_tokens_in_prompt < 0:
            raise ValueError(f"Max tokens in prompt is negative: {self.max_tokens_in_prompt}, increase max tokens or decrease characters per token")
        self.prompt_delimiter = "\n"
        self.max_distance_to_good = max_distance_to_good
    
    def _reconstruct_proof_tree(self, 
        prev_state_id_map: typing.Dict[int, typing.Set[int]], 
        good_state_ids: typing.Set[int], 
        bad_state_ids: typing.Set[int],
        distance_map: typing.Dict[int, int]):
        distance = 1
        while True:
            # The idea is at some point no new good state ids will be added
            # This is level order traversal but in reverse
            new_good_state_ids = set()
            for end_state, start_states in prev_state_id_map.items():
                if end_state in good_state_ids:
                    for start_state in start_states:
                        if start_state not in good_state_ids:
                            # To avoid infinite loop we need to not add the start_state if it is already in good_state_ids
                            new_good_state_ids.add(start_state)
                            dist = distance_map.get(start_state, distance)
                            if dist >= distance:
                                distance_map[start_state] = distance
            distance += 1
            if len(new_good_state_ids) == 0:
                break
            good_state_ids.update(new_good_state_ids)
        # Now identify the states which are not good
        for end_state in prev_state_id_map.keys():
            if end_state not in good_state_ids:
                bad_state_ids.add(end_state)
    
    def _reconstruct_prev_state_id_map(self, training_datas: typing.List[TrainingDataFormat]) -> typing.Tuple[typing.Dict[int, int], int]:
            prev_state_id_map = {}
            done_state = None
            for training_data in training_datas:
                if training_data.addition_state_info is not None and len(training_data.addition_state_info) > 0:
                    start_state = training_data.addition_state_info.get("start_goal_id", None)
                    end_state = training_data.addition_state_info.get("end_goal_id", None)
                    if start_state is not None and end_state is not None:
                        prev_to_end_state = prev_state_id_map.get(end_state, set())
                        prev_to_end_state.add(start_state)
                        prev_state_id_map[end_state] = prev_to_end_state
                    if done_state is None and training_data.addition_state_info.get("done", False):
                        done_state = end_state
            return prev_state_id_map, done_state

    def load(self, **kwargs):
        super().load(**kwargs)
        # Go over each example and create a tree map for value function training
        proof_id_maps : typing.Dict[str, typing.List[TrainingDataFormat]] = {}
        self._cached_training_data = []
        for idx in range(len(self.training_data)):
            example = self.training_data[idx]
            training_datas : typing.List[TrainingDataFormat] = proof_id_maps.get(example.proof_id, [])
            training_datas.append(example)
            self._cached_training_data.append(example)
            proof_id_maps[example.proof_id] = training_datas
        for proof_id, training_datas in proof_id_maps.items():
            prev_state_id_map, done_state = self._reconstruct_prev_state_id_map(training_datas)
            # Now we have the prev_state_id_map and done_state
            # Every state from where we can reach done_state is a should have a good value
            # Every other state should have a bad value
            # First figure out all state ids from where we can reach done_state
            good_state_ids = set()
            good_state_ids.add(done_state)
            bad_state_ids = set()
            distance_map = {done_state: 0}
            self._reconstruct_proof_tree(prev_state_id_map, good_state_ids, bad_state_ids, distance_map)
            # Now we have the good_state_ids and bad_state_ids
            # Now annotate the training data with the value function
            for training_data in training_datas:
                if training_data.addition_state_info is not None and len(training_data.addition_state_info) > 0:
                    end_state_id = training_data.addition_state_info.get("end_goal_id", None)
                    if end_state_id is not None:
                        progress = training_data.addition_state_info.get("progress", "")
                        if end_state_id in good_state_ids and (progress == "StateChanged" or progress == "Done"):
                            distance = distance_map.get(end_state_id, self.max_distance_to_good)
                            distance = min(distance, self.max_distance_to_good)
                            progress = f"[GOOD] [{distance}] {progress}"
                        else:
                            progress = f"[BAD] {progress}"
                        training_data.addition_state_info["progress"] = progress

    def __len__(self):
        super_len = super().__len__()
        return super_len * len(self.training_data_gen_config)

    def __getitem__(self, idx):
        super_len = super().__len__()
        order_idx = idx // super_len
        idx = idx % super_len
        training_data_config = self.training_data_gen_config[order_idx]
        ordering_response, ordering_request = ProofModelTrainingDataset.ordering.get(
            training_data_config, 
            ProofModelTrainingDataset.supported_training_data_configs[0])
        self.response_grammar.ordering = ordering_response
        self.request_grammar.ordering = ordering_request
        if hasattr(self, "_cached_training_data"):
            proof_state = self._cached_training_data[idx]
        else:
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
    data_folder = f".log/traces"
    meta_filename = "local.meta.json"
    training_data = TrainingData(data_folder, meta_filename)
    # Not removing system_prompt_file and conversation_prompt_file though not needed
    # as removing makes the function signature incompatable with the call-type in run_training.py
    training_data_gen_config = ProofModelTrainingDataset.supported_training_data_configs
    with ProofModelTrainingDataset(training_data, max_tokens=4096, state_delta=True, training_data_gen_config=training_data_gen_config) as dataset:
        hf_dataset = dataset.get_hf_dataset()
        formatter = ProofModelPromptTrainingDataFormatter()
        formatted_dataset = formatter(hf_dataset)
        for example_idx, example in enumerate(hf_dataset):
            prompt = example["prompt"]
            completion = example["completion"]
            print(f"[{example_idx + 1}] Prompt:\n{prompt}")
            print('-'*20)
            print(f"Completion:\n{completion}")
            print('-'*20)
            print('='*20)
        prompt_and_completions = formatter.get_prompt_and_completion(hf_dataset)
    pass