#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from proof_wala.llm_helpers.model import TrainingDataFormatterCallback

class BasicTrainingDataFormatterCallback(TrainingDataFormatterCallback):
    def __init__(self):
        pass
    
    def __call__(self, training_data_format_examples: typing.List[typing.Dict[str, str]]) -> typing.List[str]:
        output_texts = []
        all_goals = training_data_format_examples["goals"]
        all_proofsteps = training_data_format_examples["proofstep"]
        stopping_token = self.get_stopping_token()
        for goals, proofstep in zip(all_goals, all_proofsteps):
            output_text = f"{goals}{proofstep}{stopping_token}"
            output_texts.append(output_text)
        return output_texts
    
    def get_prompt_and_completion(self, training_data_examples: typing.List[typing.Dict[str, str]]) -> typing.List[typing.Tuple[str, str]]:
        all_goals = training_data_examples["goals"]
        all_proofsteps = training_data_examples["proofstep"]
        prompt_and_completions = []
        if isinstance(all_goals, str):
            all_goals = [all_goals]
            assert isinstance(all_proofsteps, str)
            all_proofsteps = [all_proofsteps]
        assert len(all_goals) == len(all_proofsteps)
        for goals, proofstep in zip(all_goals, all_proofsteps):
            prompt_and_completion = (goals, proofstep + self.get_stopping_token())
            prompt_and_completions.append(prompt_and_completion)
        return prompt_and_completions
    
    def get_stopping_token(self):
        return "[END]"