#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from itp_interface.rl.simple_proof_env import ProofState, ProofAction, ProofEnvInfo, ProgressState, ProofTree
from thrall_lib.proof_search.search_driver import ProofActionGenerator, ProofStateInfo
from thrall_lib.llm_helpers.model import Model
from thrall_lib.itp.codet5_training_data_formatter import CodeT5TrainingDataset, CoqGptResponse, CoqGptRequest

def get_qed_for_language(language: ProofAction.Language):
    if language == ProofAction.Language.COQ:
        return "Qed."
    elif language == ProofAction.Language.LEAN:
        return "end"
    else:
        raise ValueError(f"Language {language} not supported")

class LlmProofActionGenerator(ProofActionGenerator):
    def __init__(self, 
        model: Model,
        prompt_formatter: typing.Callable[[ProofStateInfo], str],
        response_parser: typing.Callable[[str], typing.List[str]],
        width: int = 4,
        **generation_args):
        assert model is not None, "Model cannot be None"
        super().__init__(width)
        self.model = model
        self.prompt_formatter = prompt_formatter
        self.response_parser = response_parser
        self._generation_args = generation_args if generation_args else {}
        self._generation_args["num_return_sequences"] = self.width
        pass
        
    def generate_actions(self, state_info: ProofStateInfo, k: int = None) -> typing.List[typing.Tuple[float, ProofAction]]:
        state : ProofState = state_info.proof_state
        done : bool = state_info.done
        if done:
            return [ProofAction(ProofAction.ActionType.EXIT, state.language) for _ in range(k)]
        elif len(state.training_data_format.start_goals) == 0: # No more goals to prove
                qed = get_qed_for_language(state.language)
                return [(1.0, ProofAction(ProofAction.ActionType.RUN_TACTIC, state.language, tactics=[qed])) for _ in range(k)]
        else:
            # generate actions using the model
            prompt = self.prompt_formatter(state_info)
            generation_results = self.model.generate(prompt, **self._generation_args)
            assert len(generation_results.results) == 1, "Only one prompt is used"
            result = generation_results[0]
            raw_outputs = result.generated_text
            neg_log_probabilties = result.neg_log_likelihood
            tactics_list = [self.response_parser(output) for output in raw_outputs]
            actions = [(neg_log_prob, ProofAction(ProofAction.ActionType.RUN_TACTIC, state.language, tactics=tactics)) for neg_log_prob, tactics in zip(neg_log_probabilties, tactics_list)]
            return actions

class CodeT5PromptFormatter:
    def __init__(self, max_token_in_prompt: int, character_per_token: float):
        self.max_token_in_prompt = max_token_in_prompt
        self.character_per_token = character_per_token

    def __call__(self, state_info: ProofStateInfo) -> str:
        state : ProofState = state_info.proof_state
        info : ProofEnvInfo = state_info.info
        success = info is None or (info.progress != ProgressState.FAILED and info.progress != ProgressState.STATE_UNCHANGED)
        proof_tree : ProofTree = state.proof_tree
        actions_till_state : typing.List[ProofAction] = proof_tree.actions
        steps = ['\n'.join(action.kwargs['tactics']) for action in actions_till_state]
        coq_response : CoqGptResponse = CoqGptResponse(
            success=success,
            steps=steps[:-1] if len(steps) > 1 else [],
            last_step=steps[-1] if len(steps) > 0 else None,
            error_message=info.error_message if info is not None else None,
            training_data_format=state.training_data_format
        )
        return CodeT5TrainingDataset.prompt_formatter(coq_response, self.max_token_in_prompt, self.character_per_token)

class CodeT5ResponseParser:        
    def __call__(self, response: str) -> typing.List[str]:
        gpt_request : CoqGptRequest = CodeT5TrainingDataset.response_parser(response)
        return gpt_request.args

if __name__ == "__main__":
    from thrall_lib.proof_search.test_search_driver import test_search_algorithm, ProofFoundHeuristic
    from thrall_lib.search.beam_search import BeamSearch
    from thrall_lib.search.best_first_search import BestFirstSearch
    from itp_interface.rl.simple_proof_env import ProofExecutorCallback, ProofEnv
    import logging
    model_path = ".log/run_training/new_model/thrall-codet5-small-compcert-2048/best-model-20240228-072825"
    is_seq2seq = True
    max_seq_length = 2048
    character_per_token = 3.6
    model = Model(model_path, is_seq2seq=is_seq2seq)
    prompt_formatter = CodeT5PromptFormatter(max_token_in_prompt=max_seq_length, character_per_token=character_per_token)
    response_parser = CodeT5ResponseParser()
    width = 10
    max_new_tokens=150
    temperature=0.75 # Nucleus sampling
    do_sample=True # Nucleus sampling
    top_k=width # Nucleus sampling
    stop_tokens=["[END]"]
    padding=True
    return_full_text=False
    compute_probabilities=True
    with model:
        generator = LlmProofActionGenerator(model, prompt_formatter, response_parser, width,
            max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_k=top_k,
            stop_tokens=stop_tokens, padding=padding, return_full_text=return_full_text, compute_probabilities=compute_probabilities)
        proof_exec_callback = ProofExecutorCallback(
            project_folder=".",
            file_path="src/thrall_lib/data/proofs/coq/simple2/thms.v"
        )
        theorem_names = [
            "nat_add_comm",
            "double_neg",
            "trival_implication",
            "modus_ponens",
            "modus_tollens",
            "disjunctive_syllogism",
            "contrapositive",
            "nat_zero_add",
            "nat_add_zero",
            "nat_add_succ",
            "nat_succ_add"
        ]
        for search_aglo in [BestFirstSearch(), BeamSearch(3)]:
            algo_name = search_aglo.__class__.__name__
            print(f"Running tests for {algo_name}")
            for theorem_name in theorem_names:
                print(f"Trying to prove {theorem_name}")
                language = ProofAction.Language.COQ
                always_retrieve_thms = False
                logger = logging.getLogger(__name__)
                env = ProofEnv("test", proof_exec_callback, theorem_name, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger)
                test_search_algorithm(
                    exp_name=os.path.join(algo_name, theorem_name),
                    algo=search_aglo,
                    proof_env=env,
                    proof_search_heuristic=ProofFoundHeuristic(),
                    action_generator=generator,
                    search_width=width,
                    attempt_count=5,
                    timeout_in_secs=12000)
                print('-' * 80)
            print('=' * 80)