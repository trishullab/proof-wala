#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from abc import ABC, abstractmethod
from itp_interface.rl.abstraction import Policy
from itp_interface.rl.simple_proof_env import ProofAction, ProofState
from thrall_lib.search.search import SearchAlgorithm


class ProofSearchPolicy(Policy):
    def __init__(self, search_algorithm: SearchAlgorithm):
        assert search_algorithm is not None, "Search algorithm cannot be None"
        self.search_algorithm = search_algorithm
    
    def __call__(self, state: ProofState) -> ProofAction:
        pass

    def update(self, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: typing.Any):
        pass

    def checkpoint(self):
        pass

    def clone(self):
        pass

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {}
    