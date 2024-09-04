import ray
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from itp_interface.tools.training_data import TrainingData

class TheoremProvingTrainingDataset(Dataset):
    def __init__(self, training_data: TrainingData):
        self.training_data = training_data
    
    def load(self, **kwargs):
        self.training_data.load()
        ray.shutdown() # Force shutdown ray because everything is in memory now.
        # execute ray stop in terminal to stop ray

    def unload(self):
        self.training_data.unload()

    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.unload()
        pass

    def __len__(self):
        assert self.training_data._is_loaded, "Training data not loaded"
        return len(self.training_data)
    
    def __getitem__(self, idx):
        assert self.training_data._is_loaded, "Training data not loaded"
        example = self.training_data[idx]
        goals = []
        for idx, goal in enumerate(example.start_goals):
            if len(goal.hypotheses) > 0:
                hyps = '\n'.join(goal.hypotheses)
                goals.append(f"[GOAL] {idx + 1}\n{goal.goal}\n[HYPOTHESES] {idx + 1}\n{hyps}")
            else:
                goals.append(f"[GOAL] {idx + 1}\n{goal.goal}")
        goals = '\n'.join(goals)
        goals = goals + '\n[PROOFSTEP]\n'
        proofstep = '\n'.join(example.proof_steps)
        proofstep = f"{proofstep}"
        return {
            "goals": goals,
            "proofstep": proofstep
        }
    
    def get_hf_dataset(self):
        assert self.training_data._is_loaded, "Training data not loaded"
        return HFDataset.from_list(self)