import typing
from datasets import Dataset
from itp_interface.tools.training_data import TrainingData
from itp_interface.tools.training_data_format import TrainingDataFormat

class TheoremProvingTrainingDataFormatCallback:
    def __init__(self):
        pass
    
    def __call__(self, training_data_format: TrainingDataFormat) -> typing.Tuple[str, str]:
        raise NotImplementedError

class TheoremProvingTrainingDataset(Dataset):
    def __init__(self, training_data: TrainingData, format_callback: TheoremProvingTrainingDataFormatCallback):
        self.training_data = training_data
        self._format_callback = format_callback
    
    def load(self, **kwargs):
        self.training_data.load()

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
        x, y = self._format_callback(self.training_data[idx])
        return x, y