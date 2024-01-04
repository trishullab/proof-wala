import os
import torch
import typing
from enum import Enum
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

class AutoRegressiveSequenceSearch(Enum):
    NucleusSampling = "nucleus_sampling"
    BeamSearch = "beam_search"
    def __str__(self):
        return self.value

@dataclass_json
@dataclass
class GenerationResult(object):
    input_text: str
    generated_text: typing.List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class GenerationResults(object):
    results: typing.List[GenerationResult] = field(default_factory=list)

    def __iter__(self):
        return iter(self.results)
    
    def __getitem__(self, key):
        assert isinstance(key, int), "Please provide an integer key"
        return self.results[key]

class StopOnTokens(StoppingCriteria):
    """Stopping criteria based on a list of tokens"""
    def __init__(self, stop_tokens: typing.List[str], tokenizer: AutoTokenizer, input_length: int, device: str = None):
        self.stop_token_ids = stop_tokens
        # self.input_length = input_length
        self.tokenizer = tokenizer
        self.stop_tokens = stop_tokens
        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        stop_token_ids = [self.tokenizer.encode(stop_word, add_special_tokens=False)
                    for stop_word in self.stop_tokens]
        stop_token_ids = [torch.LongTensor(x).to(self.device) for x in stop_token_ids]
        self.max_stop_token_id_length = max([len(x) for x in self.stop_token_ids])
        self.pad_token_tensor = torch.LongTensor([self.tokenizer.pad_token_id]).to(self.device)
        self.stop_decisions = {}
        self.input_length = input_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] <= self.input_length:
            return False
        stop_decisions = [-1] * input_ids.shape[0]
        for idx in range(input_ids.shape[0]):
            input_ids_slice = input_ids[idx][-self.max_stop_token_id_length:]
            stop_decisions[idx] = self.stop_decisions.get(idx, -1)
            if len(input_ids_slice.shape) > 0 and stop_decisions[idx] == -1:
                decoded_slice = self.tokenizer.decode(input_ids_slice)
                for stop_token in self.stop_tokens:
                    if decoded_slice.endswith(stop_token):
                        stop_decisions[idx] = len(input_ids[idx])
                        self.stop_decisions[idx] = stop_decisions[idx]
            elif stop_decisions[idx] != -1:
                # Fill the rest of the sequence with the pad token
                # Change input_ids[idx][stop_decisions[idx]:] to the pad token
                for i in range(stop_decisions[idx], len(input_ids[idx])):
                    input_ids[idx][i] = self.pad_token_tensor
        return all([x != -1 for x in stop_decisions])

class Model(object):
    """Wrapper for a Llama model"""

    def __init__(self, name: str, device: str = None, training_args: TrainingArguments = None, **kwargs):
        assert name is not None, "Please provide a model name"
        assert device is None or device.startswith("cuda") or device.startswith("cpu"), "Please provide a valid device"
        self.name = name
        self.device = (device if device is not None else "cuda:0") if torch.cuda.is_available() else "cpu"
        self.training_args = training_args
        self._is_loaded = False
        self._kwargs = kwargs
        pass

    def load(self):
        if self._is_loaded:
            return
        model_args = {k: v for k, v in self._kwargs.items() if k in
                ["token"]
        }
        tokenizer_args = {k: v for k, v in self._kwargs.items() if k in
                ["token", "pad_token", "eos_token", "bos_token", "unk_token", "mask_token", "additional_special_tokens", "max_length"]
        }
        self._model : transformers.LlamaForCausalLM = transformers.AutoModelForCausalLM.from_pretrained(self.name, **model_args).to(self.device)
        self._tokenizer : transformers.LlamaTokenizer = transformers.AutoTokenizer.from_pretrained(self.name, **tokenizer_args)
        if "pad_token" not in tokenizer_args:
            tokenizer_args["pad_token"] = self._tokenizer.eos_token
        if "padding_side" not in tokenizer_args:
            tokenizer_args["padding_side"] = "left" # This is very import for the stop token logic
        self._tokenizer.pad_token = tokenizer_args["pad_token"]
        self._tokenizer.padding_side = tokenizer_args["padding_side"]
        self._is_loaded = True
        pass

    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def generate(self, inputs: typing.Union[typing.List[str], str], **kwargs) -> GenerationResults:
        assert self._is_loaded, "Model is not loaded"
        if isinstance(inputs, str):
            inputs = [inputs]
        if "stop_tokens" in kwargs:
            stop_tokens = kwargs["stop_tokens"]
        else:
            stop_tokens = [self._tokenizer.eos_token]
        if "auto_regressive_sequence_search" in kwargs:
            auto_regressive_sequence_search = kwargs["auto_regressive_sequence_search"]
            assert isinstance(auto_regressive_sequence_search, AutoRegressiveSequenceSearch), "Please provide a valid auto_regressive_sequence_search"
        else:
            auto_regressive_sequence_search = AutoRegressiveSequenceSearch.NucleusSampling
        tokenizer_args = {k: v for k, v in kwargs.items() if k in 
        ["max_length", "padding", "truncation"]}
        tokenized_input = self._tokenizer(
            inputs, 
            return_tensors="pt",
            **tokenizer_args).to(self.device)
        input_ids=tokenized_input['input_ids']
        attention_mask=tokenized_input['attention_mask']
        max_input_length = input_ids.shape[-1]
        stopping_criteria = StopOnTokens(stop_tokens, tokenizer=self._tokenizer, input_length=max_input_length, device=self.device)
        stopping_criteria = StoppingCriteriaList([stopping_criteria])
        if auto_regressive_sequence_search == AutoRegressiveSequenceSearch.NucleusSampling:
            generate_args = {k: v for k, v in kwargs.items() if k in
            ["do_sample", "top_k", "top_p", "num_return_sequences", "temperature", "max_new_tokens"]}
        elif auto_regressive_sequence_search == AutoRegressiveSequenceSearch.BeamSearch:
            generate_args = {k: v for k, v in kwargs.items() if k in
            ["num_beams", "num_return_sequences", "max_new_tokens"]}
        else:
            raise ValueError(f"Invalid auto_regressive_sequence_search: {auto_regressive_sequence_search}")
        if "num_return_sequences" not in generate_args:
            generate_args["num_return_sequences"] = 1
        num_return_sequences = generate_args["num_return_sequences"]
        generated_output = self._model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            eos_token_id=self._tokenizer.eos_token_id,
            return_dict_in_generate=True,
            **generate_args)
        target = generated_output["sequences"]
        skip_special_tokens = kwargs.get("skip_special_tokens", True)
        decoded_str = self._tokenizer.batch_decode(target, skip_special_tokens=skip_special_tokens)
        idx = 0
        generation_results = GenerationResults()
        return_full_text = kwargs.get("return_full_text", False)
        for text in inputs:
            generated_text = decoded_str[idx: idx + num_return_sequences]
            idx += num_return_sequences
            if not return_full_text:
                generated_text = [x[len(text):] for x in generated_text]
            generation_results.results.append(GenerationResult(input_text=text, generated_text=generated_text))
        return generation_results

    def train(self, **kwargs):
        pass

if __name__ == '__main__':
    # Model from Hugging Face hub
    import os
    import json
    import transformers
    assert os.path.exists(".secrets/huggingface_token.json"), "Please create a .secrets file with your HF token"
    model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    with open(".secrets/huggingface_token.json", "r") as f:
        token = json.load(f)["token"]
    device = "cuda:8"
    model = Model(model_name, device=device, token=token)
    main_prompt = "Do simple math problems (Answer only the number and use '[END]' to finish the response):\nQuestion: 2 + 2\nAnswer: 4\n[END]"
    with model:
        for response in model.generate(
                [
                    f"{main_prompt}\nQuestion: 4 + 5\nAnswer:",
                    f"{main_prompt}\nQuestion: 2 + 2\nAnswer:",
                    f"{main_prompt}\nQuestion: 3 + 3 * 3\nAnswer:",
                ],
                max_new_tokens=10,
                temperature=0.1, # Nucleus sampling
                do_sample=True, # Nucleus sampling
                top_k=5, # Nucleus sampling
                # num_beams=5, # Beam search
                num_return_sequences=5,
                stop_tokens=["[END]", model._tokenizer.eos_token],
                padding=True,
                #truncation=True,
                return_full_text=False):
            
            print("-" * 50)
            print(f"Prompt: \n{response.input_text}")
            for idx, result in enumerate(response.generated_text):
                print(f"Result [{idx + 1}]: {result}")
            print("-" * 50)
