import os
import torch
import typing
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

class Model(object):
    def __init__(self, name: str, device: str = None, training_args: TrainingArguments = None):
        assert name is not None, "Please provide a model name"
        assert device is None or device.startswith("cuda") or device.startswith("cpu"), "Please provide a valid device"
        self.name = name
        self.device = (device if device is not None else "cuda:0") if torch.cuda.is_available() else "cpu"
        self.training_args = training_args
        self._is_loaded = False
        pass

    def load(self, **kwargs):
        self._model = AutoModelForCausalLM.from_pretrained(self.name, **kwargs).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.name, **kwargs)
        self._is_loaded = True
        pass

    def generate(self, input: str, **kwargs):
        assert self._is_loaded, "Model is not loaded"
        tokenized_input = self._tokenizer(input, return_tensors="pt").to(self.device)
        input_ids=tokenized_input['input_ids']
        attention_mask=tokenized_input['attention_mask']
        generated_output = self._model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            eos_token_id=self._tokenizer.eos_token_id,
            **kwargs)
        response = self._tokenizer.batch_decode(generated_output, skip_special_tokens=True)
        return response

    def train(self, **kwargs):
        pass

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: typing.List[torch.LongTensor], input_length: int, tokenizer: AutoTokenizer):
        self.stop_token_ids = stop_token_ids
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.stop_tokens = [tokenizer.decode(x) for x in stop_token_ids]
        self.max_stop_token_id_length = max([len(x) for x in self.stop_token_ids])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] <= self.input_length:
            return False
        input_ids_slice = input_ids[0][-self.max_stop_token_id_length:]
        if len(input_ids_slice.shape) > 0:
            decoded_slice = self.tokenizer.decode(input_ids_slice)
            for stop_token in self.stop_tokens:
                if decoded_slice.endswith(stop_token):
                    return True
        return False


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
    # model = Model(model_name)
    # model.load(token=token)
    # for response in model.generate("Translate the following sentence into German: \"I am a student.\"", 
    #         max_new_tokens=500, 
    #         temperature=0.75,
    #         do_sample=True,
    #         top_k=10,
    #         num_return_sequences=1):
    #     print(response)
    device = "cuda:2"
    tokenizer : transformers.LlamaTokenizer = transformers.LlamaTokenizer.from_pretrained(model_name, token=token)
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, token=token).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device=device,
        tokenizer=tokenizer
    )
    prompt = "Do simple math problems (End the answer with a '[END]'): Question: 2 + 2 = 5 [END] Question: 4 + 5 = "
    stop_token_ids = [tokenizer.encode(stop_word, add_special_tokens=False) 
            for stop_word in ["[END]", tokenizer.eos_token]]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    stop_token_ids = [x for x in stop_token_ids] # remove first token
    # Decode stop token ids and remove special tokens
    stop_tokens = [tokenizer.decode(x) for x in stop_token_ids]
    print(f"Stop tokens: {stop_tokens}")
    #stop_token_ids = [x[1:] if len(x.shape) > 0 else x for x in stop_token_ids]
    print(f"Stop words: {stop_token_ids}")
    while prompt != "exit":
        prompt_token_ids = tokenizer.encode(prompt)
        stopping_criteria = StopOnTokens(stop_token_ids, input_length=len(prompt_token_ids), tokenizer=tokenizer)
        stopping_criteria = StoppingCriteriaList([stopping_criteria])
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            stopping_criteria=stopping_criteria,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
            max_new_tokens=250
        )
        print("-" * 50)
        print("Responses:")
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
        print("-" * 50)
        prompt = input("Prompt:\n")
