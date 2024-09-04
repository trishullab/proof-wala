#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from proof_wala.llm_helpers.model import Model
from proof_wala.llm_helpers.llama2_chat_format import Llama2ChatFormat
from proof_wala.parsers.grammars.prompt_template_grammar import PromptGrammar


class LlamaChatBot(object):
    def __init__(self, 
        bot_name: str, 
        model_name: str, 
        system_prompt_filepath: str = None, 
        sample_conversation_filepath: str = None, 
        model_args: dict = None):
        self.bot_name = bot_name
        self.model_name = model_name
        self.system_prompt_filepath = system_prompt_filepath
        self.sample_conversation_filepath = sample_conversation_filepath
        self.model_args = model_args
        self._system_prompt = None
        self._sample_conversation = None
        self._prompt_grammar = PromptGrammar()
        self._messages = []
        self._chat_formatter = Llama2ChatFormat()
        pass

    def load(self):
        self.model = Model(self.model_name, training_args=None, log_folder=None, **self.model_args)
        self.model.load()
        self._system_prompt = self._prompt_grammar.get_main_message(self.system_prompt_filepath)
        self._sample_conversation = self._prompt_grammar.get_conv_messages(self.sample_conversation_filepath)
        self._messages.append(self._system_prompt)
        self._messages += self._sample_conversation
    
    def __call__(self, message: str) -> str:
        message = {"role": "user", "content": message}
        self._messages.append(message)
        self._respond()
        return self._messages[-1]["content"]
    
    def _respond(self) -> str:
        # Format the messages
        message, _ = self._chat_formatter(self._messages)
        # Call the model
        generated_response = self.model.generate(
                message,
                max_new_tokens=200,
                temperature=0.1, # Nucleus sampling
                do_sample=True, # Nucleus sampling
                top_k=5, # Nucleus sampling
                # num_beams=5, # Beam search
                num_return_sequences=1,
                stop_tokens=[self.model._tokenizer.eos_token],
                padding=True,
                #truncation=True,
                return_full_text=False,
                skip_special_tokens=True)
        response = generated_response[0].generated_text[0]
        self._messages.append({"role": "assistant", "content": response})

class LlamaConsoleChatBot(object):
    def __init__(self,
        bot_name: str, 
        model_name: str, 
        system_prompt_filepath: str = None, 
        sample_conversation_filepath: str = None, 
        model_args: dict = None):
        self.llama_chat_bot = LlamaChatBot(bot_name, model_name, system_prompt_filepath, sample_conversation_filepath, model_args)
        pass

    def load(self):
        self.llama_chat_bot.load()
        pass

    def run(self):
        while True:
            message = input(f"You>> ")
            if message == "exit":
                break
            response = self.llama_chat_bot(message)
            response = f"{self.llama_chat_bot.bot_name}>> {response}"
            print(response)
        pass

if __name__ == "__main__":
    import json
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    with open(".secrets/huggingface_token.json", "r") as f:
        hf_token = json.load(f)["token"]
    chatbot_name = "llama_chat_bot"
    problem_name = "simple_math"
    system_prompt_filepath = f"src/proof_wala/data/prompts/system/{problem_name}.md"
    sample_conversation_filepath = f"src/proof_wala/data/prompts/conversation/{problem_name}.md"
    console_chatbot = LlamaConsoleChatBot(
        chatbot_name, 
        model_name, 
        system_prompt_filepath, 
        sample_conversation_filepath, 
        model_args={"token": hf_token})
    console_chatbot.load()
    console_chatbot.run()