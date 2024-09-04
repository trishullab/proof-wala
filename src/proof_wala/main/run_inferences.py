#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from proof_wala.llm_helpers.llama_chatbot import LlamaConsoleChatBot

def run_chat_bot(bot_name: str, model_name: str, system_prompt_filepath: str, sample_conversation_filepath: str):
    bot = LlamaConsoleChatBot(bot_name, model_name, system_prompt_filepath, sample_conversation_filepath)
    bot.load()
    bot.run()
    pass