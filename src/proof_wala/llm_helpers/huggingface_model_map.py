from transformers import (
    T5ForConditionalGeneration,
    T5Config,
    T5Tokenizer,
)

def get_empty_config(config_class_name):
    if config_class_name == "T5Config":
        return T5Config()
    else:
        raise ValueError(f"Unknown config class name: {config_class_name}")

def get_model_class(config_class_name):
    if config_class_name == "T5Config":
        return T5ForConditionalGeneration
    else:
        raise ValueError(f"Unknown model class name: {config_class_name}")

def trim_model_kwargs(model_kwargs, config_class_name):
    if config_class_name == "T5Config":
        return {k: v for k, v in model_kwargs.items() if k in T5ForConditionalGeneration.__init__.__code__.co_varnames}
    else:
        raise ValueError(f"Unknown model class name: {config_class_name}")

def get_decoder_start_token_id(config_class_name, tokenizer: T5Tokenizer):
    if config_class_name == "T5Config":
        return tokenizer.pad_token_id
    else:
        raise ValueError(f"Unknown model class: {config_class_name}")