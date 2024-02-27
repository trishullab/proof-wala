#!/usr/bin/env python3
import typing

class CodeT5ChatFormat(object):
    """
        prompt  = f"<<SYS>>\\n{syistem}\\n<</SYS>>\\n\\n{user_1}"
        prompt += f"<s>[INST] {prompt.strip()} [/INST] {answer_1.strip()} </s>"
        prompt += f"<s>[INST] {user_2.strip()} [/INST] {answer_2.strip()} </s>"
        prompt += f"<s>[INST] {user_3.strip()} [/INST]"
    """

    delimiter = "\n"
    def __init__(self):
        pass

    def __call__(self, messages) -> typing.Tuple[str, typing.Set[str]]:
        """
        messages are of the form
            messages = [
                {
                    "role": "user",
                    "content": "",
                }
            ]
        """
        # Collect all the system messages
        user_messages = []
        assistant_messages = []
        role_names = set()
        for message in messages:
            if message["role"] == "user":
                user_messages.append(message)
                if "name" in message:
                    role_names.add(message["name"])
            elif message["role"] == "assistant":
                assistant_messages.append(message)
                if "name" in message:
                    role_names.add(message["name"])
            else:
                raise ValueError(f"Unknown role: {message['role']}")
        user_prompt = self._format_user_assistant_messages(user_messages, assistant_messages)
        prompt = user_prompt
        return prompt, role_names

    def _format_user_assistant_messages(self, user_messages, assistant_messages):
        """
        user_messages is of the form
        """
        user_messages = [user_msg["content"] for user_msg in user_messages]
        user_messages = [f"{user_msg} {CodeT5ChatFormat.delimiter}" for user_msg in user_messages]
        assistant_messages = [assistant_msg["content"] for assistant_msg in assistant_messages]
        # Combine the messages one after the other
        messages = []
        idx = 0
        while idx < len(user_messages) or idx < len(assistant_messages):
            if idx < len(user_messages):
                messages.append(user_messages[idx])
            if idx < len(assistant_messages):
                messages.append(assistant_messages[idx])
            idx += 1
        message = "".join(messages)
        return message
    
if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": "We changed the direction of the project, but we don't have time to do it.",
        },
        {
            "role": "assistant",
            "content": "Too many changes do not have time to do it.",
        },
        {
            "role": "user",
            "content": "The pot is boiling, probably the water will spill.",
        }
    ]
    codet5_chat_format = CodeT5ChatFormat()
    prompt, role_names = codet5_chat_format(messages)
    print(prompt)
    print('='*50)
    print(role_names)
    print('-'*100)
    messages = [
        {
            "role": "user",
            "content": "We changed the direction of the project, but we don't have time to do it.",
        }
    ]
    codet5_chat_format = CodeT5ChatFormat()
    prompt, role_names = codet5_chat_format(messages)
    print(prompt)
    print('='*50)
    print(role_names)
    messages = [
        {
            "role": "user",
            "content": "We changed the direction of the project, but we don't have time to do it.",
        },
        {
            "role": "assistant", 
            "content": "Our idea seems to be scooped, don't know how to change direction now."
        }
    ]
    codet5_chat_format = CodeT5ChatFormat()
    prompt, role_names = codet5_chat_format(messages)
    print(prompt)
    print('='*50)
    print(role_names)