import os
import base64
from typing import Union


class Logger:
    def __init__(self, log_dir, debug=True):
        self.set_dir(log_dir)
        self.debug = debug

    def set_dir(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def log_config(self, config):
        with open(
            os.path.join(self.log_dir, "config.yaml"), "w", encoding="utf-8"
        ) as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

    def store_chat(
        self,
        filename: str,
        prompt: str,
        answer: str,
        num_tokens: Union[int, None] = None,
    ) -> None:
        file_path = os.path.join(self.log_dir, filename)
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with open(os.path.join(self.log_dir, filename), "w", encoding="utf-8") as f:
            f.write("[Prompt]\n" + str(prompt[0]["text"]) + "\n")
            f.write("[Response]\n" + answer + "\n")
            f.write("[Tokens]\n" + str(num_tokens))

    def store_chats(self, dirname, prompts, answers):
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            with open(
                os.path.join(self.log_dir, dirname, f"{i}.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("[Prompt] " + prompt + "\n")
                f.write("[Response] " + answer + "\n")

    def store_img(self, filename: str, img_base64: str) -> None:
        file_path = os.path.join(self.log_dir, filename)
        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        img_data = base64.b64decode(img_base64)

        with open(os.path.join(self.log_dir, filename), "wb") as f:
            f.write(img_data)

    def store(self, filename: str, content: str) -> None:
        with open(os.path.join(self.log_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
