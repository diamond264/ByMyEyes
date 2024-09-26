from openai import OpenAI


def load_api_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


class ChatGPT:
    def __init__(self, version, api_key_path, max_tokens=4096, temperature=0):
        self.version = version
        self.api_key = load_api_key(api_key_path)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def ask(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.version,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=20,
        )

        response = chat_completion.choices[0].message.content
        return response


class LLM:
    def __init__(self, model: str, version: str, llm_path: str):
        self.name = model.lower()
        self.model = None

        if self.name == "chatgpt":
            self.model = ChatGPT(version, llm_path)
        else:
            raise ValueError(f"Unsupported language model: {self.name}")

    def generate(self, prompt: str) -> str:
        if self.name == "chatgpt":
            return self.model.ask(prompt)
        else:
            raise ValueError(f"Unsupported language model: {self.name}")
