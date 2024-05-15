import json
import os

from openai import OpenAI


class GPT4o:
    def __init__(self):
        self.prompt: list[dict[str, str]] = self.load_prompt()
        self.client = OpenAI(
            organization=os.environ.get("OPENAI_ORGANIZATION"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def load_prompt(self) -> list[dict[str, str]]:
        # Get Hakase Project Path
        prompt_path = (
            os.path.join(os.path.dirname(os.path.abspath(__file__)))
            + "/hakase_prompt.json"
        )
        with open(prompt_path, "r") as prompt_file:
            prompt = json.load(prompt_file)
        return prompt

    def generate_instruction(self, instruction: str) -> None:
        self.prompt.append({"role": "user", "content": f"{instruction}"})

    def generate_text(self, instruction: str) -> str:
        self.generate_instruction(instruction=instruction)
        completion = self.client.chat.completions.create(
            model="gpt-4o", messages=self.prompt
        )
        message = completion.choices[0].message.content
        return message
