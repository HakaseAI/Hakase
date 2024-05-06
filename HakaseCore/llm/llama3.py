import json
import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


class LLama3(object):
    def __init__(self, accelerate_engine: str = "cuda", debug: bool = False) -> None:
        self.prompt: list[dict[str, str]] = []
        self.model_id = "maum-ai/Llama-3-MAAL-8B-Instruct-v0.1"
        self.accelerate_engine = accelerate_engine
        if debug:
            match self.accelerate_engine:
                case "mps":
                    print("MPS Bulit : ", torch.backends.mps.is_built())
                    print("MPS available : ", torch.backends.mps.is_available())
                case "cuda":
                    print("CUDA Bulit : ", torch.backends.cuda.is_built())
                case "mkl":
                    print("MKL available : ", torch.backends.mkl.is_available())
                case _:
                    raise ValueError(
                        f"{accelerate_engine} is not a valid accelerate_engine"
                    )

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(
            self.accelerate_engine
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
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
        self.prompt = self.load_prompt()
        self.prompt.append({"role": "user", "content": f"{instruction}"})

    def generate_text(self, instruction: str) -> str:
        self.generate_instruction(instruction=instruction)
        inputs = self.tokenizer.apply_chat_template(
            self.prompt, tokenize=True, return_tensors="pt"
        ).to(self.accelerate_engine)
        outputs = self.model.generate(
            inputs,
            streamer=self.streamer,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        print(outputs)
