import json
import os.path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


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

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model_4bit = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=bnb_config, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, add_special_tokens=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.pipe = pipeline(
            "text-generation", model=self.model_4bit, tokenizer=self.tokenizer
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
        prompt = self.pipe.tokenizer.apply_chat_template(
            self.prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=100,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )
        print(outputs[0]["generated_text"][len(prompt) :])
