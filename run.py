from HakaseCore.llm.gpt4o import GPT4o
from HakaseCore.llm.llama3 import LLama3

core = LLama3(accelerate_engine="mps", debug=True)
# core = GPT4o()
while 1:
    message = input(">> ")
    msg = core.generate_text(message)
    print(msg)
