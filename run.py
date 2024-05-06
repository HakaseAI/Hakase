from HakaseCore.llm.llama3 import LLama3

core = LLama3(accelerate_engine="mps", debug=True)
while 1:
    message = input(">> ")
    core.generate_text(message)
