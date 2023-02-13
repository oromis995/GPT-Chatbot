from transformers import GPTJForCausalLM, AutoTokenizer
import torch

device = "cpu"

model = GPTJForCausalLM.from_pretrained(

    "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float32, low_cpu_mem_usage=True

)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

with open("prompt.txt", "r") as file:
    prompt = file.read()

print("Initial condition", prompt)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    min_length=30,
    max_length=250,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]