from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

device = "cpu"

model_name="EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

with open("prompt.txt", "r") as file:
    prompt = file.read()

print("Initial condition", prompt)

def parse_text(text):
    lines = text.split("\n")
    user_lines = []
    alice_lines = []
    current_speaker = ""
    for line in lines:
        line = line.strip()
        if line.startswith("user:"):
            current_speaker = "user"
            user_lines.append(line[5:].strip())
        elif line.startswith("Alice:"):
            current_speaker = "Alice"
            alice_lines.append(line[7:].strip())
        else:
            if current_speaker == "user":
                user_lines[-1] = user_lines[-1] + " " + line
            elif current_speaker == "Alice":
                alice_lines[-1] = alice_lines[-1] + " " + line
    return " ".join(user_lines), " ".join(alice_lines)



def chat_base(input):
  p = prompt + input
  input_ids = tokenizer(p, return_tensors="pt").input_ids
  gen_tokens = model.generate(
    input_ids, 
    do_sample=True, 
    temperature= 0.7,
    repetition_penalty= 0.7,
    min_length= 30, 
    max_length= 200)
  gen_text = tokenizer.batch_decode(gen_tokens)[0]
  #removes the prompt from the output
  result = gen_text[len(p):]
  print(">", result)
  user, alice = parse_text(result)
  print(">>", alice)
  return alice


def gradioInterface(message):
    history = gr.get_state() or []
    print(history)
    response = chat_base(message)
    history.append((message, response))
    gr.set_state(history)
    html = "<div class='chatbot'>"
    for user_msg, resp_msg in history:
        html += f"<div class='user_msg'>{user_msg}</div>"
        html += f"<div class='resp_msg'>{resp_msg}</div>"
    html += "</div>"
    return response

iface = gr.Interface(chat_base, gr.components.Textbox(label="Chat with Alice"), "text", allow_flagging="auto")
iface.launch()
