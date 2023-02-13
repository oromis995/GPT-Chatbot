from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

model_name="EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

with open("prompt.txt", "r") as file:
    prompt = file.read()

print("Initial condition", prompt)

def my_split(s, seps):
    res = [s]
    for sep in seps:
        s, res = res, []
        for seq in s:
            res += seq.split(sep)
    return res

def chat_base(input):
  p = prompt + input
  input_ids = tokenizer(p, return_tensors="pt").input_ids
  gen_tokens = model.generate(
    input_ids, 
    do_sample=True, 
    temperature=0.7,
    min_length= 30, 
    max_length= 250)
  gen_text = tokenizer.batch_decode(gen_tokens)[0]
  #print(gen_text)
  result = gen_text[len(p):]   
  print(">", result)
  result = my_split(result, [']', '\n'])[1]
  print(">>", result)
  if "Alice: " in result:
   result = result.split("Alice: ")[-1]
   print(">>>", result)
  return result


def chat(message):
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
