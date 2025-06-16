THINK_ANSWER_TEMPLATE = """
A conversation between User and Assistant. The user asks a question, and the assistant solves
it. The assistant first thinks about the reasoning process in the mind and then provides the
user with the answer. During thinking, the assistant can invoke the wikipedia search tool
to search for fact information about specific topics if needed. The reasoning process and
answer are enclosed within <think> </think> and <answer> </answer> tags respectively,
and the search query and result are enclosed within <search> </search> and <result>
</result> tags respectively. For example, <think> This is the reasoning process. </think>
<search> search query here </search> <result> search result here </result> <think>
This is the reasoning process. </think> <answer> The final answer is \boxed{answer here}
</answer>. In the last part of the answer, the final exact answer is enclosed within \boxed{}
with latex format.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare the model input
prompt = "Who won the 2018 presidential election in the country where the political party of Martín Ramírez Pech operates?"
messages = [
    {"role": "system", "content": THINK_ANSWER_TEMPLATE},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True, # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)