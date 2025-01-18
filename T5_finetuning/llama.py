from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
import torch

def greedy1(instruction, history, skill_set):
    return \
f"""
task overview:
your task involves selecting skills from a specific skill list provided to complete an instruction. it is crucial to choose only from this list.

task details:
instruction: complete the objective using only the skills from the provided list.
history: records the skills that have already been utilized from the list.

skill selection criteria:
1 point: choose the next skill to be performed from the list, based on existing history. if there's no history, pick the first relevant skill from the list. only one skill from this list will receive 1 point.
0.5 point: 0.5 points: 0.5 points are given to helpful skills in addition to the most necessary skills.
0 point: skills not on the list or already used do not receive point.

important emphasis:
strictly use skills from the provided skill list for decision-making and scoring. exclude skills not on this list from consideration.

scoring rules:
score each skill from the list. in your output, include only the skills that receive 0.5 or 1 point.
ensure one skill gets 1 point and several others get 0.5 point each, all from the provided list.
explicitly output a minimum of three skills, focusing on those with 0.5 and 1 point, while excluding those with 0 point.

execution:
identify and output skills from the provided skill list that are next in sequence. Assign 1 point to the most relevant skill and 1 point to other helpful skills, all from the list. skills with 0 point should not be included in your output.

additional condition:
In your response, do not use any capital letters.
Answer using the format skill_a:1\nskill_b:0.5 (Then give points for the remaining skills. Separate skills with \n)

Your Task:
Now, apply this methodology to the following scenario:
Instruction: {instruction}
Skill List: {skill_set}
History: {history}
"""


MODEL_NAME = "meta-llama/Llama-2-13b"

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")


prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]