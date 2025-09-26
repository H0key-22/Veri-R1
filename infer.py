import transformers
import torch
import random
from datasets import load_dataset
import requests

claim = "Karolina Kedzierska, a Swedish female Taekwondo practitioner, competed at 2008 Summer Olympics (an international multisport event held from 8 to 24 August 2008 in Beijing, China) where she lost to Natália Falavigna of Brazil in the Bronze Medal match."

prompt = f"""You are a claim-verification assistant. You MUST follow this protocol exactly:

<plan>…</plan>
- Once at the start: sketch your high-level strategy, such as claim decomposition, entity recognition, etc.

<search>…</search>
- When you need a fact: emit exactly this tag with your query.
- To make the most of your search turns, don’t repeat identical queries.
- You can search at most three times.

<information>
[[e_1]]: info1
[[e_2]]: info2
…
</information>
- You will be given claim related information in the format above.

<think>…</think>
- Use for every piece of reasoning; do not state your final verdict here.
- You must conduct reasoning inside <think> and </think> first every time you get new information.

<answer>
Label: SUPPORT / REFUTE / NOT ENOUGH INFO
Evidence: [[e_1]], [[e_3]], ...
</answer>
- Emit exactly once at the end, no extra text or tags.
- Evidence id such as e_1 will be replaced by real ids from the corpus. You must include useful real ids when answering
- Evidence outputs must strictly enforce the format [[e_i]], [[e_j]]…
- Answer Labels respectively stand for:
SUPPORT: The claim is consistent with the cited evidence and the evidence is sufficient to confirm the claim.
REFUTE: The claim contradicts the cited evidence and the evidence is sufficient to disprove the claim.
NOT ENOUGH INFO: The available evidence is insufficient to determine whether the claim is true or false.

- Note that the process should be: plan → (search → information → think) repeat until conclusion → answer

Verify the claim:
{claim}
"""

# Model ID and device setup
model_id = "/datadisk/model/qwen2.5-3b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

curr_eos = [151645, 151643] # for Qwen2.5 series models
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

# Initialize the tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

cnt = 0

if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print(prompt)
# Encode the chat-formatted prompt and move it to the correct device
while True:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate text with the stopping criteria
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )

    if outputs[0][-1].item() in curr_eos:
        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(output_text)
        break

    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
    if tmp_query:
        # print(f'searching "{tmp_query}"...')
        search_results = search(tmp_query)
    else:
        search_results = ''

    search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
    prompt += search_text
    cnt += 1
    print(search_text)
