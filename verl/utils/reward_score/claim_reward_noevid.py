#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### evitrue

"""
Reward computation functions for claim verification.
Modified to use evidence IDs directly for IoU calculation.
"""
import re
import string
import unicodedata

def normalize_id(eid: str) -> str:
    # NFC: Normalize composed characters to single code points
    return unicodedata.normalize('NFC', eid)

def normalize_answer(s):
    """Normalize text: lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution(solution_str: str):
    # Find all <answer>…</answer> blocks
    answers = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL)
    if not answers:
        return None, []
    # Take the last one, usually the actual LLM response
    answer_block = answers[-1]

    # Extract Label
    label_match = re.search(r"Label:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO|SUPPORT|REFUTE)", answer_block)
    pred_label = label_match.group(1) if label_match else None

    # More leniently extract all IDs within [[…]] and remove leading/trailing whitespace
    pred_evid_ids = re.findall(r"\[\[\s*([^\]]+?)\s*\]\]", answer_block)
    # pred_evid_ids = [eid.strip() for eid in raw_ids]

    # pred_evid_ids = [ normalize_id(eid.strip()) for eid in raw_ids ]

    return pred_label, pred_evid_ids


def compute_format_score(solution_str: str) -> int:
    """
    Calculate format score (no nesting allowed, and </information> must be immediately followed by <think>):
      1. Only allow five structural tags: plan, search, information, think, answer.
      2. The first tag must be <plan></plan> and appear only once.
      3. The last tag must be <answer></answer> and appear only once.
      4. No nesting of structural tags allowed (except inside information blocks).
      5. </information> closure must be immediately followed by <think>.
    """
    tags = re.findall(r'<(/?)(\w+?)>', solution_str)
    allowed = {'plan', 'search', 'information', 'think', 'answer'}

    # Filter out all structural tags (including open and close)
    pairs = [(slash, name) for slash, name in tags if name in allowed]
    #
    # # Check plan open/close appears only once each, and starts with plan
    # if pairs[:2] != [('', 'plan'), ('/', 'plan')]:
    #     return 0
    # if sum(1 for slash, name in pairs if slash == '' and name == 'plan') != 1:
    #     return 0
    # if sum(1 for slash, name in pairs if slash == '/' and name == 'plan') != 1:
    #     return 0
    #
    # Check answer open/close appears only once each, and ends with answer
    if pairs[-2:] != [('', 'answer'), ('/', 'answer')]:
        return 0
    if sum(1 for slash, name in pairs if slash == '' and name == 'answer') != 1:
        return 0
    if sum(1 for slash, name in pairs if slash == '/' and name == 'answer') != 1:
        return 0

    current = None
    inside_info = False

    # Traverse all original tags, skip non-structural inner content and information block inner content
    for idx, (slash, name) in enumerate(tags):
        if name not in allowed:
            continue

        # Information block start
        if slash == '' and name == 'information':
            if current is not None:
                return 0
            inside_info = True
            current = 'information'
            continue

        # Information block end
        if slash == '/' and name == 'information':
            if current != 'information':
                return 0
            # Check if the next structural tag is <think>
            next_struct = None
            for j in range(idx + 1, len(tags)):
                if tags[j][1] in allowed:
                    next_struct = tags[j]
                    break
            if next_struct != ('', 'think'):
                return 0
            inside_info = False
            current = None
            continue

        # Inside information block, skip remaining checks
        if inside_info:
            continue

        # Ordinary structural tags: plan/search/think/answer
        if slash == '':
            if current is not None:
                # Has unclosed tag, nesting forbidden
                return 0
            current = name
        else:
            if current != name:
                # Mismatched closure
                return 0
            current = None

    # Ensure no residual unclosed tags
    return 1 if (current is None and not inside_info) else 0


def valid_generation(solution_str: str) -> int:
    """
    Find all <information> positions and determine if they are preceded only by whitespace and </search>.

    Parameters
    ----------
    solution_str : str
        Complete string to be checked.

    Returns
    -------
    int
        1 indicates correct format; 0 indicates format error.
    """

    import re

    # Use regex to find all <information> start indices at once
    info_starts = [match.start() for match in re.finditer(r"<information>", solution_str)]

    # For each <information> tag, check if it is preceded by </search> (only whitespace separation allowed)
    for start in info_starts:
        # Extract all content before <information> and remove trailing whitespace
        prefix = re.sub(r"\s+$", "", solution_str[:start])
        if not prefix.endswith("</search>"):
            return 0

    return 1


def compute_reward(solution_str, ground_truth, format_score=0., challenge=None, data_source=None):
    """
    Calculate total reward:
      - label_reward: 1 point for correct label
      - evidence_reward: 0~1 points for evidence IoU
      - format_score: 0 or 1 points for format
      - total_score = label_reward + evidence_reward + format_score (evidence reward not added when label is wrong)
    """
    # First extract the main text (remove system prefix)
    # preamble = "<|im_start|>assistant" # qwen
    preamble = "assistant<|end_header_id|>" # llama
    body = solution_str.split(preamble, 1)[1] if preamble in solution_str else solution_str

    # Extract predicted label and evidence from main text
    pred_label, pred_evid_ids = extract_solution(body)
    true_label = ground_truth.get('label')
    true_evid_ids = ground_truth.get('evidence', [])

    pred_evid_ids = [ normalize_id(eid.strip()) for eid in pred_evid_ids ]
    true_evid_ids = [ normalize_id(eid.strip()) for eid in true_evid_ids ]

    # Calculate label reward
    label_reward = 1 if pred_label == true_label else 0

    # Calculate evidence reward
    evidence_reward = 0.0
    set_pred = set(pred_evid_ids)
    set_true = set(true_evid_ids)
    if len(true_evid_ids) > 0:
        # Compute IoU
        intersection = set_pred & set_true
        union = set_pred | set_true
        if union:
            evidence_reward = len(intersection) / len(union)

    # Calculate proof validity
    hit = len(set_pred & set_true)
    # Threshold: minimum number of hits to reach "half" (rounded up)
    threshold = (len(set_true) + 1) // 2
    if set_true.issubset(set_pred):  # If completely covered
        prove_valid = 1.0  
    elif hit >= threshold:  # If hit count exceeds threshold
        prove_valid = 0.5
    else:  # If no items hit
        prove_valid = 0.0

    # Calculate evidence coverage
    if set_true.issubset(set_pred):
        evidence_cover = 1.0
    else:
        evidence_cover = 0.0


    # Calculate format score
    if valid_generation(body) == 0:
        format_score = 0
    else:
        format_score = compute_format_score(body)

    # Debug output
    # if format_score == 0:
    #     print("Error Solution_str:", solution_str)
    #     print("\n")
    print("Solution_str:", solution_str)
    print("\n")

    print("Pred_label:", pred_label)
    print("True_label:", true_label)
    print("Set_pred:", set_pred)
    print("Set_true:", set_true)
    print("Label Reward:", label_reward)
    print("Evidence Reward:", evidence_reward)
    print("Evidence Cover:", evidence_cover)
    print("Format Reward:", format_score)
    print("Prove Valid:", prove_valid)
    # Total score calculation
    if true_label in ("REFUTE", "SUPPORT"):
        total_reward = 2 * prove_valid * label_reward + format_score
    elif true_label == "NOT ENOUGH INFO":
        total_reward = 2 * label_reward + format_score
    
    if true_label == "NOT ENOUGH INFO" and pred_label == "NOT ENOUGH INFO":
        veri_acc = 1
    elif label_reward == 1 and prove_valid == 1.0 and true_label in ["SUPPORT", "REFUTE"]:
        veri_acc = 1.0
    else:
        veri_acc = 0.0

    if true_label in ["NOT ENOUGH INFO", "NOT_SUPPORT"] and label_reward == 1:
        joint_acc = 1
    elif label_reward == 1 and set_pred == set_true and true_label in ["SUPPORT", "REFUTE"]:
        joint_acc = 1
    else:
        joint_acc = 0

    print("Verification Accuracy:", veri_acc)
    print("Joint Accuracy:", joint_acc)
    print("Total Reward:", total_reward)

    prediction = {"label": pred_label, "evidence": pred_evid_ids}

    return total_reward, label_reward, evidence_reward, format_score, prediction, evidence_cover, veri_acc, joint_acc




