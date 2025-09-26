#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def extract_solution(solution_str: str):
    # Find all <answer>…</answer> blocks
    answers = re.findall(r"<answer>(.*?)</answer>", solution_str, flags=re.DOTALL)
    if not answers:
        return None, []
    # Take the last one, usually the actual LLM response
    answer_block = answers[-1]

    # Extract Label
    label_match = re.search(r"(NOT ENOUGH INFO|SUPPORT|REFUTE)", answer_block)
    pred_label = label_match.group(1) if label_match else None

    # More leniently extract all IDs within [[…]] and remove leading/trailing whitespace
    raw_ids = re.findall(r"\[\[\s*([^\]]+?)\s*\]\]", answer_block)
    # pred_evid_ids = [eid.strip() for eid in raw_ids]
    
    pred_evid_ids = [ normalize_id(eid.strip()) for eid in raw_ids ]

    return pred_label, pred_evid_ids


def truncate_from_sentence(eid: str) -> str:
    """
    Remove the part starting from '_sentence' and everything after it:
      'doc123_sentence_45' -> 'doc123'
      'my_doc_sentenceExtraInfo' -> 'my_doc'
      'nothingHere'            -> 'nothingHere'
    """
    _truncate_re = re.compile(r'_sentence.*$')
    return _truncate_re.sub('', eid)


def compute_reward(solution_str, ground_truth, format_score=0., challenge=None, data_source=None):
    """
    Calculate total reward:
      - label_reward: 1 point for correct label
      - evidence_reward: 0~1 points for evidence IoU
      - format_score: 0 or 1 points for format
      - total_score = label_reward + evidence_reward + format_score
    """
    # 1. First extract the main text (remove system prefix)
    preamble = "<|im_start|>assistant" # qwen
    # preamble = "assistant<|end_header_id|>" # llama
    body = solution_str.split(preamble, 1)[1] if preamble in solution_str else solution_str

    # 2. Extract predicted label and evidence from main text
    pred_label, pred_evid_ids = extract_solution(body)
    # pred_evid_ids = [truncate_from_sentence(eid) for eid in pred_evid_ids] # Remove suffix

    true_label = ground_truth.get('label')
    if true_label == "SUPPORTS" or true_label == "SUPPORTED":
        true_label = "SUPPORT"
    elif true_label == "REFUTES":
        true_label = "REFUTE"
    elif true_label == "NOT_SUPPORTED":
        true_label = "NOT_SUPPORT"
        
    true_evid_ids = ground_truth.get('evidence', [])
    if data_source == 'hover':
        true_evid_ids = [truncate_from_sentence(eid) for eid in true_evid_ids]

    # 3. Calculate label reward
    if data_source == 'hover':
        if true_label == 'SUPPORT' and pred_label == 'SUPPORT':
            label_reward = 1
        elif true_label == 'NOT_SUPPORT' and pred_label != 'SUPPORT':
            label_reward = 1
        else:
            label_reward = 0
    else:   
        label_reward = 1 if pred_label == true_label else 0

    # 4. Calculate evidence reward
    evidence_reward = 0.0
    set_pred = set(pred_evid_ids)
    set_true = set(true_evid_ids)
    if len(true_evid_ids) > 0:
        # Compute IoU
        intersection = set_pred & set_true
        union = set_pred | set_true
        if union:
            evidence_reward = len(intersection) / len(union)

    # 5. Calculate proof validity
    hit = len(set_pred & set_true)
    # Threshold: minimum number of hits to reach "half" (rounded up)
    threshold = (len(set_true) + 1) // 2
    if set_true.issubset(set_pred):  # If completely covered
        prove_valid = 1.0  
    elif hit >= threshold:  # If hit count exceeds threshold
        prove_valid = 0.5
    else:  # If no items hit
        prove_valid = 0.0

    # 6. Calculate evidence recall rate
    if set_true.issubset(set_pred):
        evidence_cover = 1.0
    else:
        evidence_cover = 0.0


    # 7. Calculate format score
    if pred_label is None:
        format_score = 0
    elif pred_label in ["SUPPORT", "REFUTE", "NOT ENOUGH INFO"]:
        format_score = 1

    # 8. Calculate verification accuracy
    if true_label in ["NOT ENOUGH INFO", "NOT_SUPPORT"] and label_reward == 1:
        veri_acc = 1
    elif label_reward == 1 and prove_valid == 1.0 and true_label in ["SUPPORT", "REFUTE"]:
        veri_acc = 1.0
    else:
        veri_acc = 0.0
    
    # 9. Calculate joint accuracy
    if true_label in ["NOT ENOUGH INFO", "NOT_SUPPORT"] and label_reward == 1:
        joint_acc = 1
    elif label_reward == 1 and set_pred == set_true and true_label in ["SUPPORT", "REFUTE"]:
        joint_acc = 1
    else:
        joint_acc = 0

    prediction = {"label": pred_label, "evidence": pred_evid_ids}

    # 10. Total score calculation
    if label_reward > 0:
        total_reward = 2 * label_reward + evidence_reward + format_score
    else:
        total_reward = 2 * label_reward + format_score

    return total_reward, label_reward, evidence_reward, format_score, prediction, evidence_cover, veri_acc, joint_acc




