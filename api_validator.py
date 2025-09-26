#!/usr/bin/env python3
"""
API Validator for Multi-turn Dialogue Evaluation (Corrected Multi-threaded Version)

This module provides a validate function that replicates the exact evaluation standards
from VERL framework's _validate method, with proper turn-by-turn synchronization.
"""

import os
import json
import csv
import time
import logging
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# Import VERL reward functions
import sys
sys.path.append('.')
from verl.utils.reward_score import claim_reward_eva

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system_prompt = """
You are a claim-verification assistant. You MUST follow this protocol exactly:

<plan>…</plan>
- Once at the start: sketch your high-level strategy, such as claim decomposition, entity recognition, etc.

<search>…</search>
- When you need a fact: emit exactly this tag with your query.
- To make the most of your search turns, don't repeat identical queries.
- CRITICAL: You can search at most THREE times. After the third search, you MUST provide your final answer.

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
- IMPORTANT: After each search, you will receive information and MUST continue the conversation.
- CRITICAL: You must always respond after receiving information. Do not stop generating after a search.
- FINAL WARNING: After your third search, you MUST provide your final answer. Do not attempt a fourth search."""

@dataclass
class APIConfig:
    """Configuration for API validation with proper turn synchronization"""
    api_key: str
    model: str = "deepseek-reasoner"  # Configurable model name
    max_turns: int = 4
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    api_base: str = "https://api.deepseek.com/v1" 
    search_url: str = "http://127.0.0.1:8000/retrieve"
    search_topk: int = 3
    save_csv: bool = True
    save_jsonl: bool = True
    output_dir: str = "./validation_results"
    num_examine: int = 1
    
    # Multi-threading configuration
    max_workers: int = 32  # Maximum number of concurrent API calls per turn
    batch_size: int = 56   # Batch size for processing
    rate_limit_delay: float = 0.1  # Delay between API calls
    
    # Retry configuration
    max_retries: int = 2  # Maximum number of retries for API calls
    api_timeout: int = 15  # Timeout for API calls in seconds
    parallel_timeout: int = 60  # Timeout for parallel processing in seconds


class ThreadSafeCounter:
    """Thread-safe counter for tracking API calls and rate limiting"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0
        self._last_call_time = 0
    
    def increment(self):
        with self._lock:
            self._count += 1
    
    def get_count(self):
        with self._lock:
            return self._count
    
    def update_last_call_time(self, timestamp):
        with self._lock:
            self._last_call_time = timestamp
    
    def get_last_call_time(self):
        with self._lock:
            return self._last_call_time


class APIValidator:
    """API-based validator with proper turn-by-turn synchronization"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.api_counter = ThreadSafeCounter()
        
        # Initialize reward functions mapping
        self.reward_functions = claim_reward_eva.compute_reward
        
        # Initialize search count for tracking search operations
        self.search_count = defaultdict(int)
        
        # Initialize API failure counters
        self.api_failure_counter = ThreadSafeCounter()
        self.api_timeout_counter = ThreadSafeCounter()
        self.api_connection_error_counter = ThreadSafeCounter()
        self.api_other_error_counter = ThreadSafeCounter()

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _call_api(self, messages: List[Dict[str, str]], max_retries: int = 2) -> str:
        """Call API with rate limiting, error handling, and retry mechanism"""
        
        # Rate limiting
        current_time = time.time()
        last_call_time = self.api_counter.get_last_call_time()
        if current_time - last_call_time < self.config.rate_limit_delay:
            time.sleep(self.config.rate_limit_delay - (current_time - last_call_time))
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                self.api_counter.increment()
                self.api_counter.update_last_call_time(time.time())
                
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=30  # 30 second timeout
                )
                return response.choices[0].message.content.strip()
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                self.api_connection_error_counter.increment()
                
                if attempt < max_retries:
                    logger.info(f"Retrying connection error in 1 second...")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Connection error after {max_retries + 1} attempts, returning default answer")
                    self.api_failure_counter.increment()
                    return "<answer>\nLabel: NOT ENOUGH INFO\nEvidence: \n</answer>"
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                self.api_timeout_counter.increment()
                
                if attempt < max_retries:
                    logger.info(f"Retrying timeout error in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    logger.error(f"Timeout error after {max_retries + 1} attempts, returning default answer")
                    self.api_failure_counter.increment()
                    return "<answer>\nLabel: NOT ENOUGH INFO\nEvidence: \n</answer>"
                    
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                self.api_other_error_counter.increment()
                
                if attempt < max_retries:
                    logger.info(f"Retrying unexpected error in 1 second...")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Unexpected error after {max_retries + 1} attempts, returning default answer")
                    self.api_failure_counter.increment()
                    return "<answer>\nLabel: NOT ENOUGH INFO\nEvidence: \n</answer>"

    def _extract_action_and_content(self, response: str) -> Tuple[Optional[str], str]:
        """Extract action and content from API response"""
        search_pattern = r'<search>(.*?)</search>'
        answer_pattern = r'<answer>(.*?)</answer>'
        
        search_match = re.search(search_pattern, response, re.DOTALL)
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        
        if search_match:
            return 'search', search_match.group(1).strip()
        elif answer_match:
            return 'answer', answer_match.group(1).strip()
        else:
            return None, ''
    
    def batch_search(self, queries: List[str] = None, data_sources: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
            data_sources: data sources for each query
        Returns:
            search results which is concatenated into a string
        """
        if not queries:
            return []
            
        results = self._batch_search(queries)['result']
        
        return [
            self._passages2string(result, ds)
            for result, ds in zip(results, data_sources)
        ]

    def _batch_search(self, queries):
        """Perform batch search using the retrieval API"""
        payload = {
            "queries": queries,
            "topk": 20,
            "return_scores": True
        }
        
        try:
            response = requests.post(self.config.search_url, json=payload, timeout=30)
            response.raise_for_status()
            response_json = response.json()
            
            return response_json
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return {'result': [[] for _ in queries]}

    def _passages2string(self, retrieval_result, data_source=None):
        """Convert retrieval results to formatted string"""
        topk = self.config.search_topk
        max_tokens = 1024  # Maximum tokens for the final string
        total_tokens = 0

        # ① Filter
        if data_source in ('feverous', 'fever'):
            filtered = [d for d in retrieval_result if '<sent_id=' in d['document']['id']]
        elif data_source in ('hover', 'exfever', 'nq'):
            filtered = [d for d in retrieval_result if '<sent_id=' not in d['document']['id']]
        else:
            filtered = retrieval_result

        # ② Supplement top-k
        if len(filtered) < topk:
            extra = [d for d in retrieval_result if d not in filtered]
            filtered.extend(extra[:topk - len(filtered)])

        selected = filtered[:topk]
        is_doc = data_source in ('hover', 'exfever')
        pieces = []

        for doc in selected:
            raw_id = doc['document']['id']
            # Starting index
            m = re.search(r'<sent_id=(\d+)', raw_id)
            start_idx = int(m.group(1)) if m else 0
            # Remove <sent_id=...>
            doc_id = re.sub(r'<sent_id=[^>]*>', '', raw_id)

            # Remove title
            text = doc['document']['text']
            first, *rest = text.split('\n', 1)
            body = rest[0] if rest and first.startswith('"') and first.endswith('"') else text

            if is_doc:
                entry = f"[[{doc_id}]]:{body}\n"
                # Simple token estimation (by character count)
                if (remain := max_tokens - total_tokens) <= 0:
                    break
                if len(entry) > remain * 4:  # Rough estimate: 1 token ≈ 4 characters
                    entry = entry[:remain * 4]
                pieces.append(entry)
                total_tokens += len(entry) // 4
            else:
                for i, sent in enumerate(seg for seg in body.split('[SEP]') if seg.strip()):
                    entry = f"[[{doc_id}_sentence_{start_idx+i}]]:{sent.strip()}\n"
                    # Simple token estimation (by character count)
                    if (remain := max_tokens - total_tokens) <= 0:
                        break
                    if len(entry) > remain * 4:  # Rough estimate: 1 token ≈ 4 characters
                        entry = entry[:remain * 4]
                        pieces.append(entry)
                        total_tokens += remain
                        break
                    pieces.append(entry)
                    total_tokens += len(entry) // 4

        return ''.join(pieces)
    
    def _execute_predictions(self, responses: List[str], data_sources: List[str], 
                           active_mask: List[bool]) -> Tuple[List[str], List[bool], List[int], List[int]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            responses: List of action predictions
            data_sources: List of data sources
            active_mask: List of active flags
            
        Returns:
            Tuple of (observations, dones, valid_actions, is_searches)
        """
        actions, contents = [], []
        
        # Extract actions and contents
        for response in responses:
            action, content = self._extract_action_and_content(response)
            actions.append(action)
            contents.append(content)
        
        next_obs, dones, valid_action, is_search = [], [], [], []
        max_search = self.config.max_turns - 1

        allowed_idx = [
            idx for idx, (a, active) in enumerate(zip(actions, active_mask))
            if active and a == 'search' and self.search_count[idx] < max_search
        ]

        # Prepare queries and data sources for these indices
        search_queries = [contents[idx] for idx in allowed_idx]
        search_data_sources = [data_sources[idx] for idx in allowed_idx]
        
        # Perform search
        search_results = self.batch_search(search_queries, search_data_sources)
        print("Search queries:", len(search_queries))
        assert len(search_results) == len(allowed_idx)
        
        # Create a mapping from allowed_idx to search results
        search_result_map = {idx: result for idx, result in zip(allowed_idx, search_results)}
        
        for i, (action, active) in enumerate(zip(actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(True)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(True)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    if self.search_count[i] >= max_search:
                        next_obs.append(f"\n\nThere are no search turns left. I must think and then answer immediately.\n")
                        dones.append(False)
                        valid_action.append(0)  # Treat as invalid action
                        is_search.append(0)
                    else:
                        self.search_count[i] += 1
                        info = search_result_map[i].strip()
                        remaining = max_search - self.search_count[i]
                        next_obs.append(f"\n\n<information>\n{info}\n**{remaining} search turns left.**\n</information>\n\nYour response each turn must end with </search> or </answer>")
                        dones.append(False)
                        valid_action.append(1)
                        is_search.append(1)
                else:
                    next_obs.append(f'\n\nMy previous action is invalid. I must strictly follow the protocol. Let me try again.\n\n')
                    dones.append(False)
                    valid_action.append(0)
                    is_search.append(0)
            if dones[i]:
                self.search_count[i] = 0
        
        # All search results have been processed
            
        return next_obs, dones, valid_action, is_search
    
    def _generate_turn_parallel(self, conversations: List[List[Dict]], data_sources: List[str]) -> List[str]:
        """Generate responses for one turn in parallel with improved error handling"""
        # Prepare arguments for parallel processing
        args_list = [(i, conv) for i, conv in enumerate(conversations)]
        
        # Initialize results list with default responses
        results = ["<answer>\nLabel: NOT ENOUGH INFO\nEvidence: \n</answer>"] * len(conversations)
        
        # Process conversations in parallel for this turn
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._call_api, args[1]): args[0] 
                for args in args_list
            }
            
            # Collect results as they complete with timeout handling
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    # Add timeout for individual future completion
                    response = future.result(timeout=60)  # 60 second timeout
                    results[idx] = response
                except Exception as e:
                    logger.error(f"Error in parallel API call for index {idx}: {e}")
                    # Keep default response for this index
                    results[idx] = "<answer>\nLabel: NOT ENOUGH INFO\nEvidence: \n</answer>"
        
        return results
    
    def _run_multi_turn_generation_synchronized(self, prompts: List[str], data_sources: List[str]) -> List[str]:
        """Run multi-turn generation with proper turn-by-turn synchronization"""
        batch_size = len(prompts)
        
        # Initialize conversations
        conversations = []
        for prompt in prompts:
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Verify the following claim: {prompt}"}
            ])
        
        # Initialize state tracking (similar to VERL's active_mask)
        active_mask = [True] * batch_size
        final_responses = [""] * batch_size
        
        # Reset search count for new batch
        self.search_count.clear()
        
        logger.info(f"Starting synchronized multi-turn generation for {batch_size} conversations")
        
        # Main generation loop (turn-by-turn, like VERL)
        for turn in range(self.config.max_turns):
            if not any(active_mask):
                logger.info(f"All conversations completed at turn {turn}")
                break
            
            logger.info(f"Turn {turn + 1}/{self.config.max_turns}: {sum(active_mask)} active conversations")
            
            # 1. Generate responses for all active conversations in parallel
            active_conversations = [conv for i, conv in enumerate(conversations) if active_mask[i]]
            active_indices = [i for i, active in enumerate(active_mask) if active]
            
            if active_conversations:
                responses = self._generate_turn_parallel(active_conversations, data_sources)
                
                # 2. Update conversations with responses
                for idx, response in zip(active_indices, responses):
                    conversations[idx].append({"role": "assistant", "content": response})
                
                # 3. Execute predictions (environment step)
                active_data_sources = [data_sources[i] for i in active_indices]
                
                next_obs, dones, valid_action, is_search = self._execute_predictions(
                    responses, active_data_sources, [True] * len(active_conversations)
                )
                
                # 4. Update state for next turn
                for i, (idx, done, obs) in enumerate(zip(active_indices, dones, next_obs)):
                    active_mask[idx] = not done
                    
                    if done:
                        # Extract complete conversation history
                        conversation_history = []
                        for msg in conversations[idx]:
                            if msg["role"] == "assistant":
                                conversation_history.append(msg["content"])
                            elif msg["role"] == "user" and "Verify the following claim:" not in msg["content"]:
                                # Include user messages except the initial prompt
                                conversation_history.append(msg['content'])
                        
                        # Combine all messages to show complete reasoning process
                        final_responses[idx] = "\n\n".join(conversation_history)
                    else:
                        # Add observation for next turn
                        if obs:
                            conversations[idx].append({"role": "user", "content": obs})
            
            logger.info(f"Turn {turn + 1} completed: {sum(active_mask)} conversations still active")
        
        # Handle any remaining active conversations
        for i, active in enumerate(active_mask):
            if active and not final_responses[i]:
                logger.warning(f"Index {i}: Conversation still active after {self.config.max_turns} turns, forcing completion")
                
                # Check if the last response contains an answer
                last_response = ""
                for msg in reversed(conversations[i]):
                    if msg["role"] == "assistant":
                        last_response = msg["content"]
                        break
                
                if "<answer>" not in last_response:
                    logger.warning(f"Index {i}: No answer found in last response, generating forced completion")
                    # Add a prompt to force an answer
                    conversations[i].append({
                        "role": "user", 
                        "content": "\n\nYou must provide your final answer now. Based on the information you have, please give your conclusion in the format:\n\n<answer>\nLabel: SUPPORT / REFUTE / NOT ENOUGH INFO\nEvidence: [[evidence_id]]\n</answer>\n\n"
                    })
                    
                    try:
                        forced_response = self._call_api(conversations[i])
                        conversations[i].append({"role": "assistant", "content": forced_response})
                        
                        # Extract complete conversation history including the forced response
                        conversation_history = []
                        for msg in conversations[i]:
                            if msg["role"] == "assistant":
                                conversation_history.append(msg["content"])
                            elif msg["role"] == "user" and "Verify the following claim:" not in msg["content"]:
                                conversation_history.append(msg['content'])
                        
                        final_responses[i] = "\n\n".join(conversation_history)
                    except Exception as e:
                        logger.error(f"Failed to get forced response for index {i}: {e}")
                        # Extract existing conversation history
                        conversation_history = []
                        for msg in conversations[i]:
                            if msg["role"] == "assistant":
                                conversation_history.append(msg["content"])
                            elif msg["role"] == "user" and "Verify the following claim:" not in msg["content"]:
                                conversation_history.append(msg['content'])
                        
                        if conversation_history:
                            final_responses[i] = "\n\n".join(conversation_history)
                        else:
                            final_responses[i] = "<answer>\nLabel: NOT ENOUGH INFO\nEvidence: \n</answer>"
                else:
                    # Extract complete conversation history for remaining active conversations
                    conversation_history = []
                    for msg in conversations[i]:
                        if msg["role"] == "assistant":
                            conversation_history.append(msg["content"])
                        elif msg["role"] == "user" and "Verify the following claim:" not in msg["content"]:
                            conversation_history.append(msg['content'])
                    
                    if conversation_history:
                        final_responses[i] = "\n\n".join(conversation_history)
                    else:
                        final_responses[i] = "<answer>\nLabel: NOT ENOUGH INFO\nEvidence: \n</answer>"
        
        return final_responses
    
    def _process_batch_synchronized(self, batch_data: List[Dict[str, Any]]) -> Tuple[List, List, List, List, List, List, List, List, List, List, List[str]]:
        """Process a batch of data with synchronized multi-turn generation"""
        # Extract prompts and metadata
        prompts = [item['question'] for item in batch_data]
        data_sources = [item.get('data_source', 'fever') for item in batch_data]
        
        # Run synchronized multi-turn generation
        responses = self._run_multi_turn_generation_synchronized(prompts, data_sources)
        
        # Process results
        reward_tensor_lst = []
        acc_tensor_lst = []
        evidence_tensor_lst = []
        format_tensor_lst = []
        evid_cover_tensor_lst = []
        veri_acc_tensor_lst = []
        joint_acc_tensor_lst = []
        data_source_lst = []
        challenge_lst = []
        true_label_lst = []
        pred_label_lst = []
        
        for item, response in zip(batch_data, responses):
            sample_id = item['id']
            golden_answers = item['golden_answers']
            challenge = item.get('ability', 'unknown')
            data_source = item.get('data_source', 'fever')
            
            # Select reward function
            reward_fn = self.reward_functions
            
            # Compute metrics using VERL reward function
            try:
                total_reward, label_reward, evidence_reward, format_reward, pred_label, evidence_cover, veri_acc, joint_acc = reward_fn(
                    solution_str=response,
                    ground_truth=golden_answers,
                    format_score=0.0,
                    challenge=challenge,
                    data_source=data_source
                )
            except Exception as e:
                logger.warning(f"Error computing reward for sample {sample_id}: {e}")
                total_reward = label_reward = evidence_reward = format_reward = evidence_cover = veri_acc = joint_acc = 0.0
                pred_label = None
            
            # Store metrics
            reward_tensor_lst.append([total_reward])
            acc_tensor_lst.append([label_reward])
            evidence_tensor_lst.append([evidence_reward])
            format_tensor_lst.append([format_reward])
            evid_cover_tensor_lst.append([evidence_cover])
            veri_acc_tensor_lst.append([veri_acc])
            joint_acc_tensor_lst.append([joint_acc])
            
            data_source_lst.append([data_source])
            challenge_lst.append([challenge])
            true_label_lst.append([golden_answers.get('label', '')])
            pred_label_lst.append([pred_label])
        
        return (reward_tensor_lst, acc_tensor_lst, evidence_tensor_lst, format_tensor_lst, 
                evid_cover_tensor_lst, veri_acc_tensor_lst, joint_acc_tensor_lst, data_source_lst, challenge_lst, 
                true_label_lst, pred_label_lst, responses)
    
    def validate(self, data: List[Dict[str, Any]], experiment_name: str = "api_validation") -> Dict[str, Any]:
        """
        Main validation function with proper turn-by-turn synchronization
        
        Args:
            data: List of validation samples
            experiment_name: Name for output files
            
        Returns:
            Dictionary containing all validation metrics
        """
        logger.info(f"Starting synchronized validation with {len(data)} samples")
        logger.info(f"Using {self.config.max_workers} workers per turn")
        
        # Initialize metric collection
        reward_tensor_lst = []
        acc_tensor_lst = []
        evidence_tensor_lst = []
        format_tensor_lst = []
        evid_cover_tensor_lst = []
        veri_acc_tensor_lst = []
        joint_acc_tensor_lst = []
        data_source_lst = []
        challenge_lst = []
        true_label_lst = []
        pred_label_lst = []
        all_results = []
        
        # Process data in batches
        total_batches = (len(data) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in range(0, len(data), self.config.batch_size):
            batch_data = data[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_data)} samples)")
            
            # Process batch with synchronized generation
            (batch_reward, batch_acc, batch_evidence, batch_format, 
             batch_evid_cover, batch_veri_acc, batch_joint_acc, batch_data_source, batch_challenge, 
             batch_true_label, batch_pred_label, batch_responses) = self._process_batch_synchronized(batch_data)
            
            # Extend main lists
            reward_tensor_lst.extend(batch_reward)
            acc_tensor_lst.extend(batch_acc)
            evidence_tensor_lst.extend(batch_evidence)
            format_tensor_lst.extend(batch_format)
            evid_cover_tensor_lst.extend(batch_evid_cover)
            veri_acc_tensor_lst.extend(batch_veri_acc)
            joint_acc_tensor_lst.extend(batch_joint_acc)
            data_source_lst.extend(batch_data_source)
            challenge_lst.extend(batch_challenge)
            true_label_lst.extend(batch_true_label)
            pred_label_lst.extend(batch_pred_label)
            
            # Save detailed results
            for j, item in enumerate(batch_data):
                all_results.append({
                    'id': str(item['id']),
                    'claim': str(item['question']),
                    'accuracy': f"{float(batch_acc[j][0])*100:.2f}%",
                    'verification_accuracy': f"{float(batch_veri_acc[j][0])*100:.2f}%",
                    'joint_accuracy': f"{float(batch_joint_acc[j][0])*100:.2f}%",
                    'prediction': str(batch_pred_label[j][0]),
                    'ground_truth': item['golden_answers'].get('label', ''),
                    'response': str(batch_responses[j] if j < len(batch_responses) else ""),
                    'evidence_score': f"{float(batch_evidence[j][0]):.4f}",
                    'data_source': str(batch_data_source[j][0]),
                    'challenge': str(batch_challenge[j][0]),
                    'total_reward': float(batch_reward[j][0]),
                    'format_reward': f"{float(batch_format[j][0]):.4f}",
                    'evidence_cover': float(batch_evid_cover[j][0]),
                })
        
        # Aggregate metrics (same as before)
        reward_tensor = np.array([rw[0] for rw in reward_tensor_lst])
        acc_tensor = np.array([acc[0] for acc in acc_tensor_lst])
        evidence_tensor = np.array([ev[0] for ev in evidence_tensor_lst])
        format_tensor = np.array([fm[0] for fm in format_tensor_lst])
        evid_cover_tensor = np.array([ec[0] for ec in evid_cover_tensor_lst])
        veri_acc_tensor = np.array([va[0] for va in veri_acc_tensor_lst])
        joint_acc_tensor = np.array([ja[0] for ja in joint_acc_tensor_lst])
        data_sources = np.array([ds[0] for ds in data_source_lst])
        challenge_types = np.array([ch[0] for ch in challenge_lst])
        true_label_types = np.array([tl[0] for tl in true_label_lst])
        pred_label_types = np.array([pl[0] for pl in pred_label_lst])
        

        
        # Compute per-data-source metrics
        data_source_reward = defaultdict(list)
        data_source_acc = defaultdict(list)
        data_source_evidence = defaultdict(list)
        data_source_format = defaultdict(list)
        data_source_evid_cover = defaultdict(list)
        data_source_veri_acc = defaultdict(list)
        data_source_joint_acc = defaultdict(list)
        for i in range(len(reward_tensor)):
            data_source = data_sources[i]
            data_source_reward[data_source].append(reward_tensor[i])
            data_source_acc[data_source].append(acc_tensor[i])
            data_source_evidence[data_source].append(evidence_tensor[i])
            data_source_format[data_source].append(format_tensor[i])
            data_source_evid_cover[data_source].append(evid_cover_tensor[i])
            data_source_veri_acc[data_source].append(veri_acc_tensor[i])
            data_source_joint_acc[data_source].append(joint_acc_tensor[i])
        # Build metric dictionary
        metric_dict = {}
        
        # Overall metrics
        metric_dict['val/test_overall_acc'] = float(acc_tensor.mean())
        metric_dict['val/test_overall_veri_acc'] = float(veri_acc_tensor.mean())
        metric_dict['val/test_overall_joint_acc'] = float(joint_acc_tensor.mean())
        metric_dict['val/test_overall_evidence'] = float(evidence_tensor.mean())
        metric_dict['val/test_overall_format'] = float(format_tensor.mean())
        
        # Per-label metrics
        for lb in np.unique(true_label_types):
            idx = (true_label_types == lb)
            metric_dict[f'val/test_acc/label/{lb}'] = float(acc_tensor[idx].mean())
            metric_dict[f'val/test_veri_acc/label/{lb}'] = float(veri_acc_tensor[idx].mean())
            metric_dict[f'val/test_joint_acc/label/{lb}'] = float(joint_acc_tensor[idx].mean())
        
        # Print results
        logger.info("=" * 60)
        logger.info("VALIDATION RESULTS (Synchronized)")
        logger.info("=" * 60)
        logger.info(f"Overall acc: {metric_dict['val/test_overall_acc']*100:.2f}%")
        logger.info(f"Overall veri_acc: {metric_dict['val/test_overall_veri_acc']*100:.2f}%")
        logger.info(f"Overall joint_acc: {metric_dict['val/test_overall_joint_acc']*100:.2f}%")
        logger.info(f"Evidence score: {metric_dict['val/test_overall_evidence']:.4f}")
        logger.info(f"Format score: {metric_dict['val/test_overall_format']:.4f}")
        logger.info(f"Total API calls: {self.api_counter.get_count()}")
        
        # Print API failure statistics
        total_failures = self.api_failure_counter.get_count()
        connection_failures = self.api_connection_error_counter.get_count()
        timeout_failures = self.api_timeout_counter.get_count()
        other_failures = self.api_other_error_counter.get_count()
        
        logger.info("=" * 60)
        logger.info("API FAILURE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total API failures: {total_failures}")
        logger.info(f"Connection errors: {connection_failures}")
        logger.info(f"Timeout errors: {timeout_failures}")
        logger.info(f"Other errors: {other_failures}")
        logger.info(f"Failure rate: {total_failures / max(1, self.api_counter.get_count()) * 100:.2f}%")
        logger.info("=" * 60)
        
        # Print per-label results
        for lb in np.unique(true_label_types):
            logger.info(f"Label {lb}:")
            logger.info(
                f"  acc: {metric_dict[f'val/test_acc/label/{lb}']*100:.2f}%, "
                f"veri_acc: {metric_dict[f'val/test_veri_acc/label/{lb}']*100:.2f}%, "
                f"joint_acc: {metric_dict[f'val/test_joint_acc/label/{lb}']*100:.2f}%"
            )
        
        # Save results
        if self.config.save_csv:
            self._save_csv_results(metric_dict, data_source_reward, challenge_types, 
                                 true_label_types, experiment_name)
        
        if self.config.save_jsonl:
            self._save_jsonl_results(all_results, experiment_name)
        
        return metric_dict
    
    def _save_csv_results(self, metric_dict: Dict[str, float], data_source_reward: Dict, 
                         challenge_types: np.ndarray, true_label_types: np.ndarray,
                         experiment_name: str):
        """Save results to CSV file"""
        csv_file = os.path.join(self.config.output_dir, f"{experiment_name}.csv")
        
        # Prepare header and row
        header = ["exp_name", "overall_acc", "overall_veri_acc", "overall_joint_acc", "evidence_score", "format_score"]
        
        # Add label fields
        for lb in np.unique(true_label_types):
            header += [f"label_{lb}_acc", f"label_{lb}_veri_acc", f"label_{lb}_joint_acc"]
        
        # Prepare row data - convert accuracy metrics to percentage format and round evidence/format scores
        row = [
            experiment_name, 
            f"{metric_dict['val/test_overall_acc']*100:.2f}%", 
            f"{metric_dict['val/test_overall_veri_acc']*100:.2f}%", 
            f"{metric_dict['val/test_overall_joint_acc']*100:.2f}%", 
            f"{metric_dict['val/test_overall_evidence']:.4f}", 
            f"{metric_dict['val/test_overall_format']:.4f}"
        ]
        
        for lb in np.unique(true_label_types):
            row += [
                f"{metric_dict[f'val/test_acc/label/{lb}']*100:.2f}%",
                f"{metric_dict[f'val/test_veri_acc/label/{lb}']*100:.2f}%",
                f"{metric_dict[f'val/test_joint_acc/label/{lb}']*100:.2f}%"
            ]
        
        # Write to CSV
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        
        logger.info(f"Results saved to {csv_file}")
    
    def _save_jsonl_results(self, results: List[Dict], experiment_name: str):
        """Save detailed results to JSONL file"""
        jsonl_file = os.path.join(self.config.output_dir, f"{experiment_name}.jsonl")
        
        def safe_json_serialize(obj):
            """Safely serialize objects to JSON"""
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [safe_json_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(k): safe_json_serialize(v) for k, v in obj.items()}
            else:
                return str(obj)
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for result in results:
                try:
                    safe_result = safe_json_serialize(result)
                    f.write(json.dumps(safe_result, ensure_ascii=False) + '\n')
                except Exception as e:
                    logger.warning(f"Failed to serialize result: {e}")
                    # Try to serialize with basic conversion
                    basic_result = {str(k): str(v) for k, v in result.items()}
                    f.write(json.dumps(basic_result, ensure_ascii=False) + '\n')
        
        logger.info(f"Detailed results saved to {jsonl_file}")


def load_validation_data(data_path: str) -> List[Dict[str, Any]]:
    """Load validation data from parquet or JSONL file"""
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        data = []
        for _, row in df.iterrows():
            data.append({
                'id': row.get('id', ''),
                'question': row.get('question', ''),
                'golden_answers': row.get('golden_answers', {}),
                'data_source': row.get('data_source', 'fever'),
                'ability': row.get('ability', 'unknown'),
            })
        return data
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


def validate_api_synchronized(data_path: str, api_key: str, experiment_name: str = "api_validation_sync", 
                               max_workers: int = 10, batch_size: int = 8, api_base: str = "https://api.openai.com/v1",
                               model: str = "deepseek-reasoner") -> Dict[str, Any]:
    """
    Main function to run synchronized API validation
    
    Args:
        data_path: Path to validation data file (parquet or jsonl)
        api_key: OpenAI API key
        experiment_name: Name for output files
        max_workers: Maximum number of concurrent API calls per turn
        batch_size: Batch size for processing
        api_base: API base URL
        model: Model name to use for API calls
        
    Returns:
        Dictionary containing validation metrics
    """
    # Load data
    data = load_validation_data(data_path)
    
    # Create config
    config = APIConfig(
        api_key=api_key,
        model=model,
        api_base=api_base,
        save_csv=True,
        save_jsonl=True,
        output_dir=f"./validation_results/{experiment_name}",
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Create validator and run validation
    validator = APIValidator(config)
    metrics = validator.validate(data, experiment_name)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Synchronized API Validation Script")
    parser.add_argument("--data_path", type=str, required=True, help="Path to validation data file")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--experiment_name", type=str, default="api_validation_sync", help="Experiment name")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of concurrent API calls per turn")
    parser.add_argument("--batch_size", type=int, default=56, help="Batch size for processing")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1", help="API base URL")
    parser.add_argument("--model", type=str, default="deepseek-reasoner", help="Model name to use for API calls")
    args = parser.parse_args()
    
    # Create config
    config = APIConfig(
        api_key=args.api_key,
        model=args.model,
        api_base=args.api_base,
        save_csv=True,
        save_jsonl=True,
        output_dir=f"./validation_results/{args.experiment_name}",
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # Load data
    data = load_validation_data(args.data_path)
    
    # Create validator and run validation
    validator = APIValidator(config)
    metrics = validator.validate(data, args.experiment_name)
    print("Synchronized validation completed successfully!") 