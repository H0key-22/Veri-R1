# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of claim verification models using reward functions and ground truth.
The input is a JSONL file where each line contains one response and ground truth labels/evidence.
Supports various claim verification datasets: FEVER, SciFact, HoVER, FEVEROUS, and ExFEVER.

"""

import hydra
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.reward_score import claim_reward
from verl.utils.reward_score import claim_reward_eva
import pandas as pd
import numpy as np
import os
import json


def select_reward_fn(data_source):
    if data_source in ['fever', 'scifact', 'hover', 'feverous', 'exfever']:
        return claim_reward_eva.compute_reward
    else:
        raise NotImplementedError(f"Unsupported data source: {data_source}")


@hydra.main(config_path='config', config_name='evaluation', version_base=None)
def main(config):
    local_path = copy_local_path_from_hdfs(config.data.path)
    dataset = pd.read_json(local_path, lines=True)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    ground_truth_data = dataset[config.data.ground_truth_key]

    total = len(dataset)
    
    # Initialize metrics for claim verification
    all_scores = []
    all_accs = []
    all_evidences = []
    all_formats = []
    all_evid_covers = []
    all_veri_accs = []
    all_joint_accs = []
    
    # For detailed results saving
    all_results = []

    for i in range(total):
        # Each row has only one response, not a list
        response = responses[i]
        data_source = data_sources[i]
        ground_truth = ground_truth_data[i]
        reward_fn = select_reward_fn(data_source)

        # compute score
        score, label_acc, evidence_score, format_score, pred_label, evid_cover, veri_acc, joint_acc = reward_fn(
            solution_str=response, 
            ground_truth=ground_truth, 
            format_score=0.0, 
            challenge=dataset.iloc[i].get('ability', 'unknown'), 
            data_source=data_source
        )
        
        all_scores.append(score)
        all_accs.append(label_acc)
        all_evidences.append(evidence_score)
        all_formats.append(format_score)
        all_evid_covers.append(evid_cover)
        all_veri_accs.append(veri_acc)
        all_joint_accs.append(joint_acc)

    # Print claim verification results
    print(f'=== Claim Verification Results ===')
    print(f'Total samples: {total}')
    print(f'Average Reward Score: {np.mean(all_scores):.4f}')
    print(f'Average Accuracy: {np.mean(all_accs):.4f}')
    print(f'Average Evidence Score: {np.mean(all_evidences):.4f}')
    print(f'Average Format Score: {np.mean(all_formats):.4f}')
    print(f'Average Evidence Coverage: {np.mean(all_evid_covers):.4f}')
    print(f'Average Verification Accuracy: {np.mean(all_veri_accs):.4f}')
    print(f'Average Joint Accuracy: {np.mean(all_joint_accs):.4f}')
    
    
    # Save summary metrics to CSV file
    summary_file = config.get('summary_file', 'evaluation_summary.csv')
    summary_data = {
        'total_samples': total,
        'avg_reward_score': np.mean(all_scores),
        'avg_accuracy': np.mean(all_accs),
        'avg_evidence_score': np.mean(all_evidences),
        'avg_format_score': np.mean(all_formats),
        'avg_evidence_coverage': np.mean(all_evid_covers),
        'avg_verification_accuracy': np.mean(all_veri_accs),
        'avg_joint_accuracy': np.mean(all_joint_accs)
    }
    df_summary = pd.DataFrame([summary_data])
    df_summary.to_csv(summary_file, index=False, encoding='utf-8')
    print(f'Summary metrics saved to CSV: {summary_file}')


if __name__ == '__main__':
    main()
