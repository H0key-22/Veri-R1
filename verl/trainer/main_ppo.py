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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.utils.reward_score import claim_reward
from verl.utils.reward_score import claim_reward_eva, claim_reward_noevid, claim_reward_noformat, claim_reward_noweight, claim_reward_labelonly
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
import os

def _select_rm_score_fn(data_source):
    if data_source in ['feverous', 'exfever']:
        return claim_reward.compute_reward
    elif data_source in ['fever', 'scifact', 'hover', 'feverous_dev', 'exfever_dev']:
        return claim_reward_eva.compute_reward
    # elif data_source == 'fever':
    #     return claim_reward_fever.compute_reward
    # elif data_source == 'scifact':
    #     return claim_reward_scifact.compute_reward
    # elif data_source == 'hover':
    #     return claim_reward_hover.compute_reward
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        acc_tensor    = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        evidence_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        evid_cover_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        veri_acc_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        joint_acc_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = []
        all_accs = []
        all_evidences = []
        all_formats = []
        all_evid_covers = []
        all_veri_accs = []
        all_joint_accs = []

        already_print_data_sources = {}
        # Task categories
        challenges = []
        true_labels = []
        pred_labels = []

        all_results = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            sample_id = data_item.non_tensor_batch['id']
            claim = data_item.non_tensor_batch['question']
            challenge = data_item.non_tensor_batch['ability']
            challenges.append(challenge)
            true_labels.append(ground_truth['label'])

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score, acc, evidence, format, prediction, evid_cover, veri_acc, joint_acc = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score, challenge=challenge, data_source=data_source)

            # Save the results to jsonl if the environment variable is set
            preamble = "<|im_start|>assistant"
            response = sequences_str.split(preamble, 1)[1] if preamble in sequences_str else sequences_str

            save_flag_env = os.getenv('SAVE_JSONL')
            if save_flag_env == 'true':
                from verl.utils.logger.save_jsonl import save_responses_to_jsonl
            
                # Process numpy arrays in ground_truth
                processed_ground_truth = {}
                if hasattr(ground_truth, 'items'):
                    for key, value in ground_truth.items():
                        if isinstance(value, np.ndarray):
                            # Convert numpy array to Python list
                            processed_ground_truth[key] = value.tolist()
                        else:
                            processed_ground_truth[key] = value
                else:
                    processed_ground_truth = ground_truth
                
                all_results.append({
                'id': sample_id,
                'claim': claim,
                'accuracy': float(acc),
                "verification_accuracy": float(veri_acc),
                "joint_accuracy": float(joint_acc),
                'prediction': prediction,
                'ground_truth': processed_ground_truth,
                'response': response,
                'evidence_score': float(evidence),
                'data_source': data_source,
                'challenge': challenge,
            })

            reward_tensor[i, valid_response_length - 1] = score
            acc_tensor   [i, valid_response_length - 1] = acc
            evidence_tensor[i, valid_response_length - 1] = evidence
            format_tensor[i, valid_response_length - 1] = format
            evid_cover_tensor[i, valid_response_length - 1] = evid_cover
            veri_acc_tensor[i, valid_response_length - 1] = veri_acc
            joint_acc_tensor[i, valid_response_length - 1] = joint_acc
            all_scores.append(score)
            all_accs.append(acc)
            all_evidences.append(evidence)
            all_formats.append(format)
            all_evid_covers.append(evid_cover)
            all_veri_accs.append(veri_acc)
            all_joint_accs.append(joint_acc)

            # print(f"[DEBUG] {data_source} score: {score}, acc: {acc}")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        print(f"[DEBUG] all_accs mean: {np.mean(all_accs)}")
        print(f"[DEBUG] all_evidences mean: {np.mean(all_evidences)}")
        print(f"[DEBUG] all_formats mean: {np.mean(all_formats)}")
        print(f"[DEBUG] all_evid_covers mean: {np.mean(all_evid_covers)}")
        print(f"[DEBUG] all_veri_accs mean: {np.mean(all_veri_accs)}")
        print(f"[DEBUG] all_joint_accs mean: {np.mean(all_joint_accs)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")
    
        if save_flag_env == 'true':
            save_responses_to_jsonl(all_results)


        return reward_tensor, acc_tensor, evidence_tensor, format_tensor, challenges, true_labels, pred_labels, evid_cover_tensor, veri_acc_tensor, joint_acc_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }
    
    # Only add RefPolicy if not in val_only mode
    if not config.trainer.get('val_only', False):
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    
    # Only add RefPolicy mapping if not in val_only mode
    if not config.trainer.get('val_only', False):
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
