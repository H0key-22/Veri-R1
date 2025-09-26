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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import shutil
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Optional
import json
import csv
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_engine.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    sequence_acc = batch.batch['token_level_accs'].sum(-1)
    sequence_evidence = batch.batch['token_level_evidences'].sum(-1)
    sequence_format = batch.batch['token_level_formats'].sum(-1)
    sequence_evid_cover = batch.batch['token_level_evid_covers'].sum(-1)
    seqeunce_veri_acc = batch.batch['token_level_veri_accs'].sum(-1)
    sequence_joint_acc = batch.batch['token_level_joint_accs'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    challenges = batch.meta_info['challenges']
    true_lables = batch.meta_info['true_labels']


    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        # acc
        'critic/acc/mean': torch.mean(sequence_acc).detach().item(),
        # evidence
        'critic/evidence/mean': torch.mean(sequence_evidence).detach().item(),
        # format
        'critic/format/mean': torch.mean(sequence_format).detach().item(),
        # evidence cover
        'critic/evidence_cover/mean': torch.mean(sequence_evid_cover).detach().item(),
        # verification accuracy
        'critic/veri_acc/mean': torch.mean(seqeunce_veri_acc).detach().item(),
        # joint accuracy
        'critic/joint_acc/mean': torch.mean(sequence_joint_acc).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())

    # compute challenge metrics
    seq_acc_np = sequence_acc.detach().cpu().numpy()
    # add challenge metrics
    acc_by_challenge = defaultdict(list)
    for chall, acc in zip(challenges, seq_acc_np):
        acc_by_challenge[chall].append(acc)
    for chall, accs in acc_by_challenge.items():
        metrics[f"acc/challenge/{chall}"] = float(sum(accs) / len(accs))

    # compute label metrics
    seq_acc_np = sequence_acc.detach().cpu().numpy()
    # add label metrics
    acc_by_label = defaultdict(list)
    for label, acc in zip(true_lables, seq_acc_np):
        acc_by_label[label].append(acc)
    for label, accs in acc_by_label.items():
        metrics[f"acc/label/{label}"] = float(sum(accs) / len(accs))
    
    # Challenges to DataSources
    exfever_challenges = {
        "EX-FEVER"
    }

    feverous_challenges = {
        "Other",
        "Multi-hop Reasoning",
        "Search terms not in claim",
        "Numerical Reasoning",
        "Entity Disambiguation"
    }

    # collect all accuracies into two buckets
    exfever_accs = []
    feverous_accs = []
    for chall, accs in acc_by_challenge.items():
        if chall in exfever_challenges:
            exfever_accs.extend(accs)
        elif chall in feverous_challenges:
            feverous_accs.extend(accs)

    # compute and record the aggregate metrics
    if exfever_accs:
        metrics["acc/EX-FEVER"] = float(sum(exfever_accs) / len(exfever_accs))
    else:
        metrics["acc/EX-FEVER"] = 0.0

    if feverous_accs:
        metrics["acc/FEVEROUS"] = float(sum(feverous_accs) / len(feverous_accs))
    else:
        metrics["acc/FEVEROUS"] = 0.0

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.keep_topk = self.config.trainer.keep_topk_checkpoints
        self.metric_key = self.config.trainer.metric_for_best
        self.best_ckpts: list[tuple[float, str]] = []
    
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        
        # Skip training data loading if val_only is True
        if not self.config.trainer.get('val_only', False):
            self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                             tokenizer=self.tokenizer,
                                             prompt_key=self.config.data.prompt_key,
                                             max_prompt_length=self.config.data.max_prompt_length,
                                             filter_prompts=True,
                                             return_raw_chat=self.config.data.get('return_raw_chat', False),
                                             truncation='error')
            if self.config.data.train_data_num is not None:
                if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                    print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
                else:
                    self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
            print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                               batch_size=self.config.data.train_batch_size,
                                               shuffle=self.config.data.shuffle_train_dataloader,
                                               drop_last=True,
                                               collate_fn=collate_fn)
        else:
            print("[INFO] Skipping training data loading in val_only mode")
            self.train_dataset = None
            self.train_dataloader = None

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        if self.train_dataloader is not None:
            print(f'Size of train dataloader: {len(self.train_dataloader)}')
            assert len(self.train_dataloader) >= 1
        else:
            print('Train dataloader is None (val_only mode)')
            
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        if self.train_dataloader is not None:
            total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        else:
            # In val_only mode, use a dummy value
            total_training_steps = 1

        self.total_training_steps = total_training_steps
        if self.config.trainer.get('val_only', False):
            print(f'Total training steps: {self.total_training_steps} (dummy value for val_only mode)')
        else:
            print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
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

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            do_search = self.config.do_search,
        )
        

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )


        for batch_dict in self.val_dataloader:
            timing_raw = {}
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
            data_sources = test_batch.non_tensor_batch.get(
                'data_source',
                ['unknown'] * len(test_batch.batch['input_ids'])
            )
            
            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        initial_input_ids=first_input_ids,
                        data_sources=data_sources
                    )
                
                test_batch = test_batch.union(final_gen_batch_output)
                
                for key in test_batch.batch.keys():
                    test_batch.batch[key] = test_batch.batch[key].long()
                
                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor, acc_tensor, evidence_tensor, format_tensor, challenges, true_labels, pred_labels, evid_cover_tensor, veri_acc_tensor, joint_acc_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                acc_tensor_lst.append(acc_tensor)
                evidence_tensor_lst.append(evidence_tensor)
                format_tensor_lst.append(format_tensor)
                evid_cover_tensor_lst.append(evid_cover_tensor)
                veri_acc_tensor_lst.append(veri_acc_tensor)
                joint_acc_tensor_lst.append(joint_acc_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                challenge_lst.append(challenges)
                true_label_lst.append(true_labels)
                pred_label_lst.append(pred_labels)

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        acc_tensor = torch.cat([acc.sum(-1) for acc in acc_tensor_lst], dim=0).cpu()  # (batch_size,)
        evidence_tensor = torch.cat([ev.sum(-1) for ev in evidence_tensor_lst], dim=0).cpu()  # (batch_size,)
        format_tensor = torch.cat([fm.sum(-1) for fm in format_tensor_lst], dim=0).cpu()  # (batch_size,)
        evid_cover_tensor = torch.cat([ec.sum(-1) for ec in evid_cover_tensor_lst], dim=0).cpu()  # (batch_size,)
        veri_acc_tensor = torch.cat([va.sum(-1) for va in veri_acc_tensor_lst], dim=0).cpu()  # (batch_size,)
        joint_acc_tensor = torch.cat([ja.sum(-1) for ja in joint_acc_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # Concatenate all batch challenges into a 1D array
        challenge_types = np.concatenate(challenge_lst, axis=0)
        # Concatenate all batch labels into a 1D array
        true_label_types = np.concatenate(true_label_lst, axis=0)
        pred_label_types = np.concatenate(pred_label_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_acc = {}
        data_source_evidence = {}
        data_source_format = {}
        data_source_evid_cover = {}
        data_source_veri_acc = {}
        data_source_joint_acc = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_acc[data_source] = []
                data_source_evidence[data_source] = []
                data_source_format[data_source] = []
                data_source_evid_cover[data_source] = []
                data_source_veri_acc[data_source] = []
                data_source_joint_acc[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_acc[data_source].append(acc_tensor[i].item())
            data_source_evidence[data_source].append(evidence_tensor[i].item())
            data_source_format[data_source].append(format_tensor[i].item())
            data_source_evid_cover[data_source].append(evid_cover_tensor[i].item())
            data_source_veri_acc[data_source].append(veri_acc_tensor[i].item())
            data_source_joint_acc[data_source].append(joint_acc_tensor[i].item())

        metric_dict = {}
        
        for data_source, rewards in data_source_reward.items():
            idx = (data_sources == data_source)
            metric_dict[f'val/test_overall_acc'] = float(acc_tensor.mean())
            metric_dict[f'val/test_overall_veri_acc'] = float(veri_acc_tensor.mean())
            metric_dict[f'val/test_overall_joint_acc'] = float(joint_acc_tensor.mean())
            metric_dict[f'val/test_overall_evidence'] = float(evidence_tensor.mean())
            metric_dict[f'val/test_overall_format'] = float(format_tensor.mean())
            metric_dict[f'val/test_rewards/{data_source}'] = np.mean(rewards)
            metric_dict[f'val/test_acc/{data_source}'] = np.mean(data_source_acc[data_source])
            metric_dict[f'val/test_evidence/{data_source}'] = np.mean(data_source_evidence[data_source])
            metric_dict[f'val/test_format/{data_source}'] = np.mean(data_source_format[data_source])
            metric_dict[f'val/test_evid_cover/{data_source}'] = np.mean(data_source_evid_cover[data_source])
            metric_dict[f'val/test_veri_acc/{data_source}'] = np.mean(data_source_veri_acc[data_source])
            metric_dict[f'val/test_joint_acc/{data_source}'] = np.mean(data_source_joint_acc[data_source])
        challenge_veri_acc = {}
        challenge_joint_acc = {}
        # Per challenge type
        for ct in np.unique(challenge_types):
            idx = (challenge_types == ct)
            metric_dict[f'val/test_acc/challenge/{ct}'] = float(acc_tensor[idx].mean())
            metric_dict[f'val/test_veri_acc/challenge/{ct}'] = float(veri_acc_tensor[idx].mean())
            metric_dict[f'val/test_joint_acc/challenge/{ct}'] = float(joint_acc_tensor[idx].mean())
        

        for lb in np.unique(true_label_types):
            idx = (true_label_types == lb)
            metric_dict[f'val/test_acc/label/{lb}'] = float(acc_tensor[idx].mean())
            metric_dict[f'val/test_veri_acc/label/{lb}'] = float(veri_acc_tensor[idx].mean())
            metric_dict[f'val/test_joint_acc/label/{lb}'] = float(joint_acc_tensor[idx].mean())
    
        # Finally add a print to show overall acc/evidence/format and each challenge's acc and f1
        # ==== Print overall and each challenge metrics ====
        print(f"Overall acc: {metric_dict[f'val/test_overall_acc']:.4f}, ")
        print(f"Overall veri_acc: {metric_dict[f'val/test_overall_veri_acc']:.4f}, ")
        print(f"Overall joint_acc: {metric_dict[f'val/test_overall_joint_acc']:.4f}, ")
        print(f"Overall evidence: {metric_dict[f'val/test_overall_evidence']:.4f}, ")
        print(f"Overall format: {metric_dict[f'val/test_overall_format']:.4f}, ")
        for data_source in data_source_reward.keys():
            print(f"Data source {data_source}: ")
            print(
                f"acc: {metric_dict[f'val/test_acc/{data_source}']:.4f}, "
                f"evidence: {metric_dict[f'val/test_evidence/{data_source}']:.4f}, "
                f"format: {metric_dict[f'val/test_format/{data_source}']:.4f}, "
                f"evid_cover: {metric_dict[f'val/test_evid_cover/{data_source}']:.4f}, "
                f"veri_acc: {metric_dict[f'val/test_veri_acc/{data_source}']:.4f}, "
                f"joint_acc: {metric_dict[f'val/test_joint_acc/{data_source}']:.4f}"
            )
        
        for ct in np.unique(challenge_types):
            print(f"Challenge {ct}: acc={metric_dict[f'val/test_acc/challenge/{ct}']:.4f}, "
            f"veri_acc={metric_dict[f'val/test_veri_acc/challenge/{ct}']:.4f}, "
            f"joint_acc={metric_dict[f'val/test_joint_acc/challenge/{ct}']:.4f}")
        
        for lb in np.unique(true_label_types):
            print(f"Label {lb}: acc={metric_dict[f'val/test_acc/label/{lb}']:.4f}, "
            f"veri_acc={metric_dict[f'val/test_veri_acc/label/{lb}']:.4f}, "
            f"joint_acc={metric_dict[f'val/test_joint_acc/label/{lb}']:.4f}")
        
        print("-------------------------------------------------------")
                # ==== Save to CSV ====
        if self.config.eval.save_csv:
            # Define CSV filename
            result_path = os.getenv("RESULT_PATH")
            exp_name = os.getenv("EXPERIMENT_NAME")
            csv_file = f"{result_path}.csv"
            # Check if file exists
            file_exists = os.path.isfile(csv_file)
            # Prepare row data to write
            header = [
                "exp_name", "overall_acc", "overall_veri_acc", "overall_joint_acc", "evidence_score", "format_score"
            ]
            
            # Fields for each label
            for lb in np.unique(true_label_types):
                header += [
                    f"label_{lb}_acc", f"label_{lb}_veri_acc", f"label_{lb}_joint_acc"
                ]
            # Construct row
            row = [
                exp_name,
                f"{metric_dict['val/test_overall_acc']*100:.2f}%",
                f"{metric_dict['val/test_overall_veri_acc']*100:.2f}%",
                f"{metric_dict['val/test_overall_joint_acc']*100:.2f}%",
                f"{metric_dict['val/test_overall_evidence']:.4f}",
                f"{metric_dict['val/test_overall_format']:.4f}",
            ]
            for lb in np.unique(true_label_types):
                row += [
                    f"{metric_dict[f'val/test_acc/label/{lb}']*100:.2f}%",
                    f"{metric_dict[f'val/test_veri_acc/label/{lb}']*100:.2f}%",
                    f"{metric_dict[f'val/test_joint_acc/label/{lb}']*100:.2f}%",
                ]
            # Open file and write
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(row)
            print(f"Metrics logged to {csv_file}")

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _prune_checkpoints(self, base_dir: str, keep: int = 4):
        """
        Keep only the latest keep subdirectories under base_dir, sorted by N in global_step_N, delete the rest.
        """
        # List all subdirectories starting with "global_step_"
        all_dirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("global_step_")
        ]
        # Sort by trailing number (global_step_10 comes after global_step_2)
        sorted_dirs = sorted(
            all_dirs,
            key=lambda d: int(d.split("_")[-1])
        )
        # Delete all except the last keep directories
        to_delete = sorted_dirs[:-keep]
        for d in to_delete:
            shutil.rmtree(os.path.join(base_dir, d), ignore_errors=True)

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
         # Create directory
        actor_base = os.path.dirname(actor_local_path)
        os.makedirs(actor_base, exist_ok=True)

        # # Prune old checkpoints, keep only the latest 2
        # self._prune_checkpoints(actor_base, keep=2)

        # Save new checkpoint
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
        
        return actor_local_path
    

    def _save_checkpoint_with_score(self, extra_tag: Optional[str] = None):
        step_tag = f"global_step_{self.global_steps}"
        if extra_tag is not None:
            step_tag += f"_{extra_tag}"
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor', step_tag)
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        actor_base = os.path.dirname(actor_local_path)
        os.makedirs(actor_base, exist_ok=True)

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic', step_tag)
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)
        
        return actor_local_path


    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            do_search = self.config.do_search,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )
            
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # print("DataProto keys:", batch)

                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # data_source is used to record the data source of the batch
                data_sources = batch.non_tensor_batch.get(
                    'data_source',
                    ['unknown'] * len(batch.batch['input_ids'])
                )

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                #     if not self.config.do_search:
                #         gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                #         batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                #                                                 dtype=object)
                #         # repeat to align with repeated responses in rollout
                #         batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                #         batch = batch.union(gen_batch_output)

                # ####################
                # # Below is aLL about agents - the "LLM + forloop"
                # ####################
                # # with _timer('step', timing_raw):
                #     else:
                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                            data_sources=data_sources
                        )

                    # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                    for key in final_gen_batch_output.batch.keys():
                        final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                    with torch.no_grad():
                        output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                        final_gen_batch_output = final_gen_batch_output.union(output)

                    # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    #                                         dtype=object)
                    batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                        
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor, acc_tensor, evidence_tensor, format_tensor, challenges, true_labels, pred_labels, evid_cover_tensor, veri_acc_tensor, joint_acc_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor
                        batch.batch['token_level_accs'] = acc_tensor
                        batch.batch['token_level_evidences'] = evidence_tensor
                        batch.batch['token_level_formats'] = format_tensor
                        batch.batch['token_level_evid_covers'] = evid_cover_tensor
                        batch.batch['token_level_veri_accs'] = veri_acc_tensor
                        batch.batch['token_level_joint_accs'] = joint_acc_tensor
                        batch.meta_info['challenges'] = challenges
                        batch.meta_info['true_labels'] = true_labels
                        batch.meta_info['pred_labels'] = pred_labels

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            # if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                            batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                        if self.config.trainer.save_after_val == True:
                            score = val_metrics[self.metric_key]
                            # todo If score in self.best_score belongs to self.keep_topk, then save (need to consider saving when self.best_score is empty)
                            if len(self.best_ckpts) < self.keep_topk:
                                should_save = True                        # Leaderboard not full - save directly
                            else:
                                worst_score = self.best_ckpts[-1][0]        # Worst score in current leaderboard
                                should_save = score > worst_score         # Only save if better

                            if should_save:
                                # Save checkpoint with score in directory name for easy deletion later
                                ckpt_dir = self._save_checkpoint_with_score(extra_tag=f"val_{score:.4f}")

                                # Update leaderboard and sort by score in descending order
                                self.best_ckpts.append((score, ckpt_dir))
                                self.best_ckpts.sort(key=lambda x: x[0], reverse=True)

                                while len(self.best_ckpts) > self.keep_topk:
                                    worst_score, worst_path = self.best_ckpts.pop(-1)
                                    shutil.rmtree(worst_path, ignore_errors=True)

                    if self.config.trainer.save_freq > 0 and \
                        self.global_steps % self.config.trainer.save_freq == 0 and \
                        self.config.trainer.save_after_val == False:
                        with _timer('save_checkpoint', timing_raw):
                            ckpt_dir = self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                print(f"[Step {self.global_steps}] metrics: {metrics}")

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # save the last checkpoint
                    self._save_checkpoint()
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
