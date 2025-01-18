# Base Packages
import argparse
import datetime
import random
import os
import time
from pathlib import Path
# Installed Packages
import wandb
import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
# Working Sources
from decision_transformer.utils.utils import (
    seed_fix, 
    cycle,
    data_preprocessing,
    loss_fn,
    get_env_builder,
)
from decision_transformer.utils.replay_buffer import (
    ReplayBuffer,
)
from decision_transformer.utils.lamb import (
    Lamb,
)
from decision_transformer.models.online_decision_transformer import (
    OnlineDecisionTransformer,
)
# from decision_transformer.models.online_prompt_decision_transformer import (
#     OnlineSoftPromptDecisionTransformer,
# )
from decision_transformer.models.mlp_bc import (
    MLPBCModel,
)
from decision_transformer.training.online_trainer import (
    OnlineTrainer,
)
from decision_transformer.evaluation.evaluate_episodes import (
    eval_episodes,
    evaluate_episode, 
    evaluate_episode_rtg
)
from decision_transformer.data_loader.data_loader import (
    create_dataloader,
    MetaworldDataloader,
)


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    tuning_type = variant['tuning_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    model_post_fix = '_TRAIN_'+variant['dataset']
    if variant['stochastic_policy']:
        model_post_fix += '_STOCHASTIC_POLICY'
    if variant['tuning_type'] == "prompt":
        model_post_fix += '_SOFT_PROMPT'
    elif variant['tuning_type'] == "fine":
        model_post_fix += '_FINE_TUNE'
    if variant['deep_tuning']:
        model_post_fix += '_DEEP'
    
    dt_datetime = datetime.datetime.now()
    format = '%Y-%m-%d_%H:%M:%S'
    str_datetime = datetime.datetime.strftime(dt_datetime, format)

    seed_fix(variant["seed"])

    trajectory_data_info = data_preprocessing(variant)

    if tuning_type == "prompt":
        model = OnlineSoftPromptDecisionTransformer(
            state_dim=trajectory_data_info["state_dim"],
            act_dim=trajectory_data_info["act_dim"],
            action_range=trajectory_data_info["action_range"],
            max_length=variant["K"],
            eval_context_length=variant["K"],
            max_ep_len=variant["max_ep_len"],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            stochastic_policy = variant['stochastic_policy'],
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=-trajectory_data_info["act_dim"],
            prompt_length=variant['prompt_length'],
            prompt_pdrop=variant['dropout'],
            deep_tuning=variant['deep_tuning'],
        )
        if variant['tuning_type'] != "":
            saved_model_path = variant['model_path']
            model.load_state_dict(torch.load(saved_model_path), strict=False)
            print('model initialized from: ', saved_model_path)

            print("Turning off gradients in policy")
            for name, param in model.named_parameters():
                if "prompt" not in name:
                    param.requires_grad_(False)
            # Double check
            enabled = set()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            print(f"Parameters to be updated: {enabled}")
    else:
        model = OnlineDecisionTransformer(
            state_dim=trajectory_data_info["state_dim"],
            act_dim=trajectory_data_info["act_dim"],
            action_range=trajectory_data_info["action_range"],
            max_length=variant["K"],
            eval_context_length=variant["K"],
            max_ep_len=variant["max_ep_len"],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            stochastic_policy = variant['stochastic_policy'],
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=-trajectory_data_info["act_dim"],
        )
        if variant['tuning_type'] != "":
            saved_model_path = variant['model_path']
            model.load_state_dict(torch.load(saved_model_path), strict=True)
            print('model initialized from: ', saved_model_path)
    
    model = model.to(device=device)

    optimizer = Lamb(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
    )
    
    log_temperature_optimizer = torch.optim.Adam(
        [model.log_temperature],
        lr=1e-4,
        betas=[0.9, 0.999],
    )
    
    if not variant["evaluation"]:
        if variant["pretrain"]:
            print("\n\n\n*** Offline Pretrain ***")
            # Make Data Loader
            trajectory_dataloader = MetaworldDataloader(trajectory_data_info, variant["K"], variant["batch_size"], device=device)

            save_path = os.path.join(variant['save_path'], str_datetime+"_"+env_name+"_"+variant['tuning_type'])
            print("Save Path:", save_path)
            Path(save_path).mkdir(parents=True, exist_ok=True)

            trainer = OnlineTrainer(
                model=model,
                optimizer=optimizer,
                log_temperature_optimizer=log_temperature_optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                il_alpha=variant["il_alpha"],
                batch_size=variant["batch_size"],
                data_loader=trajectory_dataloader,
                eval_fns=[
                    eval_episodes(
                        tar, trajectory_data_info, variant['num_eval_episodes'], model_type, variant["env"], device=device
                    ) for tar in trajectory_data_info["env_targets"]
                ],
                device=device,
                max_iters = variant["max_iters"],
                eval_interval = variant["eval_interval"],
            )
                
            if log_to_wandb:
                wandb.init(
                    name=exp_prefix,
                    group=group_name,
                    project='decision-transformer',
                    config=variant
                )
                # wandb.watch(model)  # wandb has some bug

            for iters in range(variant['max_iters']):
                outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iters+1, print_logs=True)
                if log_to_wandb:
                    wandb.log(outputs)

                if iters % variant['save_interval'] == 0:
                    trainer.save_model(
                        env_name = variant["env"],
                        postfix = model_post_fix+'_iter_'+str(iters+1),
                        folder = save_path
                    )
        elif variant["adaptation"]:
            print("\n\n\n*** Online Adaptation ***")
            # Save Path
            save_path = os.path.join(variant['save_path'], str_datetime+"_"+env_name+"_"+variant['tuning_type'])
            print("Save Path:", save_path)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            # Wandb Init
            if log_to_wandb:
                wandb.init(
                    name=exp_prefix,
                    group=group_name,
                    project='decision-transformer',
                    config=variant
                )
                # wandb.watch(model)  # wandb has some bug
            # Init Evaluation
            if variant["init_eval"]:
                eval_fns=[
                    eval_episodes(
                        tar, trajectory_data_info, variant['num_eval_episodes'], model_type, variant["env"], device=device
                    ) for tar in trajectory_data_info["env_targets"]
                ]
                logs = dict()
                model.eval()
                for eval_fn in eval_fns:
                    outputs = eval_fn(model)
                    for k, v in outputs.items():
                        logs[f'evaluation/{k}'] = v

                print('=' * 80)
                print(f'Init Evaluation')
                for k, v in logs.items():
                    print(f'{k}: {v}')
                del eval_fns
            # Online Rollout Envs
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(
                        variant["seed"], 
                        env_name = variant["env"], 
                        target_goal = trajectory_data_info["train_scale"], 
                        mode = "train",
                        shuffle = i,
                    ) for i in range(variant["num_online_rollouts"])
                ]
            )
            # Replay Buffer
            replay_buffer = ReplayBuffer(variant["replay_size"], trajectory_data_info)
            # Trainer
            trainer = OnlineTrainer(
                model=model,
                optimizer=optimizer,
                log_temperature_optimizer=log_temperature_optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                il_alpha=variant["il_alpha"],
                batch_size=variant["batch_size"],
                data_loader=None,
                eval_fns=[
                    eval_episodes(
                        tar, trajectory_data_info, variant['num_eval_episodes'], model_type, variant["env"], device=device
                    ) for tar in trajectory_data_info["env_targets"]
                ],
                device=device,
                max_iters = variant["max_iters"],
                eval_interval = variant["eval_interval"],
            )
            # Train
            online_iter = 0
            start_time = time.time()
            while online_iter < variant["max_online_iters"]:
                outputs = {}
                augment_outputs = replay_buffer.augment_trajectories(
                    online_envs,
                    trajectory_data_info["train_scale"],
                    model=model,
                )
                outputs.update(augment_outputs)

                trajectory_dataloader = create_dataloader(
                    trajectories=replay_buffer.trajectories,
                    num_iters=variant["num_updates_per_online_iter"],
                    batch_size=variant["batch_size"],
                    max_len=variant["K"],
                    max_ep_len=variant["max_ep_len"],
                    state_dim=trajectory_data_info["state_dim"],
                    act_dim=trajectory_data_info["act_dim"],
                    state_mean=trajectory_data_info["state_mean"],
                    state_std=trajectory_data_info["state_std"],
                    reward_scale=trajectory_data_info["train_scale"],
                    action_range=trajectory_data_info["action_range"],
                    num_workers=2,
                )

                is_last_iter = online_iter == variant["max_online_iters"] - 1
                if (online_iter + 1) % variant["eval_interval"] == 0 or is_last_iter:
                    evaluation = True
                else:
                    evaluation = False

                train_outputs = trainer.online_train_iteration(
                    dataloader=trajectory_dataloader,
                )
                outputs.update(train_outputs)

                if evaluation:
                    model.eval()
                    for eval_fn in trainer.eval_fns:
                        eval_outputs = eval_fn(model)
                        for k, v in eval_outputs.items():
                            outputs[f'evaluation/{k}'] = v

                outputs["time/total"] = time.time() - start_time
                print('=' * 80)
                print(f'Online Iter {online_iter+1}')
                for k, v in outputs.items():
                    print(f'{k}: {v}')

                # Log

                if (online_iter+1) % variant['save_interval'] == 0:
                    trainer.save_model(
                        env_name = variant["env"],
                        postfix = model_post_fix+'_iter_'+str(online_iter+1),
                        folder = save_path
                    )
                online_iter += 1

            online_envs.close()
        else:
            print("Choose Pretrain or Adaptation")
            raise ValueError
    else:
        print("\n\n\n*** Online Evaluation ***")

        saved_model_path = variant['model_path']
        model.load_state_dict(torch.load(saved_model_path), strict=True)
        print('model initialized from: ', saved_model_path)

        eval_fns=[
            eval_episodes(
                tar, trajectory_data_info, variant['num_eval_episodes'], model_type, variant["env"], device=device
            ) for tar in trajectory_data_info["env_targets"]
        ]
        logs = dict()
        model.eval()
        for eval_fn in eval_fns:
            outputs = eval_fn(model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        print('=' * 80)
        print(f'Evaluation')
        for k, v in logs.items():
            print(f'{k}: {v}')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Exp
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument('--env', type=str, default='reach-v2')
    parser.add_argument('--dataset', type=str, default='expert')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--state_norm', type=str, default='fixed_norm') # or None, cal_norm
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    # Model
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--max_ep_len', type=int, default=500)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--stochastic_policy', action='store_true', default=True)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)
    # Optimize
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument("--il_alpha", type=float, default=0.0)
    # Train
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_online_rollouts", type=int, default=10)
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    # Tuning
    parser.add_argument('--adaptation', action='store_true', default=False)
    parser.add_argument('--tuning_type', type=str, default='')
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--deep_tuning', action='store_true', default=False)
    # Eval
    parser.add_argument('--evaluation', action='store_true', default=False)
    parser.add_argument('--init_eval', action='store_true', default=False)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--num_eval_episodes', type=int, default=50)
    # Etc
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='saved_model/online_adapt/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='saved_model/2023-07-19_18:30:52_MT10_/model_MT10_TRAIN_expert_STOCHASTIC_POLICY_iter_10')
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
