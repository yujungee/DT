# Base Packages
import argparse
import datetime
import random
import os
from pathlib import Path
# Installed Packages
import gym
import numpy as np
import torch
import wandb
# Working Sources
from decision_transformer.utils.utils import (
    seed_fix, 
    data_preprocessing,
    copy_some_parameters,
)
from decision_transformer.models.decision_transformer_backup import (
    DecisionTransformer,
)
from decision_transformer.models.decision_transformer import (
    DecisionTransformer,
)
from decision_transformer.models.prompt_decision_transformer import (
    SoftPromptDecisionTransformer,
)
from decision_transformer.models.ia3_decision_transformer import (
    IA3DecisionTransformer,
)
from decision_transformer.models.mlp_bc import (
    MLPBCModel,
)
from decision_transformer.training.act_trainer import (
    ActTrainer,
)
from decision_transformer.training.seq_trainer import (
    SequenceTrainer,
)
from decision_transformer.evaluation.evaluate_episodes import (
    eval_episodes,
    evaluate_episode, 
    evaluate_episode_rtg
)
from decision_transformer.data_loader.data_loader import (
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

    trajectory_dataloader = MetaworldDataloader(trajectory_data_info, variant["K"], variant["max_ep_len"], variant["batch_size"], device=device)

    if model_type == 'dt':
        if tuning_type == "prompt":
            model = SoftPromptDecisionTransformer(
                state_dim=trajectory_data_info["state_dim"],
                act_dim=trajectory_data_info["act_dim"],
                max_length=variant["K"],
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
                prompt_length=variant['prompt_length'],
                prompt_pdrop=variant['dropout'],
                deep_tuning=variant['deep_tuning'],
                init_temperature=variant["init_temperature"],
                target_entropy=-trajectory_data_info["act_dim"],
                action_range=trajectory_data_info["action_range"],
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
        
        elif tuning_type == "ia3":
            model = IA3DecisionTransformer(
                state_dim=trajectory_data_info["state_dim"],
                act_dim=trajectory_data_info["act_dim"],
                max_length=variant["K"],
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
                prompt_length=variant['prompt_length'],
                prompt_pdrop=variant['dropout'],
                deep_tuning=variant['deep_tuning'],
                init_temperature=variant["init_temperature"],
                target_entropy=-trajectory_data_info["act_dim"],
                action_range=trajectory_data_info["action_range"],
            )
            if variant['tuning_type'] != "":
                saved_model_path = variant['model_path']
                source_model = DecisionTransformer(
                    state_dim=trajectory_data_info["state_dim"],
                    act_dim=trajectory_data_info["act_dim"],
                    max_length=variant["K"],
                    max_ep_len=variant["max_ep_len"],
                    hidden_size=variant['embed_dim'],
                    n_layer=variant['n_layer'],
                    n_head=variant['n_head'],
                    n_inner=4*variant['embed_dim'],
                    activation_function=variant['activation_function'],
                    n_positions=1024,
                    resid_pdrop=variant['dropout'],
                    attn_pdrop=variant['dropout'],
                    stochastic_policy = variant['stochastic_policy'],
                    init_temperature=variant["init_temperature"],
                    target_entropy=-trajectory_data_info["act_dim"],
                    action_range=trajectory_data_info["action_range"],
                )
                source_model.load_state_dict(torch.load(saved_model_path), strict=True)
                copy_some_parameters(source_model=source_model, target_model=model)
                print('model initialized from: ', saved_model_path)

                print("Turning off gradients in policy")
                for name, param in model.named_parameters():
                    if "ia3" not in name:
                        param.requires_grad_(False)
                # Double check
                enabled = set()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        enabled.add(name)
                print(f"Parameters to be updated: {enabled}")

        else:
            model = DecisionTransformer(
                state_dim=trajectory_data_info["state_dim"],
                act_dim=trajectory_data_info["act_dim"],
                max_length=variant["K"],
                max_ep_len=variant["max_ep_len"],
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
                stochastic_policy = variant['stochastic_policy'],
                init_temperature=variant["init_temperature"],
                target_entropy=-trajectory_data_info["act_dim"],
                action_range=trajectory_data_info["action_range"],
            )
            if variant['tuning_type'] != "":
                saved_model_path = variant['model_path']
                model.load_state_dict(torch.load(saved_model_path), strict=True)
                print('model initialized from: ', saved_model_path)
    elif model_type == 'dif':
        pass
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=trajectory_data_info["state_dim"],
            act_dim=trajectory_data_info["act_dim"],
            max_length=variant["K"],
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )
    
    if not variant["evaluation"]:
        save_path = os.path.join(variant['save_path'], str_datetime+"_"+env_name+"_"+variant['tuning_type'])
        print("Save Path:", save_path)
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if model_type == 'dt':
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=variant["batch_size"],
                data_loader=trajectory_dataloader,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
                eval_fns=[
                    eval_episodes(
                        tar, trajectory_data_info, variant['num_eval_episodes'], model_type, variant["env"], device=device
                    ) for tar in trajectory_data_info["env_targets"]
                ],
                device=device,
                max_iters = variant["max_iters"],
                eval_interval = variant["eval_interval"],
            )
        elif model_type == 'dif':
            pass
        elif model_type == 'bc':
            trainer = ActTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=variant["batch_size"],
                data_loader=trajectory_dataloader,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
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
    else:
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
    parser.add_argument('--batch_size', type=int, default=1024)
    # Model
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--max_ep_len', type=int, default=500)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--stochastic_policy', action='store_true', default=False)
    parser.add_argument("--init_temperature", type=float, default=0.1)
    # Optimize
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    # Train
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=1)
    # Tuning
    parser.add_argument('--tuning_type', type=str, default='')
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--deep_tuning', action='store_true', default=False)
    # Eval
    parser.add_argument('--evaluation', action='store_true', default=False)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=50)
    # Etc
    parser.add_argument('--save_path', type=str, default='saved_model/pretrained')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='pretrained_model')
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
