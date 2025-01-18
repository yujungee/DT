import gym
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
import argparse
import pickle
import json
from collections import namedtuple
import random
import datetime
from pathlib import Path
import os

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_episode_rtg_dif
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.ia3_dt import Ia3DecisionTransformer
from decision_transformer.models.lora_dt import LoraDecisionTransformer
from decision_transformer.models.adapter_dt import AdapterDecisionTransformer



from decision_transformer.models.decision_transformer import SoftPromptDecisionTransformer

# from decision_transformer.models.decision_diffusion_transformer import DecisionDiffusionTransformer, update_ema, requires_grad
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
# from decision_transformer.training.gen_trainer import GenerationTrainer
# add
from decision_transformer.envs.meta_world_envs import TaskEnv, MT50_TASK_MEAN, MT50_TASK_STD
# from decision_transformer.data.data_loader import create_dataloader


def seed_fix(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def cycle(dl):
    while True:
        for data in dl:
            yield data


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

    save_path = os.path.join(variant['save_path'], str_datetime+"_"+env_name+"_"+variant['tuning_type'])
    print("Save Path:", save_path)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    seed_fix(variant["seed"])

    config_path_dict = {
        'ML10': "/home/yujung/DT/gym/config/ML10/ML10.json",
        'MT10': "/home/yujung/DT/gym/config/MT10/MT10.json",
        'ML45': "/home/yujung/DT/gym/config/ML45/ML45.json",
        'MT50': "/home/yujung/DT/gym/config/MT50/MT50.json"
    }

    fine_config_path_dict = {
        'ML10': "/home/yujung/DT/gym/config/ML10/ML10_fine.json",
        'MT10': "/home/yujung/DT/gym/config/MT10/MT10.json",
        'ML45': "/home/yujung/DT/gym/config/ML45/ML45.json",
        'MT50': "/home/yujung/DT/gym/config/MT50/MT50.json"
    }

    if env_name[:2] == 'MT':
        with open(fine_config_path_dict[variant["env"]], 'r') as f:
            task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        # example task
        env_name_list = task_config.train_tasks
        max_ep_len = 500
        env_list = []
        for env_name in env_name_list:
            env = TaskEnv(task=env_name, seed=variant["seed"], mode="eval", max_timestep=max_ep_len)
            env_list.append(env)
        env_targets = [4500]  # evaluation conditioning targets
        train_scale = 1500.  # normalization for rewards/returns
        test_scale = 4500.  # normalization for rewards/returns
    elif env_name[:2] == 'ML':
        # train_env = gym.make('Hopper-v3')
        # test_env = gym.make('Hopper-v3')
        with open(fine_config_path_dict[variant["env"]], 'r') as f:
            task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        # example task
        env_name_list = task_config.train_tasks
        test_env_name_list = task_config.test_tasks
        max_ep_len = 500
        env_list = []
        for env_name in env_name_list:
            env = TaskEnv(task=env_name, seed=variant["seed"], mode="eval", max_timestep=max_ep_len)
            env_list.append(env)
        test_env_list = []
        for env_name in test_env_name_list:
            env = TaskEnv(task=env_name, seed=variant["seed"], mode="eval", max_timestep=max_ep_len)
            test_env_list.append(env)
        env_targets = [4500]  # evaluation conditioning targets
        train_scale = 1500.  # normalization for rewards/returns
        test_scale = 4500.  # normalization for rewards/returns
    elif env_name[-2:] == 'v2':
        env_name_list = [env_name]
        max_ep_len = 500
        env_list = [TaskEnv(task=env_name, seed=variant["seed"], mode="eval", max_timestep=max_ep_len)]
        env_targets = [4500]  # evaluation conditioning targets
        train_scale = 1500.  # normalization for rewards/returns
        test_scale = 4500.  # normalization for rewards/returns
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env_list[0].observation_space.shape[0]
    act_dim = env_list[0].action_space.shape[0]
    action_range = [
        float(env_list[0].action_space.low.min()) + 1e-6,
        float(env_list[0].action_space.high.max()) - 1e-6,
    ]

    # load dataset
    trajectories = []
    for env_name in env_name_list:
        dataset_path = f'data/MT50-Offline_dataset/{env_name}/{env_name[:-3]}_train_dataset.pkl'
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        traj_lens, returns, successes = [], [], []
        for path in data:
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
            successes.append(1 if path["success"].mean() > 0 else 0)
            # action scaling
            path["actions"] = path["actions"].clip(action_range[0], action_range[1])

        traj_lens, returns, successes = np.array(traj_lens), np.array(returns), np.array(successes)
        
        num_timesteps = sum(traj_lens)
        print('=' * 50)
        print(f'Starting new experiment: {env_name} {dataset}')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f"Average success: {np.mean(successes)*100:.2f}%")
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print('=' * 50)
        
        trajectories += data

    # save all path information into separate lists
    env_name, dataset = variant['env'], variant['dataset']
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns, successes = [], [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        successes.append(1 if path["success"].mean() > 0 else 0)
    traj_lens, returns, successes = np.array(traj_lens), np.array(returns), np.array(successes)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    # state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    # print(state_mean)
    # print(state_std)
    # exit()
    state_mean, state_std = np.array(MT50_TASK_MEAN), np.array(MT50_TASK_STD)
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f"Average success: {np.mean(successes)*100:.2f}%")
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
    print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    
    if model_type == 'dt':
        if tuning_type == "prompt":
            model = SoftPromptDecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len,
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
            )

        elif tuning_type == "ia3":
            model = Ia3DecisionTransformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    max_length=K,
                    max_ep_len=max_ep_len,
                    hidden_size=variant['embed_dim'],
                    n_layer=variant['n_layer'],
                    n_head=variant['n_head'],
                    n_inner=4*variant['embed_dim'],
                    activation_function=variant['activation_function'],
                    n_positions=1024,
                    resid_pdrop=variant['dropout'],
                    attn_pdrop=variant['dropout'],
                )

        elif tuning_type == "lora":
            model = LoraDecisionTransformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    max_length=K,
                    max_ep_len=max_ep_len,
                    hidden_size=variant['embed_dim'],
                    n_layer=variant['n_layer'],
                    n_head=variant['n_head'],
                    n_inner=4*variant['embed_dim'],
                    activation_function=variant['activation_function'],
                    n_positions=1024,
                    resid_pdrop=variant['dropout'],
                    attn_pdrop=variant['dropout'],
                )
            
        elif tuning_type == 'adapter':
            model = AdapterDecisionTransformer(
                    state_dim=state_dim,
                    act_dim=act_dim,
                    max_length=K,
                    max_ep_len=max_ep_len,
                    hidden_size=variant['embed_dim'],
                    n_layer=variant['n_layer'],
                    n_head=variant['n_head'],
                    n_inner=4*variant['embed_dim'],
                    activation_function=variant['activation_function'],
                    n_positions=1024,
                    resid_pdrop=variant['dropout'],
                    attn_pdrop=variant['dropout'],
            )

        else:
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
                stochastic_policy = variant['stochastic_policy'],
            )
            if variant['tuning_type'] != "":
                saved_model_path = variant['model_path']
                model.load_state_dict(torch.load(saved_model_path), strict=False)
                print('model initialized from: ', saved_model_path)

    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    if model_type in ["dt", "bc"]:
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
    elif model_type == 'dif':
        model = model.to(device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=variant['learning_rate'],
            weight_decay=0
        )


    if variant['finetune']:

        saved_model_path = 'saved_model/multitask/2023-07-30_15:04:30_ML10_/model_ML10_TRAIN_expert_iter_9'
        model.load_state_dict(torch.load(saved_model_path), strict=False)

        ##### fine tuning parameter update #####
        if variant['tuning_type'] != None:
            print(variant['tuning_type'])
            for name, param in model.named_parameters():
                if variant['tuning_type'] not in name:
                    param.requires_grad_(False)

            enabled = set()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            print(f"Parameters to be updated: {enabled}")

        #########################################

        
    def data_loader(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / train_scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model, diffusion=None):
            print("Evaluation ...")
            results = {}
            total_returns, total_lengths, total_successes = [], [], []
            for i, env in enumerate(env_list):
                env_name = env_name_list[i]
                returns, lengths, successes = [], [], []
                for _ in tqdm(range(num_eval_episodes)):
                    with torch.no_grad():
                        if model_type == 'dt':
                            ret, length, success = evaluate_episode_rtg(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                scale=test_scale,
                                target_return=target_rew/test_scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                            )
                        elif model_type == 'dif':
                            ret, length, success = evaluate_episode_rtg_dif(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                diffusion, 
                                max_length=K,
                                max_ep_len=max_ep_len,
                                scale=test_scale,
                                target_return=target_rew/test_scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                            )
                        else:
                            ret, length, success = evaluate_episode(
                                env,
                                state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                target_return=target_rew/test_scale,
                                mode=mode,
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                            )
                    returns.append(ret)
                    lengths.append(length)
                    successes.append(success)
                result = {
                    f'target_{env_name}': i,
                    f'target_{env_name}_return_mean': np.mean(returns),
                    f'target_{env_name}_return_std': np.std(returns),
                    f'target_{env_name}_length_mean': np.mean(lengths),
                    f'target_{env_name}_length_std': np.std(lengths),
                    f'target_{env_name}_success_mean': np.mean(successes),
                    # f'target_{env_name}_success_std': np.std(successes),
                }
                results.update(result)
                
                total_returns += returns
                total_lengths += lengths
                total_successes += successes
            result = {
                    f'total_{variant["env"]}': 0,
                    f'total_{variant["env"]}_return_mean': np.mean(total_returns),
                    f'total_{variant["env"]}_return_std': np.std(total_returns),
                    f'total_{variant["env"]}_length_mean': np.mean(total_lengths),
                    f'total_{variant["env"]}_length_std': np.std(total_lengths),
                    f'total_{variant["env"]}_success_mean': np.mean(total_successes),
                    f'total_{variant["env"]}_success_std': np.std(total_successes),
            }
            results.update(result)
            return results
        return fn

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=data_loader,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            device=device,
        )

    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=data_loader,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            device=device,
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug


    max_iters = variant['max_iters']

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True, max_iters=max_iters)
        if log_to_wandb:
            wandb.log(outputs)

        if iter % variant['save_interval'] == 0:
            trainer.save_model(
                env_name = variant["env"],
                postfix = model_post_fix+'_iter_'+str(iter),
                folder = save_path
            )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument('--env', type=str, default='reach-v2')
    parser.add_argument('--dataset', type=str, default='expert')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--stochastic_policy', action='store_true', default=False)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=50)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='saved_model')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='pretrained_model')
    parser.add_argument('--tuning_type', type=str, default='')
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--deep_tuning', action='store_true', default=False)
    parser.add_argument('--finetune', type=bool, default=False)

    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
