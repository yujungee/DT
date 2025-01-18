# Base Packages
import random
import pickle
import json
from collections import namedtuple
from copy import deepcopy
# Installed Packages
import numpy as np
import torch
# Working Sources
from decision_transformer.envs.meta_world_envs import (
    TaskEnv,
    MT50_TASK_MEAN,
    MT50_TASK_STD,
)


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

def data_preprocessing(variant, ):
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']

    config_path_dict = {
        'ML10': "config/ML10/ML10.json",
        'ML30': "config/ML30/ML30.json",
        'ML35': "config/ML35/ML35.json",
        'MT10': "config/MT10/MT10.json",
        'ML45': "config/ML45/ML45.json",
        'MT50': "config/MT50/MT50.json"
    }

    if env_name[:2] == 'MT':
        with open(config_path_dict[variant["env"]], 'r') as f:
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
        test_env_name_list = None
        test_env_list = None
    elif env_name[:2] == 'ML':
        with open(config_path_dict[variant["env"]], 'r') as f:
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
        test_env_name_list = None
        test_env_list = None
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
    if variant["state_norm"] == "cal_norm":
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    elif variant["state_norm"] == "fixed_norm":
        # state_mean, state_std = np.array(MT10_TASK_MEAN), np.array(MT10_TASK_STD)
        state_mean, state_std = np.array(MT50_TASK_MEAN), np.array(MT50_TASK_STD)
    else:
        state_mean, state_std = 0, 1

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

    data_infos = {
        "trajectories": trajectories,
        "num_trajectories": num_trajectories,
        "state_dim": state_dim,
        "act_dim": act_dim,
        "action_range": action_range,
        "state_mean": state_mean,
        "state_std": state_std,
        "sorted_inds": sorted_inds,
        "p_sample": p_sample,
        "env_name_list": env_name_list,
        "env_list": env_list,
        "env_targets": [4500],
        "train_scale": 1500,
        "test_scale": 4500,
        "test_env_name_list": test_env_name_list,
        "test_env_list": test_env_list,
    }

    return data_infos

def loss_fn(a_hat_dist, a, attention_mask, entropy_reg, il_alpha=0.0):
    # a_hat is a SquashedNormal Distribution
    log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

    entropy = a_hat_dist.entropy().mean()
    loss = -(log_likelihood + entropy_reg * entropy)

    # imitation loss
    if il_alpha:
        a_hat = a_hat_dist.rsample()
        il_loss = il_alpha * torch.mean((a_hat - a) ** 2)
        loss += il_loss
    else:
        il_loss = torch.zeros(1)

    return (
        loss,
        -log_likelihood,
        entropy,
        il_loss,
    )


def get_env_builder(seed, env_name, target_goal=None, mode="train", shuffle=0):
    def make_env_fn():
        env = TaskEnv(env_name, seed = seed, max_timestep = 200, mode = mode, shuffle = shuffle)

        if target_goal:
            env.target_goal = target_goal
            print(f"Set the target goal to be {env.target_goal}")
        return env

    return make_env_fn


def copy_some_parameters(source_model, target_model):
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()

    for name, param in source_state_dict.items():
        if name in target_state_dict:
            target_state_dict[name].copy_(param)
        else:
            raise ValueError