# Installed Packages
from tqdm.auto import tqdm
import numpy as np
import torch

def eval_episodes(target_rew, data_info, num_eval_episodes, model_type, exp_env, max_ep_len=500, device=None):
    env_list = data_info["env_list"]
    env_name_list = data_info["env_name_list"]
    state_dim = data_info["state_dim"]
    act_dim = data_info["act_dim"]
    test_scale = data_info["test_scale"]
    state_mean = data_info["state_mean"]
    state_std = data_info["state_std"]
    test_env_list = data_info["test_env_list"]
    test_env_name_list = data_info["test_env_name_list"]
    def fn(model):
        print("Train Evaluation ...")
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
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                successes.append(success)
            result = {
                f'TrainEnv_{env_name}': i,
                f'TrainEnv_{env_name}_return_mean': np.mean(returns),
                f'TrainEnv_{env_name}_return_std': np.std(returns),
                f'TrainEnv_{env_name}_length_mean': np.mean(lengths),
                f'TrainEnv_{env_name}_length_std': np.std(lengths),
                f'TrainEnv_{env_name}_success_mean': np.mean(successes),
            }
            results.update(result)
            
            total_returns += returns
            total_lengths += lengths
            total_successes += successes
        result = {
                f'TotalTrainEnv_{exp_env}': 0,
                f'TotalTrainEnv_{exp_env}_return_mean': np.mean(total_returns),
                f'TotalTrainEnv_{exp_env}_return_std': np.std(total_returns),
                f'TotalTrainEnv_{exp_env}_length_mean': np.mean(total_lengths),
                f'TotalTrainEnv_{exp_env}_length_std': np.std(total_lengths),
                f'TotalTrainEnv_{exp_env}_success_mean': np.mean(total_successes),
                f'TotalTrainEnv_{exp_env}_success_std': np.std(total_successes),
        }
        results.update(result)

        if test_env_name_list is not None:
            print("Test Evaluation ...")
            total_returns, total_lengths, total_successes = [], [], []
            for i, env in enumerate(test_env_list):
                env_name = test_env_name_list[i]
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
                                state_mean=state_mean,
                                state_std=state_std,
                                device=device,
                            )
                    returns.append(ret)
                    lengths.append(length)
                    successes.append(success)
                result = {
                    f'TestEnv_{env_name}': i,
                    f'TestEnv_{env_name}_return_mean': np.mean(returns),
                    f'TestEnv_{env_name}_return_std': np.std(returns),
                    f'TestEnv_{env_name}_length_mean': np.mean(lengths),
                    f'TestEnv_{env_name}_length_std': np.std(lengths),
                    f'TestEnv_{env_name}_success_mean': np.mean(successes),
                }
                results.update(result)
                
                total_returns += returns
                total_lengths += lengths
                total_successes += successes
            result = {
                    f'TotalTestEnv_{exp_env}': 0,
                    f'TotalTestEnv_{exp_env}_return_mean': np.mean(total_returns),
                    f'TotalTestEnv_{exp_env}_return_std': np.std(total_returns),
                    f'TotalTestEnv_{exp_env}_length_mean': np.mean(total_lengths),
                    f'TotalTestEnv_{exp_env}_length_std': np.std(total_lengths),
                    f'TotalTestEnv_{exp_env}_success_mean': np.mean(total_successes),
                    f'TotalTestEnv_{exp_env}_success_std': np.std(total_successes),
            }
            results.update(result)
        return results
    return fn

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length, cur_success = 0, 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1
        if "success" in info.keys():
            cur_success = max(cur_success, info["success"])

        if done:
            break

    return episode_return, episode_length, cur_success


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length, cur_success = 0, 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        action = action.reshape(act_dim,)
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1
        if "success" in info.keys():
            cur_success = max(cur_success, info["success"])

        if done:
            break

    return episode_return, episode_length, cur_success


@torch.no_grad()
def vec_rollout_episode_rtg(
    vec_env,
    state_dim,
    act_dim,
    model,
    target_return: list,
    max_ep_len=200,
    reward_scale=1500,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
):
    assert len(target_return) == vec_env.num_envs

    model.eval()
    model.to(device=device)
    
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)
    episode_success = np.zeros((num_envs, 1)).astype(float)

    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )

        # the return action is a SquashNormal distribution
        action = action.reshape(num_envs, act_dim)
        action = action.clamp(*model.action_range)

        state, reward, done, infos = vec_env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)
        if "success" in infos[0].keys():
            for i in range(len(infos)):
                episode_success[i] = np.maximum(episode_success[i], infos[i]["success"])

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break


    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        episode_success.reshape(num_envs),
        trajectories,
        infos[0]["task_name"] if "task_name" in infos[0].keys() else ""
    )