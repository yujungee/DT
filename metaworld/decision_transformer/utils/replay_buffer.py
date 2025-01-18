"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""
# Base Packages
import copy
# Installed Packages
import numpy as np
import torch
# Working Sources
from decision_transformer.evaluation.evaluate_episodes import (
    vec_rollout_episode_rtg,
)


class ReplayBuffer(object):
    def __init__(self, capacity, data_info, device="cuda"):
        self.safe_traj_num = 20
        trajectories = data_info["trajectories"][:self.safe_traj_num]
        self.state_dim = data_info["state_dim"]
        self.act_dim = data_info["act_dim"]
        self.reward_scale = data_info["train_scale"]
        self.state_mean = data_info["state_mean"]
        self.state_std = data_info["state_std"]
        self.device = device
        self.capacity = capacity
        if len(trajectories) <= self.capacity+self.safe_traj_num:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity-self.safe_traj_num:]
            ]

        self.start_idx = 0
        self.total_transitions_sampled = 0

        self.check_sample = copy.deepcopy(self.trajectories[0]["observations"])

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        if len(self.trajectories) < self.capacity+self.safe_traj_num:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity-self.safe_traj_num :]
        else:
            print("trajectory replaced...")
            self.trajectories[
                self.start_idx+self.safe_traj_num : self.start_idx+self.safe_traj_num + len(new_trajs)
            ] = new_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        assert len(self.trajectories) <= self.capacity+self.safe_traj_num
        assert (self.check_sample == self.trajectories[0]["observations"]).all()
        
    
    def augment_trajectories(self, online_envs, target_explore, model, randomized=False):

        max_ep_len = 200

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, successes, trajs, _ = vec_rollout_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                model,
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
            )

        self.add_new_trajs(trajs)
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
            "aug_traj/success": np.mean(successes),
        }