"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time


class OnlineTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        loss_fn=None,
        il_alpha=0.0,
        batch_size=None,
        data_loader=None,
        eval_fns=None,
        device="cuda",
        max_iters=None,
        eval_interval=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.il_alpha = il_alpha
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.device = device
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.data_loader.get_batch_rand(self.batch_size)

            loss, nll, entropy, il_loss = self.train_step_stochastic(
                states, actions, rewards, dones, rtg, timesteps, attention_mask
            )
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/il_loss"] = il_loss
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        if (iter_num in list(range(0, self.max_iters, self.eval_interval))) or (iter_num == self.max_iters):

            eval_start = time.time()

            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            logs['time/total'] = time.time() - self.start_time
            logs['time/evaluation'] = time.time() - eval_start        

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step_stochastic(self, states, actions, rewards, dones, rtg, timesteps, attention_mask):

        action_target = torch.clone(actions)
        
        _, action_preds, _ = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        loss, nll, entropy, il_loss = self.loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            attention_mask,
            self.model.temperature().detach(),  # no gradient taken here
            il_alpha = self.il_alpha,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            il_loss.detach().cpu().item(),
        )

    def online_train_iteration(
        self,
        dataloader,
    ):
        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy, il_loss = self.online_train_step_stochastic(self.loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/il_loss"] = il_loss
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def online_train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        attention_mask = attention_mask.to(self.device)

        action_target = torch.clone(actions)
        
        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        loss, nll, entropy, il_loss = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            attention_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            il_loss.detach().cpu().item(),
        )
    
    def save_model(self, env_name, postfix, folder):
        model_name = '/model_' + env_name + postfix
        torch.save(self.model.state_dict(), folder + model_name)  # model save
        print('model saved to ', folder + model_name)