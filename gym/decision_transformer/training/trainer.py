import numpy as np
import torch

import time
import types


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.device = device

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, max_iters=10):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            if isinstance(self.get_batch, types.GeneratorType):
                states, actions, rewards, dones, rtg, timesteps, attention_mask = next(self.get_batch)
                states, actions, rewards, dones, rtg, timesteps, attention_mask =\
                 states.to(self.device), actions.to(self.device), rewards.to(self.device), dones.to(self.device), rtg.to(self.device), timesteps.to(self.device), attention_mask.to(self.device)
            else:
                states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
            
            train_loss = self.train_step(
                states, actions, rewards, dones, rtg, timesteps, attention_mask
            )
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        if iter_num == max_iters:
            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v

            logs['time/total'] = time.time() - self.start_time
            logs['time/evaluation'] = time.time() - eval_start
            logs['training/train_loss_mean'] = np.mean(train_losses)
            logs['training/train_loss_std'] = np.std(train_losses)

            for k in self.diagnostics:
                logs[k] = self.diagnostics[k]

            if print_logs:
                print('=' * 80)
                print(f'Iteration {iter_num}')
                for k, v in logs.items():
                    print(f'{k}: {v}')

            return logs
    
    def save_model(self, env_name, postfix, folder):
        model_name = '/model_' + env_name + postfix
        torch.save(self.model.state_dict(), folder + model_name)  # model save
        print('model saved to ', folder + model_name)