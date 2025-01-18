import numpy as np
import torch

import time
import types


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        batch_size, 
        data_loader, 
        loss_fn, 
        scheduler=None, 
        eval_fns=None, 
        max_iters=None, 
        eval_interval = 1,
        device="cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.device = device
        self.max_iters = max_iters
        self.eval_interval = eval_interval

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            states, actions, rewards, dones, rtg, timesteps, attention_mask = self.data_loader.get_batch_rand(self.batch_size)
            
            train_loss = self.train_step(
                states, actions, rewards, dones, rtg, timesteps, attention_mask
            )
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

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
    
    def save_model(self, env_name, postfix, folder):
        model_name = '/model_' + env_name + postfix
        torch.save(self.model.state_dict(), folder + model_name)  # model save
        print('model saved to ', folder + model_name)