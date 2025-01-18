import time
import types
import numpy as np
import torch
from decision_transformer.models.decision_diffusion_transformer import update_ema


class GenerationTrainer:
    def __init__(self, model, ema, diffusion, optimizer, batch_size, get_batch, eval_fns=None, device="cpu"):
        self.model = model
        self.ema = ema
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.device = device

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

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
            
            t = torch.randint(0, self.diffusion.num_timesteps, (states.shape[0],), device=self.device)
            model_kwargs = dict(y=states)
            train_loss = self.train_step(
                states, actions, rewards, dones, rtg, timesteps, attention_mask, t, model_kwargs
            )
            train_losses.append(train_loss)

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model, self.diffusion)
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

    def train_step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask, t, model_kwargs):
        action_target = torch.clone(actions)
        x = (states, actions, rewards, rtg[:,:-1], timesteps, attention_mask)
        loss_dict = self.diffusion.training_losses(self.model, actions, t, model_kwargs)

        loss = loss_dict["loss"].mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_ema(self.ema, self.model)

        with torch.no_grad():
            self.diagnostics['training/action_error'] = loss.detach().cpu().item()

        return loss.detach().cpu().item()
