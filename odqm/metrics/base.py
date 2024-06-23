import torch
import torch.nn as nn
from tqdm import tqdm

from odqm.data import ReplayBuffer
from odqm.utils import clean_data

import wandb


class BaseMetric(nn.Module):
    def __init__(self, train_steps, eval_freq, device=None, batch_size=16, **kwargs):
        super().__init__()
        self.train_steps = train_steps
        self.eval_freq = eval_freq
        self.batch_size = batch_size

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_metric(self, data):
        raise NotImplementedError

    def eval_metric(self, data):
        raise NotImplementedError

    def estimate(self, data: ReplayBuffer):
        data.to(self.device)
        for i in tqdm(range(self.train_steps)):
            train_data = data.sample(self.batch_size)
            train_logs = self.train_metric(train_data)
            wandb.log(train_logs, step=i)
            clean_data(train_data)

            if (i + 1) % self.eval_freq == 0:
                val_data = data.sample(self.batch_size)
                val_logs = self.eval_metric(val_data)
                clean_data(val_data)
                wandb.log(val_logs, step=(i+1)//self.eval_freq)