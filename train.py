import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from custom.criterion import smooth_cross_entropy
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
from torch import nn
from torch.optim import Adam
import os

class Trainer:
    def __init__(
        self,
        mt, 
        device,
        midi_encoder,
        train_loader,
        valid_loader,
        log_dir,
        model_dir,
        num_epochs,
        max_lr=1e-3
    ):
        self.mt = mt
        self.optimizer = Adam(
            mt.parameters(), 
            lr=1e-4
        )
        self.device = device
        self.midi_encoder = midi_encoder
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.current_epoch = 0
        self.last_epoch = num_epochs
        self.scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=max_lr,
            total_steps=num_epochs
        )
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.summary = SummaryWriter(log_dir=log_dir)

    def get_criterion(self):
        def criterion(input, target):
            input_flat = input.view(-1, self.midi_encoder.vocab_size)
            return smooth_cross_entropy(input_flat, target.flatten())
        return criterion

    def find_lr(self):
        lr_finder = LRFinder(
            self.mt, self.optimizer, self.get_criterion(), device=self.device)
        lr_finder.range_test(self.train_loader, end_lr=10, num_iter=100)
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset()

    def run(self):
        mt = self.mt
        epochs = range(self.current_epoch, self.last_epoch)
        criterion = self.get_criterion()
        with tqdm(total=len(epochs)) as pbar:
            train_loss = []
            eval_loss = []
            for e in epochs:
                self.epoch = e
                mt.train()
                for batch_x, batch_y in self.train_loader:
                    self.optimizer.zero_grad()
                    output = mt(batch_x.to(self.device))
                    loss = criterion(output, batch_y.to(self.device))
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(mt.parameters(), 0.5)
                    self.optimizer.step()
                    train_loss.append(loss.item())
                
                with torch.no_grad():
                    mt.eval()
                    for batch_x, batch_y in self.valid_loader:
                        output = mt(batch_x.to(self.device))
                        loss = criterion(output, batch_y.to(self.device))
                        eval_loss.append(loss.item())
                self.scheduler.step()
                pbar.set_description(
                    f"[{e}] "
                    f"loss={np.mean(train_loss):.2f} "
                    f"eval loss={np.mean(eval_loss):.2f} "
                )
                pbar.update()
                lr = self.optimizer.param_groups[0]['lr']
                self.summary.add_scalar("loss/train", np.mean(train_loss), global_step=e)
                self.summary.add_scalar("loss/eval", np.mean(eval_loss), global_step=e)
                self.summary.add_scalar("lr", lr, global_step=e)
                self.summary.flush()
                if e % 100 == 0:
                    torch.save(mt.state_dict(), os.path.join(self.model_dir, f'{e}.pth'))
