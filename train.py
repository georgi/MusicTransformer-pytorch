import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from custom.criterion import smooth_cross_entropy
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from torch.nn.functional import cross_entropy
import os


class Trainer:
    def __init__(
        self,
        mt,
        optimizer,
        device,
        midi_encoder,
        train_loader,
        valid_loader,
        log_dir,
        model_dir,
        epoch=0,
        save_every=20
    ):
        self.mt = mt
        self.optimizer = optimizer
        self.device = device
        self.midi_encoder = midi_encoder
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.epoch = epoch
        self.save_every = save_every
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.summary = SummaryWriter(log_dir=log_dir)

    def get_criterion(self, mode='train'):
        if mode == 'train':
            loss_fn = smooth_cross_entropy
        else:
            loss_fn = cross_entropy

        def criterion(input, target):
            input_flat = input.view(-1, self.midi_encoder.vocab_size)
            return loss_fn(input_flat, target.flatten())
        return criterion

    def save_checkpoint(self):
        data = {
            'model': self.mt.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        torch.save(data, os.path.join(self.model_dir, f'{self.epoch}.pth'))
        
    def load_checkpoint(self, file=None):
        if file is None:
            fnames = [
                f
                for f in os.listdir(self.model_dir)
                if '.pth' in f
            ]
            if len(fnames) == 0:
                print("cannot find any checkpoint")
                return
            f = max(fnames, key=lambda f: int(f[:-4]))
            file = os.path.join(self.model_dir, f)
        print("Loading checkpoint from ", file)
        data = torch.load(file)
        self.epoch = data['epoch']
        self.mt.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])

    def find_lr(self):
        lr_finder = LRFinder(
            self.mt, self.optimizer, self.get_criterion(), device=self.device)
        lr_finder.range_test(self.train_loader, end_lr=10, num_iter=100)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()

    def run(self, num_epochs, max_lr):
        mt = self.mt
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=num_epochs * len(self.train_loader),
            last_epoch=self.epoch * len(self.train_loader),
        )
        train_criterion = self.get_criterion('train')
        eval_criterion = self.get_criterion('eval')
        with tqdm(total=num_epochs) as pbar:
            for e in range(self.epoch, self.epoch + num_epochs):
                self.epoch = e
                train_loss = []
                eval_loss = []
                mt.train()
                for batch_x, batch_y in self.train_loader:
                    self.optimizer.zero_grad()
                    output = mt(batch_x.to(self.device))
                    loss = train_criterion(output, batch_y.to(self.device))
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(mt.parameters(), 0.5)
                    self.optimizer.step()
                    train_loss.append(loss.item())
                    scheduler.step()

                with torch.no_grad():
                    mt.eval()
                    for batch_x, batch_y in self.valid_loader:
                        output = mt(batch_x.to(self.device))
                        loss = eval_criterion(output, batch_y.to(self.device))
                        eval_loss.append(loss.item())
                pbar.set_description(
                    f"[{e}] "
                    f"loss={np.mean(train_loss):.2f} "
                    f"eval loss={np.mean(eval_loss):.2f} "
                )
                pbar.update()
                lr = self.optimizer.param_groups[0]['lr']
                self.summary.add_scalar(
                    "loss/train", np.mean(train_loss), global_step=e)
                self.summary.add_scalar(
                    "loss/eval", np.mean(eval_loss), global_step=e)
                self.summary.add_scalar("lr", lr, global_step=e)
                self.summary.flush()
                if e % self.save_every == 0:
                    self.save_checkpoint()
