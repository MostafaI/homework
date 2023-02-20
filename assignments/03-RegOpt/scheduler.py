from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import math


# Old Adagrad
class CustomLRScheduler(_LRScheduler):
    """
    Creates a custom made learning rate schehuler based on manually giving the learning rates
    """

    # def __init__(self, optimizer, last_epoch=-1, decay_factor=0.1, decay_epochs=10, milestones=[]):
    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        lrs=[],
        milestones=[],
        decay_factor=0.1,
        decay_epochs=10,
    ):
        """
        Create a new scheduler.
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            last_epoch (int): The index of the last epoch. Default: -1.
            decay_factor (float): Factor by which to decay the learning rate. Default: 0.1.
            decay_epochs (int): Number of epochs after which to decay the learning rate. Default: 10.

        """
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.last_epoch = last_epoch
        self.optimizer = optimizer
        self.milestones = milestones
        self.lrs = lrs
        self.prev_lrs = []
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        updates the learning rate
        """
        current_lr = [
            group["lr"] * self.decay_factor for group in self.optimizer.param_groups
        ][0]
        # if self.last_epoch ==0 : print('initial LR =',current_lr)
        # self.prev_lrs.append([group["lr"] for group in self.optimizer.param_groups][0])
        for milestone, lr in zip(self.milestones, self.lrs):
            if self.last_epoch == milestone:
                # print(self.last_epoch, 'old LR =', current_lr)
                # print('\tnew LR =', lr)
                return [lr for group in self.optimizer.param_groups]
        # if self.last_epoch == 14 * 372: plt.plot(self.prev_lrs)
        return [group["lr"] for group in self.optimizer.param_groups]
