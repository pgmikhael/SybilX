import warnings
import torch
from torch import optim
from sybilx.utils.registry import register_object


@register_object("reduce_on_plateau", "scheduler")
class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    """

    def __init__(self, optimizer, args):
        super().__init__(
            optimizer, patience=args.patience, factor=args.lr_decay, mode="min" if "loss" in args.monitor else "max"
        )


@register_object("exponential_decay", "scheduler")
class ExponentialLR(optim.lr_scheduler.ExponentialLR):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR
    """

    def __init__(self, optimizer, args):
        super().__init__(optimizer, gamma=args.lr_decay)


@register_object("cosine_annealing", "scheduler")
class CosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    """

    def __init__(self, optimizer, args):
        super().__init__(optimizer, args.cosine_annealing_period)


@register_object("cosine_annealing_restarts", "scheduler")
class CosineAnnealingWarmRestarts(optim.lr_scheduler.CosineAnnealingWarmRestarts):
    """
    https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    """

    def __init__(self, optimizer, args):
        super().__init__(
            optimizer,
            T_0=args.cosine_annealing_period,
            T_mult=args.cosine_annealing_period_scaling,
        )

@register_object("linear_warmup", "scheduler")
class LinearWarmupScheduler(optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, args):
        """[summary]

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            [description]
        warmup : int
            [description]
        last_epoch : int, optional
            [description], by default -1
        verbose : bool, optional
            [description], by default False
        """

        def lr_func(step: int):
            if step < args.warmup:
                return float(step) / float(max(1.0, args.warmup))
            return 1.0

        super().__init__(optimizer, lr_func, last_epoch=args.warmup_last_epoch)

@register_object("warmup_and_reduce_on_plateau", "scheduler")
class WarmupAndPlateauScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    """
    """

    def __init__(self, optimizer, args):
        super(WarmupAndPlateauScheduler, self).__init__(
            optimizer, patience=args.patience, factor=args.lr_decay, mode="min" if "loss" in args.monitor else "max"
        )
        self.warmup = args.warmup

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(optim.lr_scheduler.EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch
        
        if epoch <= self.warmup:
            self._warmup_lr(epoch)
            # TODO: should this skip all the following early-stopping code?
        else:

            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _warmup_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            lr_multiplier = float(epoch) / float(max(1.0, self.warmup))
            new_lr = old_lr * lr_multiplier
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr

