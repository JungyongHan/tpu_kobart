import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



class WarmupAndExponentialDecayScheduler(_LRScheduler):
  """Update the learning rate of wrapped optimizer based on epoch and step.

  Args:
    optimizer: Instance of torch.optim.Optimizer. Learning rate will be changed.
    num_steps_per_epoch: int, the number of steps required to finish 1 epoch.
    divide_every_n_epochs: After this number of epochs, learning rate will be
      divided by the `divisor` param.
    divisor: The learning rate will be divided by this amount when epoch %
      divide_every_n_epochs == 0 (epoch 0 is excluded).
    num_warmup_epochs: Float. Learning rate will ramp up from 0 to max learning
      rate over this many epochs. Note that partial epochs are allowed, e.g. 0.5
      epochs.
    min_delta_to_update_lr: If the new learning rate does not differ much from
      the learning rate of the previous step, don't bother updating the
      optimizer's learning rate.
    summary_writer: Instance of `torch.utils.tensorboard.SummaryWriter`. If
      provided, learning rate will be logged during calls to step if step is
      called with write_to_summary=True. If summary_writer is None, then no
      logging happens.
  """

  def __init__(self,
               optimizer,
               num_steps_per_epoch,
               divide_every_n_epochs=20,
               divisor=5,
               num_warmup_epochs=0.9,
               min_delta_to_update_lr=1e-6,
               summary_writer=None):
    self._num_steps_per_epoch = num_steps_per_epoch
    self._divide_every_n_epochs = divide_every_n_epochs
    self._divisor = divisor
    self._num_warmup_epochs = num_warmup_epochs
    self._min_delta_to_update_lr = min_delta_to_update_lr
    self._previous_lr = -1
    self._max_lr = optimizer.param_groups[0]['lr']
    self._summary_writer = summary_writer
    super(WarmupAndExponentialDecayScheduler, self).__init__(optimizer)

  def _epoch(self):
    return self._step_count // self._num_steps_per_epoch

  def _is_warmup_epoch(self):
    return self._epoch() < math.ceil(self._num_warmup_epochs)

  def get_lr(self):
    epoch = self._epoch()
    lr = 0.0

    if self._is_warmup_epoch():
      # Ramp up learning rate from 0.0 to self._max_lr using a linear slope.
      num_warmup_steps = self._num_warmup_epochs * self._num_steps_per_epoch
      lr = min(self._max_lr,
               self._max_lr * ((self._step_count + 1.0) / num_warmup_steps))
    else:
      # Normal epoch. Use an exponential decay determined by init params.
      lr = self._max_lr / (
          self._divisor**(epoch // self._divide_every_n_epochs))

    # _LRScheduler expects a list of learning rates like this.
    return [lr for _ in self.base_lrs]

  def step(self, epoch=None):
    current_lr = self.get_lr()[0]

    # Outside of warmup epochs, we use the same learning rate for every step
    # in an epoch. Don't bother updating learning rate if it hasn't changed.
    if abs(current_lr - self._previous_lr) >= self._min_delta_to_update_lr:
      super(WarmupAndExponentialDecayScheduler, self).step()
      self._previous_lr = current_lr
    else:
      self._step_count += 1  # This normally happens in super().step().

    # Add current learning rate to Tensorboard metrics. For warmup epochs,
    # log the learning rate at every step. For non-warmup epochs, log only
    # the first step since the entire epoch will use the same learning rate.
    if self._summary_writer:
      if self._is_warmup_epoch() or (self._step_count %
                                     self._num_steps_per_epoch == 0):
        test_utils.write_to_summary(
            self._summary_writer,
            self._step_count,
            dict_to_write={
                'LearningRate': self.optimizer.param_groups[0]['lr']
            },
            write_xla_metrics=False)