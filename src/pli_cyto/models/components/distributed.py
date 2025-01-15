from typing import Optional, Union, Any

import torch
from torch import Tensor
import torch.nn as nn

if torch.distributed.is_available():
    from torch.distributed import ReduceOp


#########################
# From pytorch lightning

def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def sync_ddp(
    result: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
) -> torch.Tensor:
    """Function to reduce the tensors from several ddp processes to one main process.

    Args:
        result: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        reduced value
    """
    divide_by_world_size = False

    if group is None:
        group = torch.distributed.group.WORLD

    if isinstance(reduce_op, str):
        if reduce_op.lower() in ("avg", "mean"):
            op = ReduceOp.SUM
            divide_by_world_size = True
        else:
            op = getattr(ReduceOp, reduce_op.upper())
    else:
        op = reduce_op

    # sync all processes before reduction
    torch.distributed.barrier(group=group)
    torch.distributed.all_reduce(result, op=op, group=group, async_op=False)

    if divide_by_world_size:
        result = result / torch.distributed.get_world_size(group)

    return result


def sync_ddp_if_available(
    result: torch.Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
) -> torch.Tensor:
    """Function to reduce a tensor across worker processes during distributed training.

    Args:
        result: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.

    Return:
        reduced value
    """
    if distributed_available():
        return sync_ddp(result, group=group, reduce_op=reduce_op)
    return result

#
#########################

class RunningNorm2D(nn.Module):
    """
    Compute running statistics on the fly over whole training

    Parts are copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d

    Args:
        num_features (int): numver of input features (or channels)
        max_iter (int): maximum iterations after which statistics are not updated anymore and kept constant
    """

    def __init__(self, num_features, max_iter=1024, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RunningNorm2D, self).__init__()
        self.num_features = num_features
        self.max_iter = max_iter
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.num_batches_tracked: Optional[Tensor]
        self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
        self.register_buffer('running_var', torch.zeros(num_features, **factory_kwargs))
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]

    def forward(self, input):
        self._check_input_dim(input)

        if self.training and self.num_batches_tracked < self.max_iter:
            # Detach input since no gradient is required for running stats
            self._update_stats(input.detach())

        if self.num_batches_tracked == 0:
            input_norm = input
        else:
            input_norm = (input - self.running_mean[:, None, None]) / \
                         (torch.sqrt(self.running_var)[:, None, None] + 1e-8)

        return input_norm

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_var.zero_()
        self.num_batches_tracked.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def _update_stats(self, input):
        self.num_batches_tracked += 1

        n_mean = torch.mean(input, dim=(0, 2, 3))
        n_mean = sync_ddp_if_available(n_mean, reduce_op='mean')
        self.running_mean += (n_mean - self.running_mean) / float(self.num_batches_tracked)

        n_var = torch.mean((input - self.running_mean[:, None, None]) ** 2, dim=(0, 2, 3))
        n_var = sync_ddp_if_available(n_var, reduce_op='mean')
        self.running_var += (n_var - self.running_var) / float(self.num_batches_tracked)
