# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist
from datetime import timedelta

# When world_size = sp_size * dp_size, all_to_all runs only within the SP group; the SP process group is stored here.
_sp_group = None


def init_distributed_group():
    """r initialize sequence parallel group.

    timeout=2h: the first torch.compile trace + flex_attention compile + first data
    batch can each exceed the default 10 min; rank 0 would still be compiling kernels
    while rank 1 waits in all_to_all and gets killed by the NCCL watchdog. Allow a 2h
    buffer.
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(hours=2),
        )


def set_sequence_parallel_group(group):
    """Set the SP process group. Afterwards get_rank()/get_world_size() return the in-group rank/size and all_to_all uses this group."""
    global _sp_group
    _sp_group = group


def get_rank():
    """Return the current rank: the in-group rank if an SP group is set, otherwise the global rank."""
    if _sp_group is not None:
        return dist.get_rank(_sp_group)
    return dist.get_rank()


def get_world_size():
    """Return the current world size: the in-group size if an SP group is set, otherwise the global world size."""
    if _sp_group is not None:
        return dist.get_world_size(_sp_group)
    return dist.get_world_size()


def _resolve_group(group):
    """Group used for communication: an explicitly passed group wins, otherwise the SP group (if set)."""
    if group is not None:
        return group
    return _sp_group


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    With group=None, communication stays within the SP group if set_sequence_parallel_group was called.
    """
    group = _resolve_group(group)
    world_size = dist.get_world_size(group) if group is not None else dist.get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


class AllToAllWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        world_size = dist.get_world_size(group)

        # 1. split the input
        inputs = [u.contiguous() for u in input_tensor.chunk(world_size, dim=scatter_dim)]
        # 2. prepare the output containers
        outputs = [torch.empty_like(u) for u in inputs]
        
        # 3. run the native distributed communication (no gradients, pure data movement)
        dist.all_to_all(outputs, inputs, group=group)

        # 4. concatenate and return
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        scatter_dim = ctx.scatter_dim
        gather_dim = ctx.gather_dim
        group = ctx.group
        world_size = dist.get_world_size(group)

        # The backward pass mirrors the forward pass:
        # forward:  scatter(dim2) -> gather(dim1)
        # backward: scatter(dim1) -> gather(dim2)
        
        # 1. split the incoming gradient along gather_dim
        inputs = [u.contiguous() for u in grad_output.chunk(world_size, dim=gather_dim)]
        outputs = [torch.empty_like(u) for u in inputs]

        # 2. run the inverse all_to_all
        dist.all_to_all(outputs, inputs, group=group)

        # 3. concatenate back along scatter_dim
        grad_input = torch.cat(outputs, dim=scatter_dim).contiguous()

        # the number of return values must match forward's inputs (except ctx)
        # input_tensor, scatter_dim, gather_dim, group
        return grad_input, None, None, None

def all_to_all_with_grad(x, scatter_dim, gather_dim, group=None):
    group = _resolve_group(group)
    world_size = dist.get_world_size(group) if group is not None else dist.get_world_size()
    if world_size <= 1:
        return x
    return AllToAllWithGrad.apply(x, scatter_dim, gather_dim, group)


def all_gather(tensor, group=None):
    group = _resolve_group(group)
    world_size = dist.get_world_size(group) if group is not None else dist.get_world_size()
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    return tensor_list


def gather_forward(input, dim, group=None):
    group = _resolve_group(group)
    world_size = dist.get_world_size(group) if group is not None else dist.get_world_size()
    if world_size == 1:
        return input
    output = all_gather(input, group=group)
    return torch.cat(output, dim=dim).contiguous()
