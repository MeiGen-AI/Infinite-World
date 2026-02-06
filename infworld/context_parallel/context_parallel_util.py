import os
import math
import random
import argparse
import datetime
import logging
import inspect
import subprocess

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from einops import rearrange, repeat


dp_size = None
cp_size = None
dp_group = None
cp_group = None
cp_stream = None
dp_ranks = None
cp_ranks = None
dp_rank = None
cp_rank = None


def init_context_parallel(context_parallel_size: int = 1,
                          global_rank: int = 1,
                          world_size: int = 1,):

    global dp_size
    global cp_size
    global dp_group
    global cp_group
    global dp_ranks
    global cp_ranks
    global dp_rank
    global cp_rank


    if world_size%context_parallel_size != 0:
        raise RuntimeError(f'world_size {world_size} must be multiple of context_parallel_size {context_parallel_size}!!!')


    cp_size = context_parallel_size
    dp_size = world_size//context_parallel_size


    print(f'[rank {global_rank}] init_device_mesh [dp_size x cp_size]: [{dp_size} x {cp_size}]')

    mesh_2d = init_device_mesh("cuda", (dp_size, cp_size), mesh_dim_names=("dp", "cp"))

    print(f'[rank {global_rank}] mesh_2d: {mesh_2d}')

    dp_group = mesh_2d.get_group(mesh_dim="dp")
    cp_group = mesh_2d.get_group(mesh_dim="cp")

    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    cp_ranks = torch.distributed.get_process_group_ranks(cp_group)

    dp_rank = dist.get_rank(group=dp_group)
    cp_rank = dist.get_rank(group=cp_group)

    global_rank_1 = torch.distributed.get_rank()
    print(f'[rank {global_rank_1}] [dp_rank, cp_rank]: [{dp_rank}, {cp_rank}],  dp_ranks: {dp_ranks}, cp_ranks: {cp_ranks}')


def get_cp_size():

    global cp_size
    return cp_size

def get_dp_size():

    global dp_size
    return dp_size

def get_cp_stream():

    global cp_stream
    if cp_stream == None:
        cp_stream = torch.cuda.Stream()
    
    return cp_stream

def get_dp_group():

    global dp_group
    return dp_group

def get_cp_group():

    global cp_group
    return cp_group


def get_dp_rank():

    global dp_rank
    global cp_rank

    return dp_rank


def get_cp_rank():

    global dp_rank
    global cp_rank

    return cp_rank



def get_cp_rank_list():
    
    global cp_ranks
    if cp_ranks == None:
        cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
    return cp_ranks


def cp_broadcast(tensor, cp_index=0):

    global dp_group
    global cp_group

    cp_ranks = get_cp_rank_list()

    torch.distributed.broadcast(tensor, cp_ranks[cp_index], group=cp_group)




def cp_broadcast_objects(tensor):

    global dp_group
    global cp_group

    raise NotImplementedError("cp_broadcast_objects method is not yet implemented!!!")




def split_tensor_in_cp(input, seq_dim):

    global cp_size

    seq_size = input.shape[seq_dim]

    if seq_size%cp_size != 0:
        raise RuntimeError(f'seq_length {seq_size} in dim {seq_dim} must be multiple of cp_size {cp_size}!!!')

    split_seq_size = seq_size//cp_size

    tensor_splits = input.split(split_seq_size, dim=seq_dim)

    cp_rank = get_cp_rank()

    split_tensor = tensor_splits[cp_rank]

    return split_tensor





class GatherFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, process_group, seq_dim, frames):
        ctx.cp_group = process_group
        ctx.seq_dim = seq_dim
        ctx.frames = frames
        ctx.cp_size = get_cp_size()

        input = rearrange(input, "B (T S) C -> B T S C", T=frames)

        with torch.no_grad():

            input = input.contiguous()

            output_tensors = [torch.zeros_like(input) for _ in range(ctx.cp_size)]

            dist.all_gather(output_tensors, input, group=ctx.cp_group)

            output_tensor = torch.cat(output_tensors, dim=seq_dim)



        output_tensor = rearrange(output_tensor, "B T S C -> B (T S) C", T=frames)


        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        

        with torch.no_grad():
            
            grad_output = grad_output * ctx.cp_size

            grad_output = rearrange(grad_output, "B (T S) C -> B T S C", T=ctx.frames)

            grad_input = split_tensor_in_cp(grad_output, ctx.seq_dim)

            grad_input = rearrange(grad_input, "B T S C -> B (T S) C", T=ctx.frames)
            

        return grad_input, None, None, None




class SplitFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, process_group, seq_dim):
        ctx.cp_group = process_group
        ctx.seq_dim = seq_dim
        ctx.cp_size = get_cp_size()

        output_tensor = split_tensor_in_cp(input, ctx.seq_dim)

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        

        with torch.no_grad():


            grad_output = grad_output / ctx.cp_size

            output_tensors = [torch.zeros_like(grad_output) for _ in range(ctx.cp_size)]

            dist.all_gather(output_tensors, grad_output, group=ctx.cp_group)

            grad_input = torch.cat(output_tensors, dim=ctx.seq_dim)


        return grad_input, None, None



def gather_cp(input, frames):

    cp_process_group = get_cp_group()
    
    output_tensor = GatherFunction.apply(input, cp_process_group, 2, frames)

    return output_tensor


def split_cp(input, seq_dim):

    cp_process_group = get_cp_group()
    
    output_tensor = SplitFunction.apply(input, cp_process_group, seq_dim)

    return output_tensor




class ReduceFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, process_group):
        ctx.cp_group = process_group

        output = input.detach().clone()

        dist.all_reduce(output, group=ctx.cp_group)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_input = grad_output.detach().clone()

        return grad_input, None
    


class ReplicateFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, process_group):
        ctx.cp_group = process_group

        output = input.detach().clone()


        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_input = grad_output.detach().clone()

        dist.all_reduce(grad_input, group=ctx.cp_group)


        return grad_input, None


def reduce_cp(partial_sum, partial_square_sum):

    cp_process_group = get_cp_group()
    
    all_sum = ReduceFunction.apply(partial_sum, cp_process_group)
    all_square_sum = ReduceFunction.apply(partial_square_sum, cp_process_group)

    return all_sum, all_square_sum


def replicate_cp(all_mean, all_var):

    cp_process_group = get_cp_group()
    
    all_mean = ReplicateFunction.apply(all_mean, cp_process_group)
    all_var = ReplicateFunction.apply(all_var, cp_process_group)

    return all_mean, all_var



def _all_to_all_func(input_, world_size, group, scatter_dim, gather_dim):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)

        return _all_to_all_func(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


def all_to_all_with_pad(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_pad: int = 0,
    gather_pad: int = 0,
):
    if scatter_pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[scatter_dim] = scatter_pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        input_ = torch.cat([input_, pad_tensor], dim=scatter_dim)

    assert (
        input_.shape[scatter_dim] % dist.get_world_size(process_group) == 0
    ), f"Dimension to scatter ({input_.shape[scatter_dim]}) is not divisible by world size ({dist.get_world_size(process_group)})"
    input_ = _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

    if gather_pad > 0:
        input_ = input_.narrow(gather_dim, 0, input_.size(gather_dim) - gather_pad)

    return input_


def dynamic_switch(x, scatter_dim, gather_dim):

    scatter_pad = 0
    gather_pad = 0
    cp_process_group = get_cp_group()

    x = all_to_all_with_pad(
        x,
        cp_process_group,
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
        scatter_pad=scatter_pad,
        gather_pad=gather_pad,
    )
    return x
