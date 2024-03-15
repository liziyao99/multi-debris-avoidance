import torch
import numpy as np

def affine(x, s0, t0, s1, t1):
    return ((x-s0)*t1 + (t0-x)*s1)/(t0-s0)

def dotEachRow(a, b, keepdim=False):
    prod = a*b
    if type(a) is torch.Tensor:
        dot = torch.sum(prod, dim=1, keepdim=keepdim)
    else:
        dot = np.sum(prod, axis=1, keepdims=keepdim)
    return dot

def lineProj(x, p, v):
    '''
    assuming all args are of same type (numpy or torch) and shape (batch_size, n_dim).
        args:
            `x`: point to be projected on the line.
            `p`: line's start point.
            `v`: line's direction vector.
        return:
            `x_proj`: projected point.
            `x_orth`: `x - x_proj`.
    '''
    if len(x.shape)==1:
        x = x[None, :]
        p = p[None, :]
        v = v[None, :]
    if type(x) is torch.Tensor:
        v_norm = torch.norm(v,dim=1,keepdim=True)
    else:
        v_norm = np.linalg.norm(v,axis=1,keepdims=True)
    v_unit = v/v_norm
    v_proj = dotEachRow(x-p, v_unit, keepdim=True)
    x_proj = p + v_proj * v_unit
    x_orth = x - x_proj
    return x_proj, x_orth

def compute_advantage(gamma, lmd, td_delta:torch.Tensor):
    '''
        `td_delta`: Torch.Tensor of shape (step_num, copy_num).
    '''
    td_delta = torch.flip(td_delta, dims=[0]).detach()
    advantage_list = []
    advantage = 0.
    for delta in td_delta:
        advantage = gamma * lmd * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.vstack(advantage_list)
    return advantage_list