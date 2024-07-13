from env.dynamic import matrix

import numpy as np
import numba as nb
import torch

def CW_inversion(a,
                 forecast_states,
                 t2c=None,
                 d2c=None,
                ):
    '''
            This function returns relative state of two free space objects that will collision according to CW equation, 
        in time of `t2c`, with relative speed of `c_vel`.

            args:
                `a`: radius of the target's circular orbit, m
                `forecast_states`: forecasted collision states of debris to origin. shape (batch_size, 6).
                `t2c`: time to collision
                `d2c`: distance to collision
            returns:
                `state`: relative state of the two space objects.
                `t2c`:
                `d2c`:
    '''
    states_c = forecast_states
    if t2c is not None:
        Phi = matrix.CW_TransMat(t2c, 0, a)
        states = states_c@Phi.T
        d2c = np.linalg.norm(states[:3]-states_c[:,:3], axis=1)
    elif d2c is not None:
        T = 10.
        MAX_LOOP = 10000
        Phi = matrix.CW_TransMat(T, 0, a)
        states = states_c@Phi.T
        for k in range(MAX_LOOP):
            if np.linalg.norm(states[:,:3]-states_c[:,:3],)>d2c:
                break
            states = states@Phi.T
        t2c = T*(k+1)
    else:
        raise(ValueError)
    return states, t2c, d2c

@nb.njit
def CW_inversion_t2c(a, c_pos,c_vel, t2c,):
    '''
        njit.
            args:
                `a`: radius of the target's circular orbit, m
                `c_vel`: relative velocity when collision
                `t2c`: time to collision
                `c_pos`: collision position
            returns:
                `state`: relative state of the two space objects.
                `t2c`:
                `d2c`:
    '''
    state_c = np.hstack((c_pos, c_vel))
    Phi = matrix.CW_TransMat(t2c, 0, a)
    state = Phi @ state_c
    d2c = np.linalg.norm(state[:3]-c_pos)
    return state, t2c, d2c

@nb.njit
def CW_inversion_d2c(a, c_pos, c_vel, d2c:float,):
    '''
        njit.
            args:
                `a`: radius of the target's circular orbit, m
                `c_vel`: relative velocity when collision
                `d2c`: distance to collision
                `c_pos`: collision position
            returns:
                `state`: relative state of the two space objects.
                `t2c`:
                `d2c`:
    '''
    state_c = np.hstack((c_pos, c_vel)).astype(np.float32)
    T = 10.
    MAX_LOOP = 10000
    Phi = matrix.CW_TransMat(T, 0, a)
    state = Phi @ state_c
    for k in range(MAX_LOOP):
        if np.linalg.norm(state[:3]-c_pos)>d2c:
            break
        state = Phi @ state
    t2c = T*(k+1)
    return state, t2c, d2c


def CW_tInv_batch(a:np.ndarray, forecast_states:np.ndarray, t2c:np.ndarray,):
    Phi = matrix.CW_TransMat_batch(t2c, np.zeros_like(t2c), a)
    states = Phi@np.expand_dims(forecast_states, axis=2)
    states = np.squeeze(states, axis=2)
    return states

def CW_tInv_batch_torch(a:torch.Tensor, forecast_states:torch.Tensor, t2c:torch.Tensor, device:str):
    '''
        torch wrapper of `CW_tInv_batch`
    '''
    a = a.detach().cpu().numpy()
    forecast_states = forecast_states.detach().cpu().numpy()
    t2c = t2c.detach().cpu().numpy()
    Phi = matrix.CW_TransMat_batch(t2c, np.zeros_like(t2c), a)
    states = Phi@np.expand_dims(forecast_states, axis=2)
    states = np.squeeze(states, axis=2)
    states = torch.from_numpy(states).to(device)
    return states

def CW_rInv_batch_torch(a,
                        forecast_states:torch.Tensor,
                        d2c:float,
                        dt = 1.,
                        max_loop = 10000
                        ):
    states_0 = torch.zeros_like(forecast_states)
    t2c = torch.zeros(forecast_states.shape[0], device=forecast_states.device)
    flags = torch.zeros(forecast_states.shape[0], dtype=torch.bool, device=forecast_states.device,)
    states = forecast_states
    Phi = matrix.CW_TransMat(dt, 0, a)
    Phi = torch.from_numpy(Phi).to(forecast_states.device)
    for k in range(max_loop):
        r = torch.norm(states[:,:3], dim=-1)
        out = r>d2c
        _new = out & ~flags
        states_0[_new] = states[_new]
        t2c[_new] = k*dt
        flags = (r>d2c) | flags
        if flags.all():
            break
        states = states@Phi.T
    return states_0, t2c


def get_celestial_table(states0:torch.Tensor, a:float, dt:float, step:int,):
    '''
        args:
            `states0`: initial states of the celestial bodies, shape (batch_size, n_debris, 6)
            `a`: radius of the target's circular orbit, m
            `dt`: time step, s
            `step`: number of time steps
        return:
            `table`: states of the celestial bodies at each time step, shape (step, batch_size, n_debris, 6)
    '''
    states0 = states0.detach()
    batch_size = states0.shape[0]
    n_debris = states0.shape[1]
    Phi = matrix.CW_TransMat(0, dt, a)
    Phi = torch.from_numpy(Phi).to(states0.device)
    table = torch.zeros((step, batch_size, n_debris, 6), device=states0.device)
    states = states0.clone()
    for i in range(step):
        table[i,...] = states[...]
        states = states@Phi.T
    return table

def get_closet_approach(states0:torch.Tensor, a:float, dt:float, step:int,):
    '''
        args:
            `states0`: initial states of the celestial bodies, shape (batch_size, 1+n_debris, 6)
            `a`: radius of the target's circular orbit, m
            `dt`: time step, s
            `step`: number of time steps
        returns:
            `closet_state`: closest approach state of the debris, shape (batch_size, n_debris, 6)
            `closed_approach`: closest approach distance, shape (batch_size, n_debris)
            `closet_step`: closest approach step, shape (batch_size, n_debris)
            `table`: states of the celestial bodies at each time step, shape (step, batch_size, 1+n_debris, 6)
    '''
    batch_size = states0.shape[0]
    n_debris = states0.shape[1]-1
    table = get_celestial_table(states0, a, dt, step)
    primal = table[:, :, 0:1, :]
    debris = table[:, :, 1: , :]
    primal_pos = primal[:, :, :, :3]
    debris_pos = debris[:, :, :, :3]
    pos_diff = debris_pos-primal_pos
    pos_diff_norm = torch.norm(pos_diff, dim=-1) # shape: (step, batch_size, n_debris)
    closet_approach, closet_step = torch.min(pos_diff_norm, dim=0)
    closet_state = torch.zeros((batch_size, n_debris, 6), device=states0.device)
    for i in range(batch_size):
        for j in range(n_debris):
            closet_state[i,j] = table[closet_step[i,j], i, j+1]
    return closet_state, closet_approach, closet_step, table

def get_closet_approach_static(primal_states:torch.Tensor, debris_states0:torch.Tensor, a, dt, step):
    '''
        args:
            `primal_states`: states of the primal body, shape (batch_size, 1, 6)
            `debris_states0`: initial states of the debris, shape (batch_size, n_debris, 6)
            `a`: radius of the target's circular orbit, m
            `dt`: time step, s
            `step`: number of time steps
        returns:
            `closet_state`: closest approach state of the debris, shape (batch_size, n_debris, 6)
            `closed_approach`: closest approach distance, shape (batch_size, n_debris)
            `closet_step`: closest approach step, shape (batch_size, n_debris)
            `table`: states of the debris at each time step, shape (step, batch_size, n_debris, 6)
    '''
    batch_size = primal_states.shape[0]
    n_debris = debris_states0.shape[1]
    if primal_states.dim()==2:
        primal_states = primal_states.unsqueeze(1)
    primal_states = primal_states.unsqueeze(0)
    table = get_celestial_table(debris_states0, a, dt, step)
    primal_pos = primal_states[:, :, :, :3] # shape: (1, batch_size, 1, 3)
    debris_pos = table[:, :, :, :3] # shape: (step, batch_size, n_debris, 3)
    pos_diff = debris_pos-primal_pos
    pos_diff_norm = torch.norm(pos_diff, dim=-1) # shape: (step, batch_size, n_debris)
    closet_approach, closet_step = torch.min(pos_diff_norm, dim=0)
    closet_state = torch.zeros((batch_size, n_debris, 6), device=primal_states.device)
    for i in range(batch_size):
        for j in range(n_debris):
            closet_state[i,j] = table[closet_step[i,j], i, j]
    return closet_state, closet_approach, closet_step, table