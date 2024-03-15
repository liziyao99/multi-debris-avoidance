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
            if np.linalg.norm(states[:3]-states_c[:,:3],)>d2c:
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
    Phi = matrix.CW_transMat_batch(t2c, np.zeros_like(t2c), a)
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
    Phi = matrix.CW_transMat_batch(t2c, np.zeros_like(t2c), a)
    states = Phi@np.expand_dims(forecast_states, axis=2)
    states = np.squeeze(states, axis=2)
    states = torch.from_numpy(states).to(device)
    return states