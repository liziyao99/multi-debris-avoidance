import numpy as np
import numba as nb
from numpy import sin, cos
from numpy.random import multivariate_normal as normal_dist
import torch

MU_EARTH = 3.986004418E14 # m3s-2


def CW_StateMat(a:float, 
                mu = MU_EARTH
                ) -> np.ndarray:
    '''
        args:
            `a`: radius of the target's circular orbit, m
            `mu`: standard gravitational parameter, m3s-2
        return:
            `A`: state matrix of CW equation, np array of shape [6,6]
    '''
    n = np.sqrt(mu/a**3)
    A_rr = np.zeros([3,3])
    A_vr = np.eye(3)
    A_rv = np.array([
            [3*n**2, 0, 0],
            [0, 0, 0],
            [0, 0, -n**2]
        ])
    A_vv = np.array([
            [0, 2*n, 0],
            [-2*n, 0, 0],
            [0, 0, 0]
        ])
    A = np.vstack([
            np.hstack([A_rr, A_vr]),
            np.hstack([A_rv, A_vv])
        ])
    return A

# @nb.njit
def CW_TransMat(t0:float,
                t:float, 
                a:float, 
                mu = MU_EARTH
                ):
    '''
        args:
            `t0`, `t`: time, s
            `a`: radius of the target's circular orbit, m
            `mu`: standard gravitational parameter, m3s-2
        return:
            `Phi`: transition matrix from time=0 to time=t, np array of shape [6,6]
    '''
    dt = t-t0
    n = np.sqrt(mu/a**3)
    tt = n*dt
    c = cos(tt)
    s = sin(tt)
    Phi_rr = np.array([
            [4-3*c, 0, 0],
            [6*(s-tt), 1, 0],
            [0, 0, c]
        ])
    Phi_vr = np.array([
            [s/n, 2*(1-c)/n, 0],
            [2*(c-1)/n, (4*s-3*tt)/n, 0],
            [0, 0, s/n]
        ])
    Phi_rv = np.array([
            [3*n*s, 0, 0],
            [6*n*(c-1), 0, 0],
            [0, 0, -n*s]
        ])
    Phi_vv = np.array([
            [c, 2*s, 0],
            [-2*s, 4*c-3, 0],
            [0, 0, c]
        ])
    Phi = np.vstack((
            np.hstack((Phi_rr, Phi_vr)), 
            np.hstack((Phi_rv, Phi_vv))
        )).astype(np.float32)
    return Phi

def CW_transMat_batch(t0:np.ndarray,
                      t1:np.ndarray,
                      a:np.ndarray,
                      mu:np.ndarray=None):
    '''
        all args are of shape (batch_size,). Return array of shape (batch_size, 6, 6), 
        corresponding to each entry.
    '''
    batch_size = t0.shape[0]
    if mu is None:
        mu = np.array([MU_EARTH]*batch_size, dtype=np.float64)
    dt = (t1-t0).astype(np.float64)
    a = a.astype(np.float64)
    # NOTE: below have small divide small, use float64 in case loss of accuracy.
    n = np.sqrt(mu/a**3)
    tt = n*dt
    c = cos(tt)
    s = sin(tt)
    zero = np.zeros_like(tt)
    one = np.ones_like(tt)
    Phi_rr = np.array([
            [4-3*c, zero, zero],
            [6*(s-tt), one, zero],
            [zero, zero, c]
        ]) # shape (3,3,batch_size)
    Phi_vr = np.array([
            [s/n, 2*(1-c)/n, zero],
            [2*(c-1)/n, (4*s-3*tt)/n, zero],
            [zero, zero, s/n]
        ])
    Phi_rv = np.array([
            [3*n*s, zero, zero],
            [6*n*(c-1), zero, zero],
            [zero, zero, -n*s]
        ])
    Phi_vv = np.array([
            [c, 2*s, zero],
            [-2*s, 4*c-3, zero],
            [zero, zero, c]
        ])
    Phi = np.vstack((
            np.hstack((Phi_rr, Phi_vr)),
            np.hstack((Phi_rv, Phi_vv))
        )).astype(np.float32) # shape (6, 6, batch_size)
    Phi = Phi.swapaxes(0,1).swapaxes(0,2) # shape (batch_size, 6, 6)
    return Phi

def CW_constConVec(t0:float,
                t :float,
                u, # arrayLike
                a :float,
                mu = MU_EARTH
                ) -> np.ndarray:
    dt = t-t0
    n = np.sqrt(mu/a**3)
    tt = n*dt
    # u = u.detach().cpu().numpy()
    trans_u = np.array([
        (u[0]*(1-cos(tt)) + 2*u[1]*(tt-sin(tt)))/n**2,
        (8*u[1]*(1-cos(tt)) - 3*u[1]*tt**2 + 4*u[0]*(sin(tt)-tt))/(2*n**2),
        (u[2]*(1-cos(tt)))/n**2,
        (2*u[1]*(1-cos(tt)) + u[0]*sin(tt))/n,
        (2*u[0]*(cos(tt)-1) - 3*u[1]*tt + 4*u[1]*sin(tt))/n,
        (u[2]*sin(tt))/n
    ])
    return trans_u

# @nb.njit
def CW_constConVecs(t0:float,
                    t :float,
                    u :np.ndarray,
                    a :float,
                    mu = MU_EARTH
                    ):
    '''
        `u` is of shape (population, 3).\n
        return $\\int_{0}^{t} \Phi(t-\\tau) B u(\\tau) d\\tau$, shape (population, 6).
    '''
    dt = t-t0
    n = np.sqrt(mu/a**3)
    tt = n*dt
    c = cos(tt)
    s = sin(tt)
    trans_u = np.vstack((
        (u[:,0]*(1-c) + 2*u[:,1]*(tt-s))/n**2,
        (8*u[:,1]*(1-c) - 3*u[:,1]*tt**2 + 4*u[:,0]*(s-tt))/(2*n**2),
        (u[:,2]*(1-c))/n**2,
        (2*u[:,1]*(1-c) + u[:,0]*s)/n,
        (2*u[:,0]*(c-1) - 3*u[:,1]*tt + 4*u[:,1]*s)/n,
        (u[:,2]*s)/n
    )).T.astype(np.float32)
    return trans_u

def CW_constConVecsT(t0:float,
                    t :float,
                    u :torch.Tensor,
                    a :float,
                    mu = MU_EARTH
                    ):
    '''
        torch version of `CW_constConVecs`.\n
        `u` is of shape (population, 3).\n
        return $\\int_{0}^{t} \Phi(t-\\tau) B u(\\tau) d\\tau$, shape (population, 6).
    '''
    dt = t-t0
    n = np.sqrt(mu/a**3)
    tt = n*dt
    c = cos(tt)
    s = sin(tt)
    trans_u = torch.vstack((
        (u[:,0]*(1-c) + 2*u[:,1]*(tt-s))/n**2,
        (8*u[:,1]*(1-c) - 3*u[:,1]*tt**2 + 4*u[:,0]*(s-tt))/(2*n**2),
        (u[:,2]*(1-c))/n**2,
        (2*u[:,1]*(1-c) + u[:,0]*s)/n,
        (2*u[:,0]*(c-1) - 3*u[:,1]*tt + 4*u[:,1]*s)/n,
        (u[:,2]*s)/n
    )).T
    return trans_u


def rotMat3D(axes, 
             theta):
    if type(axes) is torch.Tensor:
        axes = axes.detach().cpu().numpy()
    axes = axes.flatten()
    axes /= np.linalg.norm(axes)
    c = cos(theta)
    s = sin(theta)
    R = np.array([ [ c+axes[0]**2*(1-c), axes[0]*axes[1]*(1-c)-axes[2]*s, axes[0]*axes[2]*(1-c)+axes[1]*s ],
        [ axes[1]*axes[0]*(1-c)+axes[2]*s, c+axes[1]**2*(1-c), axes[1]*axes[2]*(1-c)-axes[0]*s ],
        [ axes[2]*axes[0]*(1-c)-axes[1]*s, axes[2]*axes[1]*(1-c)+axes[0]*s, c+axes[2]**2*(1-c) ]
        ], dtype=np.float32)
    return R

from scipy.integrate import quad_vec
def CW_ConGramMat(dt, orbit_rad) -> np.ndarray:
    def integrand(tau):
        BBT = np.vstack([
            np.hstack([np.zeros([3,3]), np.zeros([3,3])]),
            np.hstack([np.zeros([3,3]), np.eye(3)])
        ]) # B B^\top, where B is control matrix
        Phi = CW_TransMat(tau, dt, orbit_rad)
        mat = Phi @ BBT @ Phi.T
        return mat
    gram, _ = quad_vec(integrand, 0, dt)
    return gram

def CW_ConGramMat_asyn(dts, orbit_rad) -> np.ndarray:
    n = len(dts)
    def integrand(tau:float):
        BBT = np.vstack([
            np.hstack([np.zeros([3,3]), np.zeros([3,3])]),
            np.hstack([np.zeros([3,3]), np.eye(3)])
        ]) # B B^\top, where B is control matrix
        mats = np.zeros((n,6,6), dtype=np.float32)
        for i in range(n):
            if tau<=dts[i]:
                dt = dts[i]
                Phi = CW_TransMat(tau, dt, orbit_rad)
                mat = Phi @ BBT @ Phi.T
                mats[i,:] = mat
        return mats
    dt = max(dts)
    gram, _ = quad_vec(integrand, 0, dt)
    return gram
    