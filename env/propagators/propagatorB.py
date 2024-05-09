'''
    defining classes wrapped `propagator`s and their `propagatorT` counterparts.
'''
import numpy as np
import torch

import env.propagators.propagator as propagator
import env.propagators.propagatorT as propagatorT

class dummyPropagatorB:
    def __init__(self,
                 propN:propagator.Propagator,
                 propT:propagatorT.PropagatorT,
                ) -> None:
        self.N = propN
        '''
            `Propagator`, process numpy.ndarray.
        '''
        self.T = propT
        '''
            `PropagatorT`, process torch.Tensor.
        '''

    def propagate(self, states:np.ndarray, actions:np.ndarray):
        '''
            self.N.propagate(states, actions) -> Tuple[ndarray]
        '''
        return self.N.propagate(states, actions)
    
    def getObss(self, states:np.ndarray):
        '''
            self.N.getObss(states) -> ndarray
        '''
        return self.N.getObss(states)
    
    def getNextStates(self, states:np.ndarray, actions:np.ndarray):
        '''
            self.N.getNextStates(states, actions) -> ndarray
        '''
        return self.N.getNextStates(states, actions)

    def getRewards(self, states:np.ndarray, actions:np.ndarray):
        '''
            self.N.getRewards(states, actions) -> ndarray
        '''
        return self.N.getRewards(states, actions)
    
    def getDones(self, states:np.ndarray):
        '''
            self.N.getDones(states) -> ndarray
        '''
        return self.N.getDones(states)
    
    def randomInitStates(self, num_states:int):
        '''
            self.N.randomInitStates(num_states) -> ndarray
        '''
        return self.N.randomInitStates(num_states)
    
    def seqOpt(self, states:torch.Tensor, agent, horizon:int, totalOptStep=True, **kwargs):
        '''
            self.T.seqOpt(states, agent, horizon, optStep) -> Tensor
        '''
        return self.T.seqOpt(states, agent, horizon, totalOptStep=totalOptStep, **kwargs)
    
    @property
    def state_dim(self):
        return self.N.state_dim
    
    @property
    def obs_dim(self):
        return self.N.obs_dim
    
    @property
    def action_dim(self):
        return self.N.action_dim
    

class motionSystemB(dummyPropagatorB):
    def __init__(self,
                 state_mat:np.ndarray,
                 device:str,
                 max_dist=10.
                ) -> None:
        state_mat = state_mat.astype(np.float32)
        self.N = propagator.motionSystem(state_mat, max_dist=max_dist)
        state_mat_T = torch.from_numpy(state_mat).to(device)
        self.T = propagatorT.motionSystemT(state_mat_T, max_dist=max_dist, device=device)

class CWPropagatorB(dummyPropagatorB):
    def __init__(self,
                 device:str,
                 dt=1., orbit_rad=7e6, max_dist=5e3
                ) -> None:
        self.N = propagator.CWPropagator(dt=dt, orbit_rad=orbit_rad, max_dist=max_dist)
        self.T = propagatorT.CWPropagatorT(device=device, dt=dt, orbit_rad=orbit_rad, max_dist=max_dist)

class CWDebrisPropagatorB(dummyPropagatorB):
    def __init__(self,
                 device:str, n_debris:int,
                 dt=1., orbit_rad=7e6, max_dist=5e3, safe_dist=5e2,
                ) -> None:
        self.N = propagator.CWDebrisPropagator(n_debris=n_debris, dt=dt, orbit_rad=orbit_rad, max_dist=max_dist, safe_dist=safe_dist)
        self.T = propagatorT.CWDebrisPropagatorT(device=device, n_debris=n_debris, dt=dt, orbit_rad=orbit_rad, max_dist=max_dist, safe_dist=safe_dist)