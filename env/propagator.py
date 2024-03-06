import numpy as np
from env.dynamic import matrix

class Propagator:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def getObss(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getTruncatedRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getDones(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def propagate(self, states:np.ndarray, actions:np.ndarray):
        '''
            returns: `next_states`, `rewards`, `dones`, `next_obss`
        '''
        next_states = self.getNextStates(states, actions)
        rewards = self.getRewards(states, actions)
        dones = self.getDones(states)
        next_obss = self.getObss(next_states)
        return next_states, rewards, dones, next_obss
    
    def randomInitStates(self, num_states:int) -> np.ndarray:
        raise NotImplementedError
    
    def obssNormalize(self, obss:np.ndarray) -> np.ndarray:
        raise obss
    

class linearSystem(Propagator):
    def __init__(self,
                 state_mat:np.ndarray, 
                 obs_mat:np.ndarray,
                 control_mat:np.ndarray) -> None:
        if state_mat.shape[0]!=state_mat.shape[1]:
            raise ValueError("state_mat must be a square matrix")
        if obs_mat.shape[1]!=obs_mat.shape[0]:
            raise ValueError("shape incompatible: `state_mat` and `obs_dim`")
        if control_mat.shape[0]!=state_mat.shape[0]:
            raise ValueError("shape incompatible: `state_mat` and `control_mat`")
        state_dim = state_mat.shape[0]
        obs_dim = obs_mat.shape[0]
        action_dim = control_mat.shape[1]
        super().__init__(state_dim, obs_dim, action_dim)
        self.state_mat = state_mat.astype(np.float32)
        self.obs_mat = obs_mat.astype(np.float32)
        self.control_mat = control_mat.astype(np.float32)

    def getObss(self, states:np.ndarray) -> np.ndarray:
        obs = states@self.obs_mat.T
        return obs

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise(NotImplementedError)

    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        next_states = states@self.state_mat.T + actions@self.control_mat.T
        return next_states

    def getDones(self, states:np.ndarray) -> np.ndarray:
        raise(NotImplementedError)
    

class debugPropagator(Propagator):
    def __init__(self, max_dist=10.) -> None:
        state_dim = 6
        obs_dim = 6
        action_dim = 3
        super().__init__(state_dim, obs_dim, action_dim)
        self.max_dist = max_dist
        self.k = 0.2

    def getObss(self, states:np.ndarray) -> np.ndarray:
        return states

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        return (self.max_dist-rad-self.k*vel)/self.max_dist

    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        next_states = np.zeros_like(states)
        next_states[:, 3:] = states[:, 3:] + actions # vel
        # next_states[:, :3] = states[:, :3] + next_states[:, 3:] # pos
        next_states[:, :3] = states[:, :3] + states[:, 3:] # pos
        return next_states

    def getDones(self, states:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        dones = rad>self.max_dist
        return dones
    
    def norms(self, states):
        return np.linalg.norm(states[:, :3], axis=1), np.linalg.norm(states[:, 3:], axis=1)
    
    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.zeros((num_states, self.state_dim), dtype=np.float32)
        states[:,:3] = np.random.uniform(low=-self.max_dist/2, high=self.max_dist/2, size=(num_states,3))
        states[:,3:] = np.random.uniform(low=-self.max_dist/100, high=self.max_dist/100, size=(num_states,3))
        return states

class debugLinearSystem(linearSystem):
    def __init__(self, state_mat:np.ndarray, obs_mat:np.ndarray, control_mat:np.ndarray, 
                 max_state_norm=10., k1=1., k2=.2) -> None:
        super().__init__(state_mat, obs_mat, control_mat)
        self.max_state_norm = max_state_norm
        self.k1 = k1
        self.k2 = k2

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        state_rewards = self.k1*(self.max_state_norm-np.linalg.norm(states[:,:3], axis=1))
        state_rewards2 = -self.k2*np.linalg.norm(states[:,3:], axis=1)
        action_rewards = -self.k2*np.linalg.norm(actions, axis=1)
        return (state_rewards+state_rewards2)/self.max_state_norm

    def getDones(self, states:np.ndarray) -> np.ndarray:
        return np.linalg.norm(states[:,:3], axis=1)>self.max_state_norm
    
    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.random.uniform(low=-self.max_state_norm/self.state_dim/10, 
                                   high=self.max_state_norm/self.state_dim/10, 
                                   size=(num_states,self.state_dim))
        return states
    
class motionSystem(linearSystem):
    def __init__(self, state_mat: np.ndarray, max_dist=10.) -> None:
        state_dim = state_mat.shape[0]
        if state_dim%2!=0:
            raise ValueError("state_mat must be an even integer, representing position and velocity.")
        space_dim = state_dim//2
        obs_mat = np.eye(state_dim)
        control_mat = np.vstack((np.zeros((space_dim,space_dim)),np.eye(space_dim)))
        super().__init__(state_mat, obs_mat, control_mat)
        self.space_dim = space_dim
        
        self.max_dist = max_dist
        self.k=0.2

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        return (self.max_dist-rad-self.k*vel)/self.max_dist

    def getDones(self, states:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        dones = rad>self.max_dist
        return dones
    
    def norms(self, states):
        return np.linalg.norm(states[:, :self.space_dim], axis=1), np.linalg.norm(states[:, self.space_dim:], axis=1)

    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.zeros((num_states, self.state_dim), dtype=np.float32)
        f1 = self.space_dim
        f2 = 10*self.space_dim
        states[:,:self.space_dim] = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states,self.space_dim))
        states[:,self.space_dim:] = np.random.uniform(low=-self.max_dist/f2, high=self.max_dist/f2, size=(num_states,self.space_dim))
        return states
    

class CWPropagator(motionSystem):
    def __init__(self, dt=1., orbit_rad=7e6, max_dist=5e3) -> None:
        state_mat = matrix.CW_TransMat(0, dt, orbit_rad)
        super().__init__(state_mat, max_dist=max_dist)
        self.dt = dt
        self.orbit_rad = orbit_rad

    def getNextStates(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        con_vec = matrix.CW_constConVecs(0, self.dt, actions, self.orbit_rad)
        next_states = states@self.state_mat.T + con_vec
        return next_states
    
    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.zeros((num_states, self.state_dim), dtype=np.float32)
        f1 = self.space_dim
        f2 = 100*self.space_dim
        states[:,:self.space_dim] = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states,self.space_dim))
        states[:,self.space_dim:] = np.random.uniform(low=-self.max_dist/f2, high=self.max_dist/f2, size=(num_states,self.space_dim))
        return states
    
    def obssNormalize(self, obss: np.ndarray) -> np.ndarray:
        f1 = self.max_dist
        f2 = self.max_dist/100
        obss_n = obss.copy()
        obss_n[:,:self.space_dim] /= f1
        obss_n[:,self.space_dim:] /= f2
        return obss_n
    
    def getObss(self, states: np.ndarray) -> np.ndarray:
        obss = super().getObss(states)
        return self.obssNormalize(obss)