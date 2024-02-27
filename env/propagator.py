import numpy as np

class Propagator:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def getObss(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getReward(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getTruncatedReward(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getNextState(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getDone(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def propagate(self, states:np.ndarray, actions:np.ndarray):
        '''
            returns: `next_states`, `rewards`, `dones`, `next_obss`
        '''
        next_states = self.getNextState(states, actions)
        rewards = self.getReward(states, actions)
        dones = self.getDone(states)
        next_obss = self.getObss(next_states)
        return next_states, rewards, dones, next_obss
    

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

    def getReward(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        return (self.max_dist-rad-self.k*vel)/self.max_dist

    def getNextState(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        next_states = np.zeros_like(states)
        next_states[:, 3:] = states[:, 3:] + actions # vel
        next_states[:, :3] = states[:, :3] + next_states[:, 3:] # pos
        return next_states

    def getDone(self, states:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        dones = rad>self.max_dist
        return dones
    
    def norms(self, states):
        return np.linalg.norm(states[:, :3], axis=1), np.linalg.norm(states[:, 3:], axis=1)