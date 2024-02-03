import numpy as np

class Propagator:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def getObs(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getReward(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
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
        next_obss = self.getObs(states)
        return next_states, rewards, dones, next_obss
    

class debugPropagator(Propagator):
    def __init__(self) -> None:
        state_dim = 6
        obs_dim = 6
        action_dim = 3
        super().__init__(state_dim, obs_dim, action_dim)

    def getObs(self, states:np.ndarray) -> np.ndarray:
        return states

    def getReward(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        return self.dist2origin(states)

    def getNextState(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        next_states = np.zeros_like(states)
        next_states[:, 3:] = states[:, 3:] + actions # vel
        next_states[:, :3] = states[:, :3] + next_states[:, 3:] # pos
        return next_states

    def getDone(self, states:np.ndarray) -> np.ndarray:
        dones = self.dist2origin(states) > 10
        return dones
    
    def dist2origin(self, states):
        return np.linalg.norm(states[:, :3], axis=1)