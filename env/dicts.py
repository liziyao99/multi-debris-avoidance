import numpy as np

class stateDict:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int) -> None:
        self.state = np.zeros(state_dim, dtype=np.float32)
        self.obs = np.zeros(obs_dim, dtype=np.float32)
        self.action = np.zeros(action_dim, dtype=np.float32)
        self.reward = 0.
        self.done = False

    @classmethod
    def from_data(cls, state:np.ndarray, action:np.ndarray, reward:float, done:bool, obs:np.ndarray):
        sd = cls(state.shape[0], obs.shape[0], action.shape[0])
        sd.state = state
        sd.action = action
        sd.reward = reward
        sd.done = done
        sd.obs = obs
        return sd
    