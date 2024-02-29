import numpy as np
import typing, math

class stateDict:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int) -> None:
        self.state = np.zeros(state_dim, dtype=np.float32)
        self.obs = np.zeros(obs_dim, dtype=np.float32)
        self.action = np.zeros(action_dim, dtype=np.float32) # action done in last state
        self.reward = 0. # reward for transition from last state to current state
        self.done = False # current state done

    @classmethod
    def from_data(cls, state:np.ndarray, action:np.ndarray, reward:float, done:bool, obs:np.ndarray):
        sd = cls(state.shape[0], obs.shape[0], action.shape[0])
        sd.state = state
        sd.action = action
        sd.reward = reward
        sd.done = done
        sd.obs = obs
        return sd
    
    def __str__(self) -> str:
        s = f"state:\t{self.state}\n"
        s += f"obs:\t{self.obs}\n"
        s += f"action:\t{self.action}\n"
        s += f"reward:\t{self.reward}\n"
        s += f"done:\t{self.done}\n"
        return s
    
def init_transDict(length:int, state_dim:int, obs_dim:int, action_dim:int, 
                   items:typing.Tuple[str]=("td_targets", "regrets", "advantages")):
    '''
        dict keys: "states", "obss", "actions", "next_states", "next_obss", "rewards", "dones" and items.
        Items are float32 of shape (length,).
    '''
    trans_dict = {
        "states": np.zeros((length,state_dim), dtype=np.float32),
        "obss": np.zeros((length,obs_dim), dtype=np.float32),
        "actions": np.zeros((length,action_dim), dtype=np.float32),
        "next_states": np.zeros((length,state_dim), dtype=np.float32),
        "next_obss": np.zeros((length,obs_dim), dtype=np.float32),
        "rewards": np.zeros(length, dtype=np.float32),
        "dones": np.zeros(length, dtype=np.bool_),
    }
    for item in items:
        trans_dict[item] = np.zeros(length, dtype=np.float32)
    return trans_dict

def concat_dicts(dicts:typing.List[dict]):
    d = {}
    for key in dicts[0].keys():
        d[key] = []
    for i in range(1,len(dicts)):
        if d.keys()!=dicts[i].keys():
            raise(ValueError("dicts must have same keys."))
        for key in d.keys():
            d[key].append(dicts[i][key])
    for key in d.keys():
        d[key] = np.concatenate(d[key], axis=0)
    return d

def batch_dict(trans_dict:dict, batch_size:int):
    dicts = []
    total = trans_dict["states"].shape[0]
    n_batches = math.ceil(total/batch_size)
    for i in range(n_batches):
        d = {}
        for key in trans_dict.keys():
            d[key] = trans_dict[key][i*batch_size:(i+1)*batch_size]
        dicts.append(d)
    return dicts