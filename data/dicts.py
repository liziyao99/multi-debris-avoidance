import numpy as np
import torch
import typing, math

BASIC_KEYS = ("states", "obss", "actions", "rewards", "next_states", "next_obss", "dones")

class stateDict:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int, 
                 flags:typing.Tuple[str]=tuple(), items:typing.Tuple[str]=tuple()) -> None:
        self.state = np.zeros(state_dim, dtype=np.float32)
        self.obs = np.zeros(obs_dim, dtype=np.float32)
        self.action = np.zeros(action_dim, dtype=np.float32) # action done in last state
        self.reward = 0. # reward for transition from last state to current state
        self.done = False # current state done
        self.flags = {}
        for flag in flags:
            self.flags[flag] = False
        self.items = {}
        for item in items:
            self.items[item] = 0.

    @classmethod
    def from_data(cls, state:np.ndarray, action:np.ndarray, reward:float, done:bool, obs:np.ndarray, 
                  flags:typing.Tuple[str]=tuple(),
                  item_datas:dict={}):
        sd = cls(state.shape[0], obs.shape[0], action.shape[0], flags=flags, items=tuple(item_datas.keys()))
        sd.state = state
        sd.action = action
        sd.reward = reward
        sd.done = done
        sd.obs = obs
        for key in item_datas.keys():
            sd.items[key] = item_datas[key]
        return sd
    
    def __str__(self) -> str:
        s = f"state:\t{self.state}\n"
        s += f"obs:\t{self.obs}\n"
        s += f"action:\t{self.action}\n"
        s += f"reward:\t{self.reward}\n"
        s += f"done:\t{self.done}\n"
        return s
    
    def load(self, another):
        self.state[...] = another.state[...]
        self.obs[...] = another.obs[...]
        self.action[...] = another.action[...]
        self.reward = another.reward
        self.done = another.done
        for key in self.flags.keys():
            if key not in another.flags.keys():
                continue
            self.flags[key] = another.flags[key]
        for key in self.items.keys():
            if key not in another.items.keys():
                continue
            self.items[key] = another.items[key]
        return self
    
    def deflag(self, flags:typing.Tuple[str]=None):
        if keys is None:
            keys = self.flags.keys()
        for flag in flags:
            self.flags[flag] = False
    
def init_transDict(length:int, state_dim:int, obs_dim:int, action_dim:int, 
                   items:typing.Tuple[str]=("td_targets", "regrets", "advantages"),
                   other_terms:typing.Dict[str,typing.Tuple[int]]={}):
    '''
        dict keys: "states", "obss", "actions", "next_states", "next_obss", "rewards", "dones" and items and other_terms.
        Items are float32 of shape (length,).
        Other_terms are of shape (length,*other_terms[key]).
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
    for key in other_terms.keys():
        trans_dict[key] = np.zeros((length, *other_terms[key]), dtype=np.float32)
    return trans_dict

def init_transDictBatch(length:int, batch_size:int, state_dim:int, obs_dim:int, action_dim:int,
                   items:typing.Tuple[str]=(),
                   other_terms:typing.Dict[str,typing.Tuple[int]]={}, struct="numpy", device="cpu"):
    '''
        dict keys: "states", "obss", "actions", "next_states", "next_obss", "rewards", "dones" and items and other_terms.
        Items are float32 of shape (length,batch_size).
        Other_terms are of shape (length,batch_size,*other_terms[key]).
    '''
    if struct not in ["numpy", "torch"]:
        raise(ValueError("struct must be \"numpy\" or \"torch\"." ))
    trans_dict = {
        "states": np.zeros((length,batch_size,state_dim), dtype=np.float32),
        "obss": np.zeros((length,batch_size,obs_dim), dtype=np.float32),
        "actions": np.zeros((length,batch_size,action_dim), dtype=np.float32),
        "next_states": np.zeros((length,batch_size,state_dim), dtype=np.float32),
        "next_obss": np.zeros((length,batch_size,obs_dim), dtype=np.float32),
        "rewards": np.zeros((length,batch_size), dtype=np.float32),
        "dones": np.zeros((length,batch_size), dtype=np.bool_),
    }
    for item in items:
        trans_dict[item] = np.zeros((length,batch_size), dtype=np.float32)
    for key in other_terms.keys():
        trans_dict[key] = np.zeros((length,batch_size,*other_terms[key]), dtype=np.float32)

    if struct=="torch":
        for key in trans_dict.keys():
            trans_dict[key] = torch.from_numpy(trans_dict[key]).to(device=device)
    return trans_dict

def concat_dicts(dicts:typing.List[dict]):
    d = {}
    for key in dicts[0].keys():
        d[key] = []
    for i in range(0,len(dicts)):
        if d.keys()!=dicts[i].keys():
            raise(ValueError("dicts must have same keys."))
        for key in d.keys():
            d[key].append(dicts[i][key])
    for key in d.keys():
        d[key] = np.concatenate(d[key], axis=0)
    return d

def split_dict(trans_dict:dict, batch_size:int, shuffle=False):
    dicts = []
    key = list(trans_dict.keys())[0]
    total = len(trans_dict[key])
    if shuffle:
        pass # TODO: shuffle for list
    n_batches = math.ceil(total/batch_size)
    for i in range(n_batches):
        d = {}
        for key in trans_dict.keys():
            d[key] = trans_dict[key][i*batch_size:(i+1)*batch_size]
        dicts.append(d)
    return dicts

def deBatch_dict(trans_dict:dict) -> typing.List[dict]:
    '''
        arg:
            `trans_dict`: generated by `init_transDictBatch`
        return:
            `dicts`: each entry in `trans_dict`
    '''
    dicts = []
    for i in range(trans_dict["states"].shape[1]):
        d = {}
        for key in trans_dict.keys():
            d[key] = trans_dict[key][:,i]
        dicts.append(d)
    return dicts

def numpy_dict(trans_dict:dict) -> dict:
    for key in trans_dict.keys():
        if type(trans_dict[key]) is torch.Tensor:
            trans_dict[key] = trans_dict[key].detach().cpu().numpy()
    return trans_dict

def torch_dict(trans_dict:dict, device=None, detach=False) -> dict:
    for key in trans_dict.keys():
        if type(trans_dict[key]) is np.ndarray:
            trans_dict[key] = torch.from_numpy(trans_dict[key])
        trans_dict[key] = trans_dict[key].to(device=device)
        if detach:
            trans_dict[key] = trans_dict[key].detach()
    return trans_dict