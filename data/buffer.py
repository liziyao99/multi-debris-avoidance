import numpy as np
import torch
import collections
import random, math
from typing import List

class replayBuffer:
    def __init__(self,
                 keys:List[str],
                 capacity=10000,
                 minimal_size=1000,
                 batch_size=128) -> None:
        '''
            `keys`: list of str, should be keys of trans_dict used for training. Data in trans_dict would be added into buffer.
        '''
        self.buffer = collections.deque(maxlen=capacity)
        self.minimal_size = minimal_size
        self.batch_size = batch_size
        self._keys = keys

    def from_dict(self, trans_dict:dict, extend=True):
        datas = []
        if len(trans_dict.keys())==0:
            # empty dict passed when buffer is empty
            return []
        for key in self._keys:
            datas.append(trans_dict[key])
        zipped = zip(*datas)
        if extend: self.buffer.extend(zipped)
        return list(zipped)

    def from_dicts(self, dicts:List[dict]):
        for dict in dicts:
            self.from_dict(dict)

    def to_dict(self, transitions:List[tuple]):
        if len(transitions) == 0:
            return {}
        trans_dict = dict(zip(self._keys, zip(*transitions)))
        for key in self._keys:
            if type(trans_dict[key][0]) is torch.Tensor:
                trans_dict[key] = torch.vstack(trans_dict[key])
            elif type(trans_dict[key][0]) is np.ndarray:
                trans_dict[key] = np.vstack(trans_dict[key])
            else:
                trans_dict[key] = np.array(trans_dict[key])
        return trans_dict
    
    def to_dicts(self, is_batch=True, batch_size=None) -> List[dict]:
        dicts = []
        if is_batch:
            batch_size = self.batch_size if batch_size is None else batch_size
            batch_size = min(batch_size, self.size)
            n_batch = math.ceil(self.size//batch_size)
            for i in range(n_batch):
                batch = list(self.buffer[i*batch_size:(i+1)*batch_size])
                dicts.append(self.to_dict(batch))
        else:
            dicts.append(self.to_dict(list(self.buffer)))
        return dicts

    def sample(self, batch_size:int=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        batch_size = min(batch_size, self.size)
        transitions = random.sample(self.buffer, batch_size)
        trans_dict = self.to_dict(transitions)
        return trans_dict
    
    def key2idx(self, key:str):
        return self._keys.index(key)
    
    @property
    def size(self):
        return len(self.buffer)
    
    @property
    def capacity(self):
        return self.buffer.maxlen
    
    @property
    def n_key(self):
        return len(self._keys)
    
    def keys(self):
        return self._keys.copy()

    def clear(self):
        self.buffer.clear()
