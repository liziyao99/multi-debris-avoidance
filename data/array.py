import numpy as np
import anytree
import typing

class indexNode(anytree.Node):
    def __init__(self, name, gen=-1, idx=-1):
        super().__init__(name)
        self.gen = gen
        self.idx = idx

    def attach(self, parent):
        '''
            attach self to parent.
        '''
        self.parent = parent
    
    def detach(self):
        '''
            cutoff all children.
        '''
        self.children = []

    @property
    def indexPair(self):
        '''
            return self.gen and self.idx.
        '''
        return self.gen, self.idx
    
    @property
    def childrenIdx(self):
        '''
            return indecies of all children.
        '''
        return np.array([child.idx for child in self.children])
    
    @property
    def parentIdx(self):
        '''
            return indecies of parent.
        '''
        if self.parent:
            return self.parent.idx
        else:
            return None

class treeDataArray:
    def __init__(self,
                 population,
                 max_gen,
                 state_dim,
                 obs_dim,
                 action_dim,
                 flags:typing.Tuple[str],
                 items:typing.Tuple[str]) -> None:
        self.population = population
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((max_gen, population, state_dim), dtype=np.float32)
        self.obss = np.zeros((max_gen, population, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_gen, population, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_gen, population, 1), dtype=np.float32)
        self.next_states = np.zeros((max_gen, population, state_dim), dtype=np.float32)
        self.next_obss = np.zeros((max_gen, population, obs_dim), dtype=np.float32)
        self.dones = np.zeros((max_gen, population, 1), dtype=np.bool_)

        self.flags = {}
        for flag in flags:
            self.flags[flag] = np.zeros((max_gen, population, 1), dtype=np.bool_)

        self.items = {}
        for item in items:
            self.items[item] = np.zeros((max_gen, population, 1), dtype=np.float32)

    def deflag(self, keys:typing.Tuple[str]=None):
        if keys is None:
            keys = self.flags.keys()
        for key in keys:
            self.flags[key][...] = False

    def fill_gen(self, 
                 gen:int, 
                 states:np.ndarray, 
                 obss:np.ndarray, 
                 actions:np.ndarray, 
                 rewards:np.ndarray, 
                 next_states:np.ndarray,
                 next_obss:np.ndarray,
                 dones:np.ndarray, 
                 **kwargs):
        '''
            `**kwargs` are np arrays of shape (population, 1) to be filled into `self.items`.
        '''
        self.states[gen,:] = states.reshape((self.population,self.state_dim))[:]
        self.obss[gen,:] = obss.reshape((self.population, self.obs_dim))[:]
        self.actions[gen,:] = actions.reshape((self.population, self.action_dim))[:]
        self.rewards[gen,:] = rewards.reshape((self.population, 1))[:]
        self.next_states[gen,:] = next_states.reshape((self.population, self.state_dim))[:]
        self.next_obss[gen,:] = next_obss.reshape((self.population, self.obs_dim))[:]
        self.dones[gen,:] = dones.reshape((self.population, 1))[:]
        for key, value in kwargs.items():
            self.items[key][gen,:] = value.reshape((self.population, 1))[:]