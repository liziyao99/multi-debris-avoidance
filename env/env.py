import numpy as np
import typing

from env.propagator import Propagator, debugPropagator
from env.propagatorT import PropagatorT
from env.propagatorB import dummyPropagatorB
from tree.tree import stateDict
from agent.agent import rlAgent
from data.tree import GST
import torch

class dummyEnv:
    def __init__(self,
                 state_dim:int,
                 obs_dim:int,
                 action_dim:int,
                ) -> None:
        self.propagator = Propagator(state_dim, obs_dim, action_dim)

    @classmethod
    def from_propagator(cls, propagator:Propagator):
        de = cls(propagator.state_dim, propagator.obs_dim, propagator.action_dim)
        de.propagator = propagator
        return de

    def reset(self, *args, **kwargs):
        raise(NotImplementedError)
    
    def step(self, agent:rlAgent):
        raise(NotImplementedError)

    @property
    def state_dim(self):
        return self.propagator.state_dim
    
    @property
    def obs_dim(self):
        return self.propagator.obs_dim
    
    @property
    def action_dim(self):
        return self.propagator.action_dim

class singleEnv(dummyEnv):
    def __init__(self,
                 state_dim:int,
                 obs_dim:int,
                 action_dim:int,
                 max_episode:int,
                ) -> None:
        self.propagator = Propagator(state_dim, obs_dim, action_dim)
        self.max_episode = max_episode

        self.state = np.zeros(state_dim, dtype=np.float32)
        self.episode = 0

    @classmethod
    def from_propagator(cls, propagator:Propagator, max_episode:int):
        se = cls(propagator.state_dim, propagator.obs_dim, propagator.action_dim, max_episode)
        se.propagator = propagator
        return se

    def reset(self, state):
        self.state[:] = state[:]
        self.episode = 0

    def step(self, a:np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, bool]:
        '''
            args: 
                `a`:action.
            returns: 
                state, obs, action, reward, next_state, next_obs, done.
        '''
        s = np.expand_dims(self.state, axis=0)
        if len(a.shape)==1:
            a = np.expand_dims(a, axis=0)
        o = self.propagator.getObss(s)
        ns, r, d, no = self.propagator.propagate(s, a)

        s = s.flatten()
        o = o.flatten()
        a = a.flatten()
        r = r[0]
        ns = ns.flatten()
        no = no.flatten()
        d = d[0] or self.episode>=self.max_episode
        transit = (s, o, a, r, ns, no, d)

        self.state[:] = ns[:]
        self.episode += 1

        return transit
    
    def fill_dict(self, trans_dict:dict, transit:tuple):
        i = self.episode-1
        trans_dict["states"][i,:] = transit[0]
        trans_dict["obss"][i,:] = transit[1]
        trans_dict["actions"][i,:] = transit[2]
        trans_dict["rewards"][i] = transit[3]
        trans_dict["next_states"][i,:] = transit[4]
        trans_dict["next_obss"][i,:] = transit[5]
        trans_dict["dones"][i] = transit[6]
        return trans_dict
    
    def cut_dict(self, trans_dict:dict):
        for key in trans_dict.keys():
            trans_dict[key] = trans_dict[key][:self.episode]

class treeEnv(dummyEnv):
    def __init__(self,
                 state_dim:int,
                 obs_dim:int,
                 action_dim:int,
                 population:int,
                 max_gen:int
                ) -> None:
        self.propagator = Propagator(state_dim, obs_dim, action_dim)
        self.tree = GST(population, max_gen, state_dim, obs_dim, action_dim)

    @classmethod
    def from_propagator(cls, propagator: Propagator, population:int, max_gen:int):
        te = cls(propagator.state_dim, propagator.obs_dim, propagator.action_dim, population, max_gen)
        te.propagator = propagator
        return te
    
    def reset(self, root_stateDict:stateDict=None):
        if root_stateDict is None:
            root_stateDict = stateDict(self.propagator.state_dim, self.propagator.obs_dim, self.propagator.action_dim)
        self.tree.reset(root_stateDict)

    def step(self, agent:rlAgent):
        '''
            return: `done`
        '''
        return self.tree.step(agent, self.propagator)
    
    def simulate(self, agent:rlAgent):
        while not self.step(agent):
            self._stepProcess(agent)
        return self.tree.get_transDicts()
    
    def _stepProcess(self, agent:rlAgent, *args, **kwargs):
        pass
    
class treeEnvB(treeEnv):
    def __init__(self, state_dim: int, obs_dim: int, action_dim: int, population: int, max_gen: int, device:str) -> None:
        propN = Propagator(state_dim, obs_dim, action_dim)
        propT = PropagatorT(state_dim, obs_dim, action_dim, device=device)
        self.propagator = dummyPropagatorB(propN, propT)
        self.tree = GST(population, max_gen, state_dim, obs_dim, action_dim)
        self.seqOpt = True

    @classmethod
    def from_propagator(cls, propagator: dummyPropagatorB, population: int, max_gen: int, device:str):
        te = cls(propagator.state_dim, propagator.obs_dim, propagator.action_dim, population, max_gen, device)
        te.propagator = propagator
        return te
    
    def _stepProcess(self, agent: rlAgent, horizon=None):
        if self.seqOpt:
            if horizon is None:
                horizon = self.tree.max_gen//10
            states = self.tree.nodes.next_states[self.tree.gen-1]
            states = torch.from_numpy(states).to(agent.device)
            _ = self.propagator.seqOptTgt(states, agent, horizon=horizon)
    

class debugSingleEnv(singleEnv):
    def __init__(self, max_episode: int) -> None:
        self.propagator = debugPropagator()
        self.max_episode = max_episode

        self.state = np.zeros(self.propagator.state_dim, dtype=np.float32)
        self.episode = 0
    

class debugTreeEnv(treeEnv):
    def __init__(self, population: int, max_gen: int) -> None:
        self.propagator = debugPropagator()
        self.tree = GST(population, max_gen, self.propagator.state_dim, self.propagator.obs_dim, self.propagator.action_dim)

