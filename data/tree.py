from data import dicts as D
from data.array import nodesArray, edgesArray
from agent import agent
from env import propagator

import typing
import numpy as np
import torch
import time

class GST:
    def __init__(self, 
                 population: int, 
                 max_gen: int, 
                 state_dim: int, 
                 obs_dim: int, 
                 action_dim: int, 
                 gamma=1, 
                 act_explore_eps=0.5,
                 select_explore_eps=0.2
                ) -> None:
        self.population = population
        self._init_pop = population
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma # discount factor
        self.act_explore_eps = act_explore_eps # exploration probability
        self.select_explore_eps = select_explore_eps # exploration probability of selecting nodes

        self.gen = 0
        '''
            indicate the next generation, `gen-1` is the last genaration has been completed.\n
            Initially 0. `step` will complete `gen` and then `gen+=1`.
        '''

        flags = ("preserved", "aborted")
        items = ("values", "td_targets", "regrets", "advantages")
        self.flag_keys = flags
        self.item_keys = items

        self.root = D.stateDict(state_dim, obs_dim, action_dim, flags=flags, items=items)

        self.nodes = nodesArray(population, max_gen, state_dim, obs_dim, action_dim, flags=flags, items=items)
        '''
            `states` transfer to `next_states` by applying `actions`, get `rewards` and `dones`.\n
            `items["values"]` are estimated values of `next_obss`.\n
            `items["td_targets"]` are of `obss`.\n
            `item["regrets"]` are of `actions`.\n
        '''
        self.edges = edgesArray(population, max_gen, self.root)

    def reset(self, root_stateDict:D.stateDict):
        self.root.load(root_stateDict)
        self.gen = 0
        self.nodes.deflag()
        self.edges.clear()

    def deflag(self, flags: typing.Tuple[str] = None):
        '''
            deflag all nodes.
        '''
        self.root.deflag(flags)
        self.nodes.deflag(flags)

    def step(self, 
             agent:agent.boundedRlAgent, 
             propagator:propagator.Propagator, 
             explore_action=True, 
             explore_select=True):
        '''
            return: `done`
        '''
        if self.gen<self.max_gen:
            states, obss, indices = self.select(explore=explore_select)
            batch_size = states.shape[0]
            # actor takes action
            _, a_exploit = agent.act(torch.from_numpy(obss).to(agent.device))
            a_exploit = a_exploit.detach().cpu().numpy()
            if explore_action:
                _, a_explore = agent.explore(batch_size)
                a_explore = a_explore.detach().cpu().numpy()
                explore_cond = np.tile(np.random.rand(batch_size)<self.act_explore_eps, (self.action_dim,1)).T
                a = np.where(explore_cond, a_explore, a_exploit)
            else:
                a = a_exploit
            # env propagates
            ns, r, d, no = propagator.propagate(states, a)
            if self.gen==self.max_gen-1: 
                d[:] = True
            # critic evaluates new states
            v = agent.critic(torch.from_numpy(no).to(agent.device)).detach().cpu().numpy()
            # nodes attach to parents
            self.nodes.fill_gen(self.gen, states, obss, a, r, ns, no, d, values=v)
            for i in range(self.population):
                if self.gen>0:
                    self.edges.attach(child_gen=self.gen, child_idx=i, parent_idx=indices[i])
            done = d.min()
            self.gen += 1
        else:
            done = True
        return done
    
    def select(self, 
               explore=True
               ) -> typing.Tuple[np.ndarray[float],np.ndarray[float],np.ndarray[int]]:
        '''
            select nodes for next gen.\n
            returns:
                `states`: last gen's next states wrt `indices`.
                `obss`: last gen's next observations wrt `indices`.
                `indices`: selected indices from last gen undone nodes, sampled by fitness or explored uniformally.
        '''
        if self.gen == 0:
            states = np.tile(self.root.state, (self.population,1))
            obss = np.tile(self.root.obs, (self.population,1))
            indices = -np.ones(self.population, dtype=np.int32)
        else:
            done = self.nodes.dones[self.gen-1].flatten()
            undone_indices = np.where(np.array(done)==False)[0]
            values = self.nodes.items["values"][self.gen-1][undone_indices].flatten()
            fitness = torch.softmax(torch.from_numpy(values), dim=0)
            dist_exploit = torch.distributions.Categorical(fitness)
            uid_indices_exploit = dist_exploit.sample((self.population,)).detach().cpu().numpy() # uid: indices of `undone_indices`
            if explore:
                uid_indices_explore = np.random.randint(0, len(undone_indices), (self.population,))
                uid_indices = np.where(np.random.rand(self.population)<self.select_explore_eps, uid_indices_explore, uid_indices_exploit)
            else:
                uid_indices = uid_indices_exploit
            indices = undone_indices[uid_indices]
            states = self.nodes.next_states[self.gen-1,indices]
            obss = self.nodes.next_obss[self.gen-1,indices]
        return states, obss, indices
    
    def backup(self, mode="mean"):
        '''
            update td_targets and descendants.
        '''
        # update td_targets
        if self.gen<=0:
            raise(ValueError("gen must be greater than 0."))
        for gen in range(self.gen).__reversed__():
            leaves_td_targets = self.nodes.rewards[gen] + self.gamma*self.nodes.items["values"][gen]*(1-self.nodes.dones[gen])
            for idx in range(self.population):
                if self.isLeaf(gen, idx):
                    self.nodes.items["td_targets"][gen,idx,0] = leaves_td_targets[idx,0]
                else:
                    children_td_targets = self.nodes.items["td_targets"][gen+1,self.edges.childrenTable[gen][idx],0]
                    if mode=="mean":
                        next_td_target = np.mean(children_td_targets)
                    elif mode=="max":
                        next_td_target = np.max(children_td_targets)
                    else:
                        raise(ValueError("mode must be 'mean' or 'max'."))
                    self.nodes.items["td_targets"][gen,idx,0] = self.nodes.rewards[gen,idx,0] + self.gamma*next_td_target
        self.root.items["td_target"] = np.mean(self.nodes.items["td_targets"][0,:,0])

        # update descendants as regrets
        self.edges.updateDescNum(self.gen-1)
        for g in range(self.gen):
            total = (self.gen-g)*self.population
            normalized_num = (self.edges.desc_num[g]+1)/total
            self.nodes.items["regrets"][g,:,0] = -normalized_num[:]

        # update regrets, now deprived
        # TODO: why regret makes training worse? 

    def get_transDicts(self, redundant=False):
        self.backup()
        dicts = []
        # self.deflag(("traced",))
        self.edges.traced_flags[...] = False
        leaves = self.leaves
        for leaf in leaves:
            gen, idx = leaf
            if not self.nodes.dones[gen,idx]:
                continue
            if redundant:
                lineage = self.pathTo(gen, idx, trace=False, toTraced=False)
            else:
                lineage = self.pathTo(gen, idx, trace=True, toTraced=True)[1:]
            lineage = tuple(lineage)
            length = len(lineage)
            trans_dict = D.init_transDict(length, self.state_dim, self.obs_dim, self.action_dim,
                                          items=self.item_keys)
            for i in range(length):
                gen, idx = lineage[i]
                trans_dict["states"][i,:] = self.nodes.states[gen,idx]
                trans_dict["obss"][i,:] = self.nodes.obss[gen,idx]
                trans_dict["actions"][i,:] = self.nodes.actions[gen,idx]
                trans_dict["rewards"][i] = self.nodes.rewards[gen,idx]
                trans_dict["next_states"][i,:] = self.nodes.next_states[gen,idx]
                trans_dict["next_obss"][i,:] = self.nodes.next_obss[gen,idx]
                trans_dict["dones"][i] = self.nodes.dones[gen,idx,0]
                for key in self.item_keys:
                    trans_dict[key][i] = self.nodes.items[key][gen,idx,0]
            dicts.append(trans_dict)
        return dicts
    
    def pick(self) -> D.stateDict:
        '''
            pick the best action from gen 0
            TODO: descendants mode
        '''
        self.backup(mode="max")
        idx = np.argmax(self.nodes.items["td_targets"][0,:,0])
        sd = D.stateDict.from_data(state=self.nodes.next_states[0,idx],
                                    action=self.nodes.actions[0,idx],
                                    reward=self.nodes.rewards[0,idx,0],
                                    done=self.nodes.dones[0,idx,0],
                                    obs=self.nodes.next_obss[0,idx])
        return sd

    def decide(self, root_stateDict, agent:agent.rlAgent, propagator:propagator.Propagator, decide_mode="time", t_max:float=.1, g_max=5):
        '''
            pick the best action from gen 0 in given time `t`.\n
            args:
                `root_stateDict`: root stateDict for reset.
                `agent`: to make decision.
                `propagator`: to propagate state.
                `t_max`: max searching time each step, second.
        '''
        self.reset(root_stateDict)
        if decide_mode=="time":
            t0 = time.time()
            while time.time()-t0<t_max:
                done = self.step(agent, propagator, explore_action=False, explore_select=False)
                if done:
                    break
        elif decide_mode=="gen":
            for _ in range(g_max):
                done = self.step(agent, propagator, explore_action=False, explore_select=False)
                if done:
                    break
        else:
            raise(ValueError("decide_mode must be 'time' , 'gen'."))
        return self.pick()

    def parentOf(self, gen, idx) -> typing.Tuple[int,int]:
        '''
            return parent's idx of node `idx` in generation `gen`.
        '''
        return self.edges.parentOf(gen, idx)

    def childrenOf(self, gen, idx) -> typing.List[typing.Tuple[int,int]]:
        '''
            return children's idx of node `idx` in generation `gen`.
        '''
        return self.edges.childrenOf(gen, idx)
    
    def siblingsOf(self, gen, idx, exclude_self=False) -> typing.List[typing.Tuple[int,int]]:
        '''
            return siblings's idx of node `idx` in generation `gen`.
        '''
        return self.edges.siblingsOf(gen, idx, exclude_self)
    
    def pathTo(self, gen, idx, trace=False, toTraced=False) -> typing.List[typing.Tuple[int,int]]:
        return self.edges.pathTo(gen, idx, trace=trace, toTraced=toTraced)

    def isLeaf(self, gen, idx):
        if gen>=self.gen:
            raise(ValueError("gen>=self.gen"))
        return self.edges.isLeaf(gen, idx, neglectTop=True)
    
    @property
    def leaves(self):
        return self.edges.leaves(last_gen=self.gen-1)