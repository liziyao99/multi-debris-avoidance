import anytree
import numpy as np
import torch
import time
from env.propagator import Propagator
from data.dicts import stateDict, init_transDict
from agent.agent import rlAgent


class stateNode(anytree.Node):
    def __init__(self, name, 
                 state_dim:int, obs_dim:int, action_dim:int, 
                 gen=-1, idx=-1):
        super().__init__(name)
        self.sd = stateDict(state_dim, obs_dim, action_dim)
        self.td_target = 0.
        self.regret = 0.
        self.value = 0. # approximated value given by critic
        self.gen = gen
        self.idx = idx

    def __str__(self) -> str:
        s = f"name:\t{self.name}\n"
        s += self.sd.__str__()
        s += f"td_target:\t{self.td_target}\n"
        s += f"regret:\t{self.regret}\n"
        s += f"value:\t{self.value}\n"
        return s

    def backup(self, gamma=1., mode="mean"):
        '''
            args:
                `self` is root.
                `mode`: "max" or "mean".
        '''
        if self.is_leaf:
            self.td_target = self.reward if self.done else self.reward+gamma*self.value
        else:
            next_td_targets = [child.backup(gamma) for child in self.children]
            if mode=="max":
                self.td_target = self.reward + gamma*np.max(next_td_targets)
            elif mode=="mean":
                self.td_target = self.reward + gamma*np.mean(next_td_targets)
            else:
                raise(ValueError("mode must be \"max\" or \"mean\"."))
        return self.td_target
    
    def update_regret(self, gamma=1.):
        '''
            `self` is root.
            call `backup` first.
        '''
        peer_td_target = (self.td_target-self.reward)/gamma # mean or max td_target of self.children
        for child in self.children:
            child.regret = peer_td_target - child.td_target # regret of each child
            child.update_regret() # push down
        return
    
    def backup_regret(self, gamma=1.):
        '''
            args:
                `self` is root.
                `mode`: "max" or "mean".
        '''
        self.backup(gamma, "mean")
        self.update_regret(gamma)
    
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
    def state(self):
        return self.sd.state
    
    @property
    def action(self):
        return self.sd.action
    
    @property
    def reward(self):
        return self.sd.reward
    
    @property
    def done(self):
        return self.sd.done
    
    @property
    def obs(self):
        return self.sd.obs

class GST:
    def __init__(self, population:int, max_gen:int, state_dim:int, obs_dim:int, action_dim:int, gamma=1.) -> None:
        self.population = population
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma # discount factor
        self.root = stateNode('root', state_dim, obs_dim, action_dim)
        self.gen = 0
        self.nodes = []
        for i in range(max_gen):
            self.nodes.append([])
            for j in range(population):
                self.nodes[i].append(stateNode(f'gen{i}_node{j}',state_dim, obs_dim, action_dim, gen=i, idx=j))
                if i==0:
                    self.nodes[i][j].parent = self.root
        self.nodes = np.array(self.nodes)

    def reset(self, root_stateDict:stateDict):
        self.gen = 0
        self.root.detach()
        self.root.sd = root_stateDict
        for i in range(self.max_gen):
            for j in range(self.population):
                self.nodes[i][j].detach()

    def step(self, agent:rlAgent, propagator:Propagator):
        '''
            return: `done`
        '''
        if self.gen<self.max_gen:
            states, obss, indices = self.select()
            # actor takes action
            _, a = agent.act(torch.from_numpy(obss).to(agent.device))
            a = a.detach().cpu().numpy()
            # env propagates
            ns, r, d, no = propagator.propagate(states, a)
            if self.gen==self.max_gen-1: 
                d[:] = True
            # critic evaluates new states
            v = agent.critic(torch.from_numpy(no).to(agent.device))
            # nodes attach to parents
            for i in range(self.population):
                sd = stateDict.from_data(ns[i], a[i], r[i], d[i], no[i])
                self.nodes[self.gen][i].sd = sd
                self.nodes[self.gen][i].value = v[i].item()
                if self.gen>0:
                    self.nodes[self.gen][i].attach(self.nodes[self.gen-1][indices[i]])
                else: # gen == 0
                    self.nodes[self.gen][i].attach(self.root)
            done = bool(np.min([node.done for node in self.nodes[self.gen]]))
            self.gen += 1
        else:
            done = True
        return done

    def select(self):
        '''
            select nodes for next gen.\n
            returns:
                `states`: states wrt `indices`.
                `obss`: observations wrt `indices`.
                `indices`: selected indices from last gen undone nodes, sampled by fitness.
        '''
        if self.gen == 0:
            states = np.tile(self.root.state, (self.population,1))
            obss = np.tile(self.root.obs, (self.population,1))
            indices = None
        else:
            done = [node.done for node in self.nodes[self.gen-1]]
            undone_indices = np.where(np.array(done)==False)[0]
            fitness = [node.value for node in self.nodes[self.gen-1][undone_indices]]
            dist = torch.distributions.Categorical(torch.softmax(torch.tensor(fitness), dim=0))
            uid_indices = dist.sample((self.population,)).detach().cpu().numpy() # indices of `undone_indices`
            indices = undone_indices[uid_indices]
            states = np.vstack([self.nodes[self.gen-1][i].state for i in indices])
            obss = np.vstack([self.nodes[self.gen-1][i].obs for i in indices])
        return states, obss, indices
    
    def get_transDicts(self):
        self.root.backup_regret(gamma=self.gamma)
        dicts = []
        for leaf in self.root.leaves:
            if not leaf.done:
                continue
            lineage = list(leaf.ancestors) + [leaf]
            lineage = tuple(lineage)
            length = len(lineage)
            trans_dict = init_transDict(length, self.state_dim, self.obs_dim, self.action_dim)
            for i in range(length):
                trans_dict["states"][i,:] = lineage[i].state
                trans_dict["obss"][i,:] = lineage[i].obs
                trans_dict["dones"][i] = lineage[i].done
                trans_dict["td_targets"][i] = lineage[i].td_target
                trans_dict["regrets"][i] = lineage[i].regret
                if i<length-1:
                    trans_dict["actions"][i,:] = lineage[i+1].action
                    trans_dict["rewards"][i] = lineage[i+1].reward
                    trans_dict["next_states"][i,:] = lineage[i+1].state
                    trans_dict["next_obss"][i,:] = lineage[i+1].obs
            dicts.append(trans_dict)
        return dicts
    

    def pick(self) -> stateDict:
        '''
            pick the best action from gen 0
        '''
        idx = np.argmax([len(node.descendants) for node in self.nodes[0]])
        return self.nodes[0][idx].sd
    
    def decide(self, root_stateDict, agent:rlAgent, propagator:Propagator, t_max:float):
        '''
            pick the best action from gen 0 in given time `t`.\n
            args:
                `root_stateDict`: root stateDict for reset.
                `agent`: to make decision.
                `propagator`: to propagate state.
                `t_max`: max calculating time, second.
        '''
        t0 = time.time()
        self.reset(root_stateDict)
        while time.time()-t0<t_max:
            done = self.step(agent, propagator)
            if done:
                break
        return self.pick()