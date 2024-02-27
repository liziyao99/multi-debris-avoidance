from typing import Tuple
import anytree
import numpy as np
import torch
import time, typing
from env.propagator import Propagator
from data.dicts import stateDict, init_transDict
from data.array import indexNode, treeDataArray
from agent.agent import rlAgent, boundedRlAgent


class stateNode(indexNode):
    def __init__(self, name, 
                 state_dim:int, obs_dim:int, action_dim:int, 
                 gen=-1, idx=-1, flags:typing.Tuple[str]=[]):
        super().__init__(name)
        self.sd = stateDict(state_dim, obs_dim, action_dim)
        self.td_target = 0.
        self.regret = 0.
        self.value = 0. # approximated value given by critic
        self.gen = gen
        self.idx = idx
        self.flags = {"dummy": False}
        '''
            default values are False.
        '''
        for flag in flags:
            self.flags[flag] = False

    def __str__(self) -> str:
        s = f"name:\t{self.name}\n"
        s += self.sd.__str__()
        s += f"td_target:\t{self.td_target}\n"
        s += f"regret:\t{self.regret}\n"
        s += f"value:\t{self.value}\n"
        return s
    
    def deflag(self, flags:typing.Tuple[str]=None):
        '''
            set all flag values to False.
        '''
        if flags is None:
            flags = self.flags.keys()
        for flag in flags:
            self.flags[flag] = False

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
    def __init__(self, population:int, max_gen:int, state_dim:int, obs_dim:int, action_dim:int, 
                 gamma=1., act_explore_eps=0.2, select_explore_eps=0.2) -> None:
        self.population = population
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma # discount factor
        self.act_explore_eps = act_explore_eps # acting exploration probability
        self.select_explore_eps = select_explore_eps # selection exploration probability
        stateNode_kwargs = {"state_dim": state_dim, "obs_dim": obs_dim, "action_dim": action_dim, 
                            "flags": ("traced",)}
        self.root = stateNode(name= 'root', **stateNode_kwargs)
        self.gen = 0
        self.nodes = []
        for i in range(max_gen):
            self.nodes.append([])
            for j in range(population):
                self.nodes[i].append(stateNode(name=f'gen{i}_node{j}', gen=i, idx=j,
                                               **stateNode_kwargs))
                if i==0:
                    self.nodes[i][j].parent = self.root
        self.nodes = np.array(self.nodes)

    def reset(self, root_stateDict:stateDict):
        self.gen = 0
        self.root.detach()
        self.root.deflag()
        self.root.sd = root_stateDict
        for i in range(self.max_gen):
            for j in range(self.population):
                self.nodes[i][j].detach()
                self.nodes[i][j].deflag()

    def deflag(self, flags:typing.Tuple[str]=None):
        '''
            deflag all nodes.
        '''
        self.root.deflag(flags)
        for i in range(self.max_gen):
            for j in range(self.population):
                self.nodes[i][j].deflag(flags)

    def step(self, agent:boundedRlAgent, propagator:Propagator, explore=True):
        '''
            return: `done`
        '''
        if self.gen<self.max_gen:
            states, obss, indices = self.select()
            batch_size = states.shape[0]
            # actor takes action
            _, a_exploit = agent.act(torch.from_numpy(obss).to(agent.device))
            a_exploit = a_exploit.detach().cpu().numpy()
            if explore:
                _, a_explore = agent.explore(batch_size)
                a_explore = a_explore.detach().cpu().numpy()
                explore_cond = np.tile(np.random.rand(batch_size)<self.act_explore_eps, (3,1)).T
                a = np.where(explore_cond, a_explore, a_exploit)
            else:
                a = a_exploit
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
            done = d.min()
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
            dist_exploit = torch.distributions.Categorical(torch.softmax(torch.tensor(fitness), dim=0))
            uid_indices_exploit = dist_exploit.sample((self.population,)).detach().cpu().numpy() # uid: indices of `undone_indices`
            uid_indices_explore = torch.randint(0, len(undone_indices), (self.population,)).detach().cpu().numpy()
            uid_indices = np.where(np.random.rand(self.population)<self.select_explore_eps, uid_indices_explore, uid_indices_exploit)
            indices = undone_indices[uid_indices]
            states = np.vstack([self.nodes[self.gen-1][i].state for i in indices])
            obss = np.vstack([self.nodes[self.gen-1][i].obs for i in indices])
        return states, obss, indices
    
    def get_transDicts(self, redundant=False):
        self.root.backup_regret(gamma=self.gamma)
        dicts = []
        for leaf in self.root.leaves:
            if not leaf.done:
                continue
            if redundant:
                lineage = list(leaf.ancestors) + [leaf]
            else:
                self.deflag(("traced",))
                lineage = [leaf]
                node = leaf.parent
                while not node.flags["traced"]:
                    lineage.append(node)
                    node.flags["traced"] = True
                    if not node.is_root:
                        node = node.parent
                    else:
                        break
                lineage.reverse()
            lineage = tuple(lineage)
            length = len(lineage)
            trans_dict = init_transDict(length, self.state_dim, self.obs_dim, self.action_dim,
                                        items=("td_targets", "regrets"))
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
    

    def pick(self, mode="descendants") -> stateDict:
        '''
            pick the best action from gen 0
        '''
        if mode=="descendants":
            idx = np.argmax([len(node.descendants) for node in self.nodes[0]])
        elif mode=="td_target":
            self.root.backup(gamma=self.gamma, mode="max")
            idx = np.argmax([node.td_target for node in self.nodes[0]])
        else:
            raise(ValueError("mode must be \"descendants\" or \"td_target\"."))
        return self.nodes[0][idx].sd
    
    def decide(self, root_stateDict, agent:rlAgent, propagator:Propagator, t_max:float, pick_mode="descendants"):
        '''
            pick the best action from gen 0 in given time `t`.\n
            args:
                `root_stateDict`: root stateDict for reset.
                `agent`: to make decision.
                `propagator`: to propagate state.
                `t_max`: max searching time each step, second.
        '''
        t0 = time.time()
        self.reset(root_stateDict)
        while time.time()-t0<t_max:
            done = self.step(agent, propagator, explore=False)
            if done:
                break
        return self.pick(mode=pick_mode)
    

class GST_A:
    def __init__(self, 
                 population: int, 
                 max_gen: int, 
                 state_dim: int, 
                 obs_dim: int, 
                 action_dim: int, 
                 gamma=1, 
                 act_explore_eps=0.2,
                 select_explore_eps=0.5
                ) -> None:
        self.population = population
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma # discount factor
        self.act_explore_eps = act_explore_eps # exploration probability
        self.select_explore_eps = select_explore_eps # exploration probability of selecting nodes

        self.gen = 0

        flags = ("traced",)
        items = ("values", "td_targets", "regrets", "advantages")

        self.root = stateNode(name='root', state_dim=state_dim, obs_dim=obs_dim, action_dim=action_dim, flags=flags)
        self.nodes = np.empty((max_gen, population), dtype=object)
        for i in range(max_gen):
            for j in range(population):
                self.nodes[i,j] = indexNode(name=f"gen{i}node{j}", gen=i, idx=j)

        self.data_array = treeDataArray(population, max_gen, state_dim, obs_dim, action_dim, flags, items)
        '''
            `states` transfer to `next_states` by applying `actions`, get `rewards` and `dones`.\n
            `items["values"]` are estimated values of `next_states`.\n
            `items["td_targets"]` are of `states`.\n
            `item["regrets"]` are of `actions`.\n
        '''

    def reset(self, root_stateDict: stateDict):
        self.gen = 0
        self.root.detach()
        self.root.deflag()
        self.root.sd = root_stateDict
        for node in self.nodes.flatten():
                node.detach()
        self.data_array.deflag()

    def deflag(self, flags: Tuple[str] = None):
        '''
            deflag all nodes.
        '''
        self.root.deflag(flags)
        self.data_array.deflag(flags)

    def step(self, agent:boundedRlAgent, propagator:Propagator, explore=True):
        '''
            return: `done`
        '''
        if self.gen<self.max_gen:
            states, obss, indices = self.select()
            batch_size = states.shape[0]
            # actor takes action
            _, a_exploit = agent.act(torch.from_numpy(obss).to(agent.device))
            a_exploit = a_exploit.detach().cpu().numpy()
            if explore:
                _, a_explore = agent.explore(batch_size)
                a_explore = a_explore.detach().cpu().numpy()
                explore_cond = np.tile(np.random.rand(batch_size)<self.act_explore_eps, (3,1)).T
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
            self.data_array.fill_gen(self.gen, states, obss, a, r, ns, no, d, values=v)
            for i in range(self.population):
                if self.gen>0:
                    self.nodes[self.gen][i].attach(self.nodes[self.gen-1][indices[i]])
                else: # gen == 0
                    self.nodes[self.gen][i].attach(self.root)
            done = d.min()
            self.gen += 1
        else:
            done = True
        return done
    
    def select(self):
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
            indices = None
        else:
            done = self.data_array.dones[self.gen-1].flatten()
            undone_indices = np.where(np.array(done)==False)[0]
            values = self.data_array.items["values"][self.gen-1][undone_indices].flatten()
            dist_exploit = torch.distributions.Categorical(torch.softmax(torch.tensor(values), dim=0))
            uid_indices_exploit = dist_exploit.sample((self.population,)).detach().cpu().numpy() # uid: indices of `undone_indices`
            uid_indices_explore = np.random.randint(0, len(undone_indices), (self.population,))
            uid_indices = np.where(np.random.rand(self.population)<self.select_explore_eps, uid_indices_explore, uid_indices_exploit)
            indices = undone_indices[uid_indices]
            states = self.data_array.next_states[self.gen-1][indices]
            obss = self.data_array.next_obss[self.gen-1][indices]
        return states, obss, indices
    
    def backup(self, mode="mean"):
        '''
            update td_targets and regrets.
        '''
        # update td_targets
        if self.gen<=0:
            raise(ValueError("gen must be greater than 0."))
        for gen in range(self.gen).__reversed__():
            leaves_td_targets = self.data_array.rewards[gen] + self.gamma*self.data_array.items["values"][gen]*(1-self.data_array.dones[gen])
            for idx in range(self.population):
                node = self.nodes[gen,idx]
                if node.is_leaf:
                    self.data_array.items["td_targets"][gen,idx,0] = leaves_td_targets[idx,0]
                else:
                    children_td_targets = self.data_array.items["td_targets"][gen+1,node.childrenIdx,0]
                    if mode=="mean":
                        next_td_target = np.mean(children_td_targets)
                    elif mode=="max":
                        next_td_target = np.max(children_td_targets)
                    else:
                        raise(ValueError("mode must be 'mean' or 'max'."))
                    self.data_array.items["td_targets"][gen,idx,0] = self.data_array.rewards[gen,idx,0] + self.gamma*next_td_target
        self.root.td_target = np.mean(self.data_array.items["td_targets"][0,:,0])
        # update regrets
        for gen in range(self.gen):
            for idx in range(self.population):
                node = self.nodes[gen,idx]
                if gen==0:
                    peer_td_targets = self.root.td_target
                else:
                    peer_td_targets = (self.data_array.items["td_targets"][node.gen-1,node.parentIdx,0]-self.data_array.rewards[node.gen-1,node.parentIdx,0])/self.gamma
                self.data_array.items["regrets"][gen,idx,0] = peer_td_targets-self.data_array.items["td_targets"][gen,idx,0]

    def get_transDicts(self, redundant=False):
        '''
            call `backup` first.
        '''
        # self.backup()
        dicts = []
        self.deflag(("traced",))
        for leaf in self.root.leaves:
            if not self.data_array.dones[leaf.gen,leaf.idx]:
                continue
            if redundant:
                lineage = list(leaf.ancestors)[1:] + [leaf] # exclude root
            else:
                lineage = [leaf]
                node = leaf.parent
                while not self.data_array.flags["traced"][node.gen,node.idx]:
                    if node.is_root:
                        break
                    else:
                        lineage.append(node)
                        self.data_array.flags["traced"][node.gen,node.idx] = True
                        node = node.parent
                lineage.reverse()
            lineage = tuple(lineage)
            length = len(lineage)
            trans_dict = init_transDict(length, self.state_dim, self.obs_dim, self.action_dim,
                                        items=("td_targets", "regrets"))
            for i in range(length):
                gen, idx = lineage[i].gen, lineage[i].idx
                trans_dict["states"][i,:] = self.data_array.states[gen,idx]
                trans_dict["obss"][i,:] = self.data_array.obss[gen,idx]
                trans_dict["actions"][i,:] = self.data_array.actions[gen,idx]
                trans_dict["rewards"][i] = self.data_array.rewards[gen,idx]
                trans_dict["next_states"][i,:] = self.data_array.next_states[gen,idx]
                trans_dict["next_obss"][i,:] = self.data_array.next_obss[gen,idx]
                trans_dict["dones"][i] = self.data_array.dones[gen,idx,0]
                for key in self.data_array.items.keys():
                    if key in trans_dict.keys():
                        trans_dict[key][i] = self.data_array.items[key][gen,idx,0]
            dicts.append(trans_dict)
        return dicts
    
    def pick(self, mode="descendants") -> stateDict:
        '''
            pick the best action from gen 0
        '''
        if mode=="descendants":
            idx = np.argmax([len(node.descendants) for node in self.nodes[0]])
        elif mode=="td_target":
            self.backup(mode="max")
            idx = np.argmax([node.td_target for node in self.nodes[0]])
        else:
            raise(ValueError("mode must be \"descendants\" or \"td_target\"."))
        return self.nodes[0][idx].sd
    
    def decide(self, root_stateDict, agent:rlAgent, propagator:Propagator, t_max:float, pick_mode="descendants"):
        '''
            pick the best action from gen 0 in given time `t`.\n
            args:
                `root_stateDict`: root stateDict for reset.
                `agent`: to make decision.
                `propagator`: to propagate state.
                `t_max`: max searching time each step, second.
        '''
        t0 = time.time()
        self.reset(root_stateDict)
        while time.time()-t0<t_max:
            done = self.step(agent, propagator, explore=False)
            if done:
                break
        return self.pick(mode=pick_mode)