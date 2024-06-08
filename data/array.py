import numpy as np
import anytree
import typing

from data import dicts as D

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

class nodesArray:
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


class edgesArray:
    def __init__(self, population, max_gen, root:D.stateDict) -> None:
        self.max_gen = max_gen
        self.population = population
        self.parentsTable = np.zeros((max_gen, population), dtype=np.int32)
        '''
            parent's index of each node, -1 if not attach to a parent. 
            Parents of 0th gen are always -1, because they all attach to a virtual root.
        '''
        self.parentsTable[...] = -1

        self.childrenTable = [[[]for i in range(population)] for j in range(max_gen)]
        '''
            list of lists of lists.\n
            Children's index of each node, empty list if not attach to any children.
        '''
        self.root = root
        '''
            state of root, will boardcast to gen0 nodes.
        '''
        self.traced_flags = np.zeros((max_gen, population), dtype=np.bool_)
        '''
            True if {gen,idx} is attached to root and have been processed in some function.
        '''
        self.desc_num = np.zeros((max_gen, population), dtype=np.int32)
        '''
            recording number of descendants. Will NOT update automatically, please call `updateDescNum` first.
        '''

    def attach(self, child_gen, child_idx, parent_idx):
        '''
            attach child to parent.
        '''
        if child_gen<=0:
            raise(ValueError("child_gen must be greater than 0"))
        self.parentsTable[child_gen,child_idx] = parent_idx
        self.childrenTable[child_gen-1][parent_idx].append(child_idx)

    def detach(self, child_gen, child_idx):
        if child_gen<=0:
            raise(ValueError("child_gen must be greater than 0"))
        parent_idx = self.parentsTable[child_gen][child_idx]
        if child_idx not in self.childrenTable[child_gen-1][parent_idx]:
            # child_idx is not a child of parent_idx
            return
        self.parentsTable[child_gen][child_idx] = -1
        self.childrenTable[child_gen-1][parent_idx].remove(child_idx)

    def clear(self):
        '''
            set all nodes' parent to -1, clear all children.
        '''
        self.parentsTable[...] = -1
        for g in range(self.max_gen):
            for p in range(self.population):
                self.childrenTable[g][p].clear()

    def parentOf(self, gen, idx) -> typing.Tuple[int,int]:
        '''
            return parent's idx of node `idx` in generation `gen`.
        '''
        if gen<=0:
            raise(ValueError("child_gen must be greater than 0"))
        if self.parentsTable[gen,idx]==-1:
            return None, None
        else:
            return gen-1, self.parentsTable[gen,idx]
        
    def childrenOf(self, gen, idx) -> typing.Tuple[typing.Tuple[int,int]]:
        '''
            return children's idx of node `idx` in generation `gen`.
        '''
        if gen<0:
            raise(ValueError("gen<0"))
        if gen>=self.max_gen-1:
            raise(ValueError("gen>=max_gen"))
        return tuple([(gen+1, child_idx) for child_idx in self.childrenTable[gen][idx]])
    
    def siblingsOf(self, gen, idx, exclude_self=False) -> typing.Tuple[typing.Tuple[int,int]]:
        '''
            return siblings's idx of node `idx` in generation `gen`.
        '''
        if gen<0:
            raise(ValueError("gen<0"))
        if gen==0: # gen 0 are all root's children.
            siblings = [(0, i) for i in range(self.population)]
        else:
            parent_idx = self.parentsTable[gen,idx]
            if parent_idx==-1: # no parent
                return []
            else:
                siblings = [(gen, child_idx) for child_idx in self.childrenTable[gen-1][parent_idx]]
        if exclude_self:
            siblings.remove((gen,idx))
        return tuple(siblings)
    
    def pathTo(self, gen, idx, trace=False, toTraced=False) -> typing.Tuple[typing.Tuple[int,int]]:
        '''
            path from root to {gen,idx}.
            Notice that {gen,idx} may not be a descendant of root, please check `path[0][0]==0`.\n
            If `trace` AND {gen,idx} trace back to gen0, set `self.traced_flags` to True for all nodes in the path.\n
            If `toTraced` is activated, then path will be truancated at the first traced node on the path upward, 
            this traced node will be included in the path.
        '''
        if gen<0:
            raise(ValueError("gen<0"))
        path = []
        g = gen
        i = idx
        while g>=0:
            if i==-1:
                break
            path.append((g, i))
            node_traced = self.traced_flags[g, i]
            if toTraced and node_traced:
                break
            i = self.parentsTable[g, i]
            g -= 1
        if trace and (g==-1 or node_traced):
            for pair in path:
                self.traced_flags[*pair] = True # python >= 3.11
        return tuple(path.__reversed__())
    
    def isLeaf(self, gen, idx, neglectTop=True, trace=False, traced=False) -> bool:
        '''
            return Ture only if {gen,idx} is a leaf AND attached to root.\n
            If `neglectTop` is activated, only consider children of {gen,idx}. 
            Deactivate it when you are not sure whether the structure is legal\n
            `traced` means `trace` has been activated in previous `isLeaf` and `pathTo`, 
            thus if a node is leaf and descendante of a traced node, it's also a leaf of root.\n
            If `traced` is activated, then so do `trace`.
        '''
        if gen<0:
            raise(ValueError("gen<0"))
        if neglectTop:
            return len(self.childrenTable[gen][idx])==0
        if not traced:
            if len(self.childrenTable[gen][idx])==0:
                path = self.pathTo(gen, idx, trace=trace, toTraced=False)
                top_gen = path[0][0]
                if top_gen==0:
                    return True
                else: # not attached to root
                    return False
            else:
                return False
        else: # `traced` is activated, so do `trace`.
            path = self.pathTo(gen, idx, trace=True, toTraced=True)
            if len(path)==0: # current node had been traced.
                return len(self.childrenTable[gen][idx])==0
            else:
                top_gen, top_idx = path[0]
                if top_gen==0:
                    return len(self.childrenTable[gen][idx])==0
                elif self.traced_flags[top_gen,top_idx]:
                    return len(self.childrenTable[gen][idx])==0
                else:
                    return False

    def leaves(self, last_gen:int=None) -> typing.Tuple[typing.Tuple[int,int]]:
        if last_gen is None:
            last_gen = 0
            while np.max([len(childrenList) for childrenList in self.childrenTable[last_gen]]):
                last_gen += 1
        L = []
        g = last_gen
        while g>=0:
            for i in range(self.population):
                if self.isLeaf(g, i, neglectTop=True):
                    L.append((g, i))
            g -= 1
        return tuple(L)
    
    def updateDescNum(self, last_gen:int=None):
        '''
            update `self.desc_num` for all nodes.
        '''
        if last_gen is None:
            last_gen = 0
            while np.max([len(childrenList) for childrenList in self.childrenTable[last_gen]]):
                last_gen += 1
        for g in range(last_gen).__reversed__():
            for i in range(self.population):
                self.desc_num[g,i] = len(self.childrenTable[g][i])
                for child_idx in self.childrenTable[g][i]:
                    self.desc_num[g,i] += self.desc_num[g+1,child_idx]