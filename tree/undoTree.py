import anytree
import numpy as np
import torch
import typing
import data.dicts as D

class stateNode(anytree.Node):
    def __init__(self, 
                 state_dim:int, 
                 action_dim:int, 
                 obs_dim:int,
                 name, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gen = None
        self.idx = None

        self.state = np.zeros(state_dim, dtype=np.float32)
        self.action = np.zeros(action_dim, dtype=np.float32)
        '''
            action that make transition from parent to this node.
            Notice that root node has no action.
        '''
        self.obs = np.zeros(obs_dim, dtype=np.float32)

        self.done = False
        self.reward = None
        '''
            reward recived from parent's state and self.action.
            Notice that root node has no reward.
        '''
        self.terminal_reward = 0.
        '''
            reward recived from this node if is terminal.
        '''

        self.V_target = None
        '''
            value target for self.state.
        '''
        self.Q_target = None
        '''
            Q value target for parent's state and self.action.
            Notice that root node has no Q_target.
        '''
        self.regret_mc = None
        '''
            self.action's regret about parent's state.
            "mc" means this value was calculated by Monte-Carlo method: `self.parent.V_target - self.Q_target`.
            Notice that root node has no regret.
        '''
        self.regret_td = None
        '''
            self.action's regret about parent's state.
            "td" means this value was calculated by temporal difference method: `critic(...) - self.Q_target`.
            Notice that root node has no regret.
        '''

    def __str__(self) -> str:
        return f"gen:{self.gen} idx:{self.idx}"
    
class undoTree:
    def __init__(self, max_gen:int, state_dim, obs_dim, action_dim, gamma=0.95, act_explore_eps=0.2, select_explore_eps=0.2) -> None:
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma # discount factor
        self.act_explore_eps = act_explore_eps # acting exploration probability
        self.select_explore_eps = select_explore_eps # selection exploration probability
        self.gens = [[] for _ in range(max_gen)]
        self.nodes = []

    def clear(self):
        self.gens = [[] for _ in range(self.max_gen+1)]
        self.nodes = []

    def reset_root(self, state:np.ndarray, obs:np.ndarray,):
        self.clear()
        root = stateNode(self.state_dim, self.action_dim, self.obs_dim,
                        name="root")
        root.gen = 0
        root.idx = 0
        root.state[...] = state[...]
        root.obs[...] = obs[...]
        self.gens[0].append(root)
        self.nodes.append(root)

    @property
    def root(self) -> stateNode:
        return self.gens[0][0]
    
    @property
    def N(self):
        '''
            number of nodes.
        '''
        return len(self.nodes)
    
    def new_node(self, parent_gen:int, parent_idx:int, action:np.ndarray, state:np.ndarray, obs:np.ndarray, reward:float, done:bool, terminal_reward=0.):
        '''
            returns: `new_node`, `new_node`'s gen, `new_node`'s idx.
        '''
        if parent_gen>=self.max_gen:
            raise(ValueError("parent_gen out of range."))
        parent_node = self.gens[parent_gen][parent_idx]
        new_node = stateNode(self.state_dim, self.action_dim, self.obs_dim, 
                             name=f"gen{parent_gen}-idx{parent_idx}",
                             parent=parent_node)
        new_node.state[...] = state[...]
        new_node.obs[...] = obs[...]
        new_node.action[...] = action[...]
        new_node.reward = reward
        new_node.done = done
        new_node.terminal_reward = terminal_reward*done
        new_node.gen = parent_gen+1
        new_node.idx = len(self.gens[parent_gen+1])
        self.gens[parent_gen+1].append(new_node)
        self.nodes.append(new_node)
        return new_node, parent_gen+1, len(self.gens[parent_gen+1])-1
    
    def new_traj(self, parent_gen:int, parent_idx:int, trans_dict:dict):
        length = trans_dict["states"].shape[0]
        flag = ("terminal_rewards" in trans_dict.keys())
        gen = parent_gen
        idx = parent_idx
        for i in range(length):
            state = trans_dict["next_states"][i]
            obs = trans_dict["next_obss"][i]
            action = trans_dict["actions"][i]
            reward = trans_dict["rewards"][i]
            done = trans_dict["dones"][i]
            terminal_reward = trans_dict["terminal_rewards"][i] if flag else 0.
            _, gen, idx = self.new_node(gen, idx, action, state, obs, reward, done, terminal_reward)
            if done:
                break

    def backup(self, mode="mean"):
        '''
            args:
                `mode`: "max" or "mean".
        '''
        if mode not in ["max", "mean"]:
            raise(ValueError("mode must be \"max\" or \"mean\"."))
        for g in range(self.max_gen,-1,-1):
            for node in self.gens[g]:
                if node.is_leaf:
                    if not node.done:
                        raise(ValueError("leaves must be terminal states."))
                    node.V_target = node.terminal_reward
                    if not node.is_root:
                        node.Q_target = node.reward + self.gamma*node.terminal_reward
                else:
                    next_V_targets = np.array([child.V_target for child in node.children], dtype=np.float32)
                    next_Q_targets = np.array([child.Q_target for child in node.children], dtype=np.float32)
                    rewards = np.array([child.reward for child in node.children], dtype=np.float32)
                    if mode=="max":
                        node.V_target = np.max(rewards + self.gamma*next_V_targets)
                        if not node.is_root:
                            node.Q_target = node.reward + self.gamma*np.max(next_Q_targets)
                    elif mode=="mean":
                        node.V_target = np.mean(rewards + self.gamma*next_V_targets)
                        if not node.is_root:
                            node.Q_target = node.reward + self.gamma*np.mean(next_Q_targets)
        return
    
    def update_regret_mc(self):
        nodes = self.nodes[1:] # exclude root
        for node in nodes:
            node.regret_mc = node.parent.V_target - node.Q_target
        return

    def update_regret_td(self, critic):
        nodes = self.nodes[1:] # exclude root
        obss = [node.obs for node in nodes]
        obss = np.vstack(obss)
        obss = torch.from_numpy(obss).to(torch.float32).to(critic.device)
        V_targets = critic(obss).detach().cpu().numpy().flatten()
        for i in range(len(nodes)):
            nodes[i].regret_mc = V_targets[i] - nodes[i].Q_target
        return

    def to_transdict(self):
        if self.N<1:
            raise(ValueError("tree must have at least one non-root node."))
        trans_dict = D.init_transDict(self.N-1, self.state_dim, self.obs_dim, self.action_dim,
                                      items=("Q_targets", "V_targets", "regret_mc"))
        for i in range(self.N-1):
            node = self.nodes[i+1]
            trans_dict["states"][i,:] = node.parent.state
            trans_dict["obss"][i,:] = node.parent.obs
            trans_dict["actions"][i,:] = node.action
            trans_dict["rewards"][i] = node.reward
            trans_dict["next_states"][i,:] = node.state
            trans_dict["next_obss"][i,:] = node.obs
            trans_dict["dones"][i] = node.done
            trans_dict["Q_targets"][i] = node.Q_target
            trans_dict["V_targets"][i] = node.parent.V_target
            trans_dict["regret_mc"][i] = node.regret_mc
        return trans_dict

    def select(self, size, mode="uniform"):
        if mode not in ["uniform", "regret"]:
            raise(ValueError("mode must be \"uniform\" or \"regret\"."))
        nodes = [node for node in self.nodes if not node.done]
        if mode=="uniform":
            selected = np.random.choice(nodes, size)
        elif mode=="regret":
            raise(NotImplementedError)
        states = [n.state for n in selected]
        idx_pairs = [(n.gen, n.idx) for n in selected]
        return states, idx_pairs
        

