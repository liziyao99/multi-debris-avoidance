import anytree
import numpy as np
import torch
import typing

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

    def backup(self, gamma=1., mode="mean"):
        '''
            args:
                `mode`: "max" or "mean".
            TODO: remove iterition.
        '''
        if self.is_leaf:
            if not self.done:
                raise(ValueError("leaves must be terminal states."))
            self.V_target = self.terminal_reward
            self.Q_target = self.reward + gamma*self.terminal_reward
        else:
            next_targets = [child.backup(gamma, mode) for child in self.children]
            next_V_targets = np.array([target[0] for target in next_targets], dtype=np.float32)
            next_Q_targets = np.array([target[1] for target in next_targets], dtype=np.float32)
            rewards = np.array([child.reward for child in self.children], dtype=np.float32)
            if mode=="max":
                self.V_target = np.max(rewards + gamma*next_V_targets)
                self.Q_target = self.reward + gamma*np.max(next_Q_targets)
            elif mode=="mean":
                self.V_target = np.mean(rewards + gamma*next_V_targets)
                self.Q_target = self.reward + gamma*np.mean(next_Q_targets)
            else:
                raise(ValueError("mode must be \"max\" or \"mean\"."))
        return self.V_target, self.Q_target
    
    def update_regret_mc(self):
        nodes = list(self.descendants)
        if not self.is_root:
            nodes = [self] + nodes
        for node in nodes:
            node.regret_mc = node.parent.V_target - node.Q_target
        return
    
    def update_regret_td(self, critic):
        nodes = list(self.descendants)
        if not self.is_root:
            nodes = [self] + nodes
        obss = [node.obs for node in nodes]
        obss = np.vstack(obss)
        obss = torch.from_numpy(obss).to(torch.float32).to(critic.device)
        V_targets = critic(obss).detach().cpu().numpy().flatten()
        for i in range(len(nodes)):
            nodes[i].regret_mc = V_targets[i] - nodes[i].Q_target
        return
    
class undoTree:
    def __init__(self, max_gen:int, state_dim, obs_dim, action_dim, gamma=1., act_explore_eps=0.2, select_explore_eps=0.2) -> None:
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma # discount factor
        self.act_explore_eps = act_explore_eps # acting exploration probability
        self.select_explore_eps = select_explore_eps # selection exploration probability
        self.gens = [[] for _ in range(max_gen)]

    def clear(self):
        self.gens = [[] for _ in range(self.max_gen)]

    def reset_root(self, root:stateNode):
        self.clear()
        self.gens[0] = [root]

    @property
    def root(self):
        return self.gens[0][0]
    
    def new_node(self, parent_gen:int, parent_idx:int, action:np.ndarray, state:np.ndarray, obs:np.ndarray, reward:float, done:bool, terminal_reward=0.):
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
        self.gens[parent_gen+1].append(new_node)
        return new_node
    
    def new_traj(self, parent_gen:int, parent_idx:int, trans_dict:dict):
        pass

    def backup(self, mode="mean"):
        '''
            args:
                `mode`: "max" or "mean".
        '''
        if mode not in ["max", "mean"]:
            raise(ValueError("mode must be \"max\" or \"mean\"."))
        for g in range(self.max_gen-1,-1,-1):
            for node in self.gens[g]:
                if node.is_leaf:
                    if not node.done:
                        raise(ValueError("leaves must be terminal states."))
                    node.V_target = node.terminal_reward
                    node.Q_target = node.reward + self.gamma*node.terminal_reward
                else:
                    next_V_targets = np.array([child.V_target for child in node.children], dtype=np.float32)
                    next_Q_targets = np.array([child.Q_target for child in node.children], dtype=np.float32)
                    rewards = np.array([child.reward for child in node.children], dtype=np.float32)
                    if mode=="max":
                        node.V_target = np.max(rewards + self.gamma*next_V_targets)
                        node.Q_target = node.reward + self.gamma*np.max(next_Q_targets)
                    elif mode=="mean":
                        node.V_target = np.mean(rewards + self.gamma*next_V_targets)
                        node.Q_target = node.reward + self.gamma*np.mean(next_Q_targets)
        return
    
    def update_regret_mc(self):
        self.root.update_regret_mc()

    def update_regret_td(self, critic):
        self.root.update_regret_td(critic)

