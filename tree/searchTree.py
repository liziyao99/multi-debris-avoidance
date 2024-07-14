import torch
import numpy as np
import anytree
import typing

class stateNodeV2(anytree.Node):
    def __init__(self, 
                 state_dim:int, 
                 obs_dim:int,
                 action_dim:int, 
                 name=None, parent=None, children=None, **kwargs):
        super().__init__(name, parent, children, **kwargs)
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.state = torch.zeros((1, state_dim), dtype=torch.float32) if state_dim is not None else None
        self.action = torch.zeros((1, action_dim), dtype=torch.float32) if action_dim is not None else None
        '''
            action that make transition from parent to this node.
            Notice that root node has no action.
        '''
        self.obs = torch.zeros((1, obs_dim), dtype=torch.float32) if obs_dim is not None else None

        self.done = False
        self.reward = torch.zeros(1, dtype=torch.float32)
        '''
            reward recived from parent's state and self.action.
            Notice that root node has no reward.
        '''
        self.terminal_reward = torch.zeros(1, dtype=torch.float32)
        '''
            reward recived from this node if is terminal.
        '''
        self.V_critic = None
        '''
            value given by critic for self.state.
        '''
        self.V_target = None
        '''
            value target for self.state.
        '''
        self.Q_critic = None
        '''
            Q value given by critic for parent's state and self.action.
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

    @classmethod
    def from_data(cls, state:torch.Tensor, obs:torch.Tensor, action:torch.Tensor, reward, done, name=None, parent=None, children=None, 
                  device=None, **kwargs):
        node = cls(state_dim=None,
                   obs_dim=None,
                   action_dim=None,
                   name=name, parent=parent, children=children, **kwargs)
        node.state = state
        node.action = action
        node.obs = obs
        node.reward = node.reward.to(device)
        node.reward[...] = reward
        node.terminal_reward = node.terminal_reward.to(device)
        node.done = done
        return node
    
class searchTree:
    def __init__(self, 
                 max_gen:int,
                 state_dim:int, 
                 obs_dim:int, 
                 action_dim:int, 
                 gamma=0.95, 
                 select_explore_eps=0.2, 
                 device=None) -> None:
        self.max_gen = max_gen
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.select_explore_eps = select_explore_eps
        self.device = device

        self.root = stateNodeV2(state_dim, obs_dim, action_dim, name="root")
        self.nodes = [self.root]
        self.gens = [[] for _ in range(max_gen)]
        self.gens[0].append(self.root) # Note: root is gen0, different from `undoTree`.

        self._backup_modes = ("max", "mean")
        self._select_modes = ("uniform", "QMax", "value")

    @classmethod
    def from_root(cls, root:stateNodeV2, max_gen:int, gamma=0.95, select_explore_eps=0.2, device=None):
        tree = cls(max_gen=max_gen, 
                   state_dim=root.state_dim,
                   obs_dim=root.obs_dim,
                   action_dim=root.action_dim,
                   gamma=gamma,
                   select_explore_eps=select_explore_eps,
                   device=device)
        tree.root = root
        tree.nodes = [root]
        tree.gens[0] = [root]
        return tree
    
    @classmethod
    def from_data(cls, state, obs, max_gen:int, gamma=0.95, select_explore_eps=0.2, device=None):
        root = stateNodeV2.from_data(state=state, obs=obs, action=None, reward=0., done=False, name="root", device=device)
        tree = cls.from_root(root, max_gen, gamma, select_explore_eps, device=device)
        return tree
    
    def backup(self, mode="mean"):
        '''
            args:
                `mode`: {self._backup_modes}.
        '''
        if mode not in self._backup_modes:
            raise(ValueError(f"mode must be in {self._backup_modes}."))
        for g in range(self.max_gen-1,-1,-1):
            for node in self.gens[g]:
                if node.is_leaf:
                    if not node.done:
                        if node.V_critic is None:
                            raise(ValueError("undone leaf node with no `V_critic`."))
                        node.V_target = node.V_critic
                    else:
                        node.V_target = node.terminal_reward
                    if not node.is_root:
                        if not node.done:
                            node.Q_target = node.reward + self.gamma*node.V_critic
                        else:
                            node.Q_target = node.reward + self.gamma*node.terminal_reward
                else:
                    next_V_targets = torch.cat([child.V_target for child in node.children])
                    next_Q_targets = torch.cat([child.Q_target for child in node.children])
                    rewards = torch.cat([child.reward for child in node.children])
                    if mode=="max":
                        node.V_target = torch.max(rewards + self.gamma*next_V_targets)
                        if not node.is_root:
                            node.Q_target = node.reward + self.gamma*torch.max(next_Q_targets)
                    elif mode=="mean":
                        node.V_target = torch.mean(rewards + self.gamma*next_V_targets)
                        if not node.is_root:
                            node.Q_target = node.reward + self.gamma*torch.mean(next_Q_targets)
                node.V_target = node.V_target.reshape((1,1))
                if not node.is_root:
                    node.Q_target = node.Q_target.reshape((1,1))
        return
    
    def select(self, size, mode="uniform") -> typing.List[stateNodeV2]:
        if mode not in self._select_modes:
            raise(ValueError(f"mode must be in {self._select_modes}."))
        nodes = [node for node in self.nodes if (not node.done and node.depth<self.max_gen-1)]
        if mode=="uniform":
            selected = np.random.choice(nodes, size)
        elif mode=="QMax":
            q = [node.Q_target for node in nodes]
            q = torch.cat(q)
            idx = q.argmax()
            selected = [nodes[idx]]*size
        elif mode=="value":
            v = [node.V_target for node in nodes]
            v = torch.cat(v).flatten()
            dist = torch.distributions.Categorical(logits=v)
            idx = dist.sample((size,))
            selected = [nodes[_i] for _i in idx]
        return selected
    
    def new_node(self, state, obs, action, reward, done, V_critic, parent:stateNodeV2, check_parent=False):
        if check_parent and (parent not in self.nodes):
            raise(ValueError("parent not in tree."))
        node = stateNodeV2.from_data(state, obs, action, reward, done, parent=parent, device=self.device)
        node.V_critic = V_critic
        self.nodes.append(node)
        self.gens[parent.depth+1].append(node)
        return node

    def extend(self, states, obss, actions, rewards, dones, V_critics, parents:typing.List[stateNodeV2], check_parent=False):
        for i in range(len(parents)):
            self.new_node(states[i], obss[i], actions[i], rewards[i], dones[i], V_critics[i], parents[i], check_parent=check_parent)

    def best_action(self) -> torch.Tensor:
        '''
            return the best action in current state.
        '''
        Q = [node.Q_target.item() for node in self.root.children]
        best_idx = np.argmax(Q)
        return self.root.children[best_idx].action