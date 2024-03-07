'''
    torch version of `propagator`
'''
import torch
from env.propagator import Propagator
from env.dynamic import matrix

class PropagatorT(Propagator):
    '''
        torch version of `Propagator`
    '''
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int, device:str) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    def getObss(self, states:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def getRewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def getTruncatedRewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def getNextStates(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def getDones(self, states:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def propagate(self, states:torch.Tensor, actions:torch.Tensor):
        '''
            returns: `next_states`, `rewards`, `dones`, `next_obss`
        '''
        next_states = self.getNextStates(states, actions)
        rewards = self.getRewards(states, actions)
        dones = self.getDones(states)
        next_obss = self.getObss(next_states)
        return next_states, rewards, dones, next_obss
    
    def randomInitStates(self, num_states:int) -> torch.Tensor:
        raise NotImplementedError
    
    def obssNormalize(self, obss:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def seqOptTgt(self, states:torch.Tensor, agent, horizon:int, optStep=True):
        '''
            sequence optimization target function, total reward to be maximized, 
            available when reward is contiunous function of state and action.\n
            Only for `normalDistAgent`, to be overloaded.\n
            args:
                `states`: initial states.
                `agent`: see `rlAgent`.
                `horizon`: sequence length for optimization.
            return:
                `reward_total`: target tensor to be maximize.
        '''
        batch_size = states.shape[0]
        if self.device!=agent.device:
            raise ValueError("device mismatch")
        reward_seq = []
        for i in range(horizon):
            obss = self.getObss(states)
            outputs, actions = agent.act(obss)
            nominal_actions = outputs[:,:3]
            '''
                NOTE: `actions` are sampled from distribution, have no grad. 
                `nominal_actions` are means of normal distribution (see `normalDistAgent`), have grad. 
                To be overloaded.
            '''
            states, rewards, dones, _ = self.propagate(states, nominal_actions)
            reward_seq.append(rewards.sum()/batch_size)
            undone_idx = torch.where(dones==False)[0]
            if undone_idx.shape[0]==0:
                break
            states = states[undone_idx]
        reward_total = torch.mean(torch.stack(reward_seq))
        if optStep:
            agent.actor_opt.zero_grad()
            loss = -reward_total
            loss.backward()
            agent.actor_opt.step()
        return reward_total
    
class linearSystemT(PropagatorT):
    def __init__(self,
                 state_mat:torch.Tensor, 
                 obs_mat:torch.Tensor,
                 control_mat:torch.Tensor,
                 device:str) -> None:
        if state_mat.shape[0]!=state_mat.shape[1]:
            raise ValueError("state_mat must be a square matrix")
        if obs_mat.shape[1]!=obs_mat.shape[0]:
            raise ValueError("shape incompatible: `state_mat` and `obs_dim`")
        if control_mat.shape[0]!=state_mat.shape[0]:
            raise ValueError("shape incompatible: `state_mat` and `control_mat`")
        state_dim = state_mat.shape[0]
        obs_dim = obs_mat.shape[0]
        action_dim = control_mat.shape[1]
        super().__init__(state_dim, obs_dim, action_dim, device=device)
        self.state_mat = state_mat.to(device)
        self.obs_mat = obs_mat.to(device)
        self.control_mat = control_mat.to(device)

    def getObss(self, states:torch.Tensor) -> torch.Tensor:
        obs = states@self.obs_mat.T
        return obs

    def getRewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        raise(NotImplementedError)

    def getNextStates(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        next_states = states@self.state_mat.T + actions@self.control_mat.T
        return next_states

    def getDones(self, states:torch.Tensor) -> torch.Tensor:
        raise(NotImplementedError)
    
class motionSystemT(linearSystemT):
    def __init__(self, state_mat: torch.Tensor, device:str, max_dist=10.) -> None:
        state_dim = state_mat.shape[0]
        if state_dim%2!=0:
            raise ValueError("state_mat must be an even integer, representing position and velocity.")
        space_dim = state_dim//2
        obs_mat = torch.eye(state_dim)
        control_mat = torch.vstack((torch.zeros((space_dim,space_dim)),torch.eye(space_dim)))
        super().__init__(state_mat, obs_mat, control_mat, device=device)
        self.space_dim = space_dim
        
        self.max_dist = max_dist
        self.k=0.2

    def getRewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        rad, vel = self.norms(states)
        return (self.max_dist-rad-self.k*vel)/self.max_dist

    def getDones(self, states:torch.Tensor) -> torch.Tensor:
        rad, vel = self.norms(states)
        dones = rad>self.max_dist
        return dones
    
    def norms(self, states):
        return torch.norm(states[:, :self.space_dim], dim=1), torch.norm(states[:, self.space_dim:], dim=1)

    def randomInitStates(self, num_states: int) -> torch.Tensor:
        states = torch.zeros((num_states, self.state_dim), dtype=torch.float32)
        f1 = self.space_dim
        f2 = 10*self.space_dim
        dist1 = torch.distributions.Uniform(low=-self.max_dist/f1, high=self.max_dist/f1)
        dist2 = torch.distributions.Uniform(low=-self.max_dist/f2, high=self.max_dist/f2)
        states[:,:self.space_dim] = dist1.sample((num_states,self.space_dim))
        states[:,self.space_dim:] = dist2.sample((num_states,self.space_dim))
        return states
    

class CWPropagatorT(motionSystemT):
    def __init__(self, device:str, dt=1., orbit_rad=7e6, max_dist=5e3) -> None:
        state_mat = matrix.CW_TransMat(0, dt, orbit_rad)
        state_mat = torch.from_numpy(state_mat).float().to(device)
        super().__init__(state_mat, device=device, max_dist=max_dist)
        self.dt = dt
        self.orbit_rad = orbit_rad
        self.k = .01

    def getRewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        device = states.device
        iter_step = 10
        r0, v0 = states[:, :3], states[:, 3:]
        R = torch.zeros((iter_step, batch_size, 3),device=device)
        V = torch.zeros((iter_step, batch_size, 3),device=device)
        R[0,...] = r0
        V[0,...] = v0
        for i in range(1, iter_step):
            R[i,...] = R[i-1,...] + V[i-1,...]
            V[i,...] = V[i-1,...] + actions*self.dt
        rads = torch.linalg.norm(R, dim=2)
        mean_rads = torch.mean(rads, dim=0)
        return (self.max_dist-mean_rads)/self.max_dist

    def getNextStates(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        con_vec = matrix.CW_constConVecsT(0, self.dt, actions, self.orbit_rad)
        next_states = states@self.state_mat.T + con_vec
        return next_states
    
    def randomInitStates(self, num_states: int):
        states = torch.zeros((num_states, self.state_dim), dtype=torch.float32)
        f1 = self.space_dim
        f2 = 100*self.space_dim
        dist1 = torch.distributions.Uniform(low=-self.max_dist/f1, high=self.max_dist/f1)
        dist2 = torch.distributions.Uniform(low=-self.max_dist/f2, high=self.max_dist/f2)
        states[:,:self.space_dim] = dist1.sample((num_states,self.space_dim))
        states[:,self.space_dim:] = dist2.sample((num_states,self.space_dim))
        return states