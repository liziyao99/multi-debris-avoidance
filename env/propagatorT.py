'''
    torch version of `propagator`
'''
import torch
import torch.nn.functional as F
from env.propagator import Propagator
from env.dynamic import matrix
from utils import lineProj

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
    
    def seqOpt(self, states:torch.Tensor, agent, horizon:int, optStep=True, smooth=False, smooth_k=0.01):
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
        delta_u_seq = []
        for i in range(horizon):
            if i>0:
                last_nominal_actions = nominal_actions[undone_idx]
            obss = self.getObss(states)
            nominal_actions = agent.nominal_act(obss)
            states, rewards, dones, _ = self.propagate(states, nominal_actions)
            reward_seq.append(rewards.sum()/batch_size)
            if smooth and i>0:
                delta_u_seq.append(smooth_k*F.mse_loss(nominal_actions, last_nominal_actions))
            undone_idx = torch.where(dones==False)[0]
            if undone_idx.shape[0]==0:
                break
            states = states[undone_idx]
        reward_total = torch.mean(torch.stack(reward_seq))
        delta_u_total = torch.mean(torch.stack(delta_u_seq)) if smooth else 0.
        loss = -reward_total + delta_u_total
        if optStep:
            agent.actor_opt.zero_grad()
            loss.backward()
            agent.actor_opt.step()
        return loss
    
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
        iter_step = 1
        r0, v0 = states[:, :3], states[:, 3:]
        R = torch.zeros((iter_step, batch_size, 3),device=device)
        V = torch.zeros((iter_step, batch_size, 3),device=device)
        R[0,...] = r0
        V[0,...] = v0
        for i in range(1, iter_step):
            R[i,...] = R[i-1,...] + V[i-1,...]*self.dt
            V[i,...] = V[i-1,...] + actions*self.dt
        rads = torch.linalg.norm(R, dim=2)
        mean_rads = torch.mean(rads, dim=0)
        return (self.max_dist-mean_rads)/self.max_dist - self.k*torch.linalg.norm(actions, dim=1)

    def getNextStates(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        con_vec = matrix.CW_constConVecsT(0, self.dt, actions, self.orbit_rad)
        next_states = states@self.state_mat.T + con_vec
        return next_states
    
    def randomInitStates(self, num_states: int):
        states = torch.zeros((num_states, self.state_dim), dtype=torch.float32)
        f1 = self.space_dim
        f2 = 1000*self.space_dim
        dist1 = torch.distributions.Uniform(low=-self.max_dist/f1, high=self.max_dist/f1)
        dist2 = torch.distributions.Uniform(low=-self.max_dist/f2, high=self.max_dist/f2)
        states[:,:self.space_dim] = dist1.sample((num_states,self.space_dim))
        states[:,self.space_dim:] = dist2.sample((num_states,self.space_dim))
        return states
    
    def obssNormalize(self, obss: torch.Tensor) -> torch.Tensor:
        f1 = self.max_dist
        f2 = self.max_dist/100
        obss_n = obss.clone()
        obss_n[:,:self.space_dim] /= f1
        obss_n[:,self.space_dim:] /= f2
        return obss_n
    
    def getObss(self, states: torch.Tensor) -> torch.Tensor:
        obss = super().getObss(states)
        return self.obssNormalize(obss)
    

from env.dynamic import cwutils
class CWDebrisPropagatorT(PropagatorT):
    def __init__(self, device:str, n_debris, dt=1., orbit_rad=7e6, max_dist=5e3, safe_dist=5e2) -> None:
        state_dim = 6*(1+n_debris)+7*n_debris # states of primal and debris, and debris' forecast data
        obs_dim = 6*(1+n_debris)+7*n_debris # same
        action_dim = 3 # thrust applied on primal
        super().__init__(state_dim, obs_dim, action_dim, device)

        state_mat = matrix.CW_TransMat(0, dt, orbit_rad)
        self.state_mat = torch.from_numpy(state_mat).float().to(device)
        self.dt = dt
        self.orbit_rad = orbit_rad
        self.n_debris = n_debris
        self.max_dist = max_dist
        self.safe_dist = safe_dist

        self.k = .01
        '''
            parameter of control to reward
        '''

    def statesDecode(self, states:torch.Tensor):
        '''
            return a dict containing copied data.
            "primal": state of primal, shape (batch_size, 1, 6).
            "debris": state of debris, shape (batch_size, n_debris, 6).
            "forecast_time": time of debris' forecast, shape (batch_size, n_debris, 1).
            "forecast_pos": position of debris' forecast, shape (batch_size, n_debris, 3).
            "forecast_vel": velocity of debris' forecast, shape (batch_size, n_debris, 3).
        '''
        batch_size = states.shape[0]
        primal_states = states[:, :6].reshape((batch_size, 1, 6))
        debris_states = states[:, 6:6*(1+self.n_debris)].reshape((batch_size, self.n_debris, 6))
        debris_forecast = states[:, 6*(1+self.n_debris):]
        forecast_time = debris_forecast[:, :self.n_debris].reshape(batch_size,self.n_debris,1)
        forecast_pos = debris_forecast[:, self.n_debris:4*self.n_debris].reshape(batch_size,self.n_debris,3)
        forecast_vel = debris_forecast[:, 4*self.n_debris:].reshape(batch_size,self.n_debris,3)
        datas = {
            "primal": primal_states.clone(),
            "debris": debris_states.clone(),
            "forecast_time": forecast_time.clone(),
            "forecast_pos": forecast_pos.clone(),
            "forecast_vel": forecast_vel.clone(),
        }
        return datas
    
    def statesEncode(self, datas:dict):
        '''
            `datas`'s keys and items:
            "primal": state of primal, shape (batch_size, 1, 6).
            "debris": state of debris, shape (batch_size, n_debris, 6).
            "forecast_time": time of debris' forecast, shape (batch_size, n_debris, 1).
            "forecast_pos": position of debris' forecast, shape (batch_size, n_debris, 3).
            "forecast_vel": velocity of debris' forecast, shape (batch_size, n_debris, 3).
        '''
        primal_states = datas["primal"]
        debris_states = datas["debris"]
        forecast_time = datas["forecast_time"]
        forecast_pos = datas["forecast_pos"]
        forecast_vel = datas["forecast_vel"]
        states = torch.concatenate((primal_states.reshape(-1, 6), 
                                 debris_states.reshape(-1, 6*self.n_debris),
                                 forecast_time.reshape(-1, self.n_debris), 
                                 forecast_pos.reshape(-1, 3*self.n_debris),
                                 forecast_vel.reshape(-1, 3*self.n_debris)
                                ), dim=1)
        return states

    def getObss(self, states:torch.Tensor) -> torch.Tensor:
        obss = states.clone()
        return self.obssNormalize(obss)
    
    def getRewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        d2o, d2d, d2p = self.distances(states)
        nd2p = d2p/self.safe_dist
        decoded = self.statesDecode(states)
        forecast_time = decoded["forecast_time"].squeeze(dim=-1)
        approaching = forecast_time>0
        n_approaching = torch.sum(approaching, dim=1)

        primal_reward = (self.max_dist-d2o.flatten())/self.max_dist - self.k*torch.linalg.norm(actions, dim=1)
        debris_reward_each = torch.log(nd2p)/nd2p
        debris_reward_each = debris_reward_each*approaching
        debris_reward = torch.sum(debris_reward_each, dim=1)
        rewards = (primal_reward+debris_reward)/(1+n_approaching)
        return rewards
    
    def getTruncatedRewards(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def getNextStates(self, states:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        decoded = self.statesDecode(states)
        object_states = torch.concatenate((decoded["primal"], decoded["debris"]), dim=1) # shape: (batch_size, 1+n_debris, 6)
        next_object_states = object_states@self.state_mat.T
        con_vec = matrix.CW_constConVecsT(0, self.dt, actions, self.orbit_rad)
        next_object_states[:,0,:] += con_vec
        decoded["primal"] = next_object_states[:,0,:]
        decoded["debris"] = next_object_states[:,1:,:]
        decoded["forecast_time"] -= self.dt
        next_states = self.statesEncode(decoded)
        return next_states
    
    def getDones(self, states:torch.Tensor) -> torch.Tensor:
        d2o, d2d, _ = self.distances(states)
        out_dist = (d2o>self.max_dist).flatten()
        collision = torch.sum(d2d<self.safe_dist, dim=1).to(torch.bool)
        dones = out_dist+collision
        return dones
    
    def randomInitStates(self, num_states:int) -> torch.Tensor:
        space_dim = 3
        f1 = 10*space_dim
        f2 = 1000*space_dim
        f3 = 10*space_dim
        max_time = 3600.

        primal_pos_dist = torch.distributions.Uniform(low=-self.max_dist/f1, high=self.max_dist/f1)
        primal_vel_dist = torch.distributions.Uniform(low=-self.max_dist/f2, high=self.max_dist/f2)
        forecast_pos_dist = torch.distributions.Uniform(low=-self.max_dist/f1, high=self.max_dist/f1)
        forecast_vel_dist = torch.distributions.Uniform(low=-self.max_dist/f3, high=self.max_dist/f3)
        forecast_time_dist = torch.distributions.Uniform(low=max_time/2, high=max_time)

        primal_pos = primal_pos_dist.sample((num_states, 1, space_dim)).to(self.device)
        primal_vel = primal_vel_dist.sample((num_states, 1, space_dim)).to(self.device)
        forecast_pos = forecast_pos_dist.sample((num_states, self.n_debris, space_dim)).to(self.device)
        forecast_vel = forecast_vel_dist.sample((num_states, self.n_debris, space_dim)).to(self.device)
        forecast_time = forecast_time_dist.sample((num_states, self.n_debris, 1)).to(self.device)

        primal_states = torch.concatenate((primal_pos, primal_vel), dim=2)
        forecast_states = torch.concatenate((forecast_pos, forecast_vel), dim=2)
        debris_states = cwutils.CW_tInv_batch_torch(a = torch.tensor([self.orbit_rad]*(num_states*self.n_debris)), 
                                                    forecast_states = forecast_states.reshape((num_states*self.n_debris,space_dim*2)), 
                                                    t2c = forecast_time.flatten(),
                                                    device = self.device)
        debris_states = debris_states.reshape((num_states, self.n_debris, space_dim*2))
        states_dict = {
            "primal": primal_states,
            "debris": debris_states,
            "forecast_time": forecast_time,
            "forecast_pos": forecast_pos,
            "forecast_vel": forecast_vel
        }
        states = self.statesEncode(states_dict)
        return states
    
    def obssNormalize(self, obss:torch.Tensor) -> torch.Tensor:
        f1 = self.max_dist
        f2 = self.max_dist/100
        states = obss.clone()
        decoded = self.statesDecode(states) # obss is states
        primal_pos = decoded["primal"][:, :, :3]
        debris_pos = decoded["debris"][:, :, :3]
        primal_vel = decoded["primal"][:, :, 3:]
        debris_vel = decoded["debris"][:, :, 3:]
        primal_pos_n = primal_pos/f1
        debris_pos_n = debris_pos/f1
        primal_vel_n = primal_vel/f2
        debris_vel_n = debris_vel/f2
        decoded["primal"][:, :, :3] = primal_pos_n
        decoded["debris"][:, :, :3] = debris_pos_n
        decoded["primal"][:, :, 3:] = primal_vel_n
        decoded["debris"][:, :, 3:] = debris_vel_n
        decoded["forecast_pos"] /= f1
        decoded["forecast_vel"] /= f2
        states_n = self.statesEncode(decoded)
        obss_n = states_n.clone()
        return obss_n
    
    def distances(self, states):
        '''
            returns:
                `d2o`: primal's distance to origin.
                `d2d`: primal's distance to debris.
                `d2p`: primal's distance to debris' forecast line.
        '''
        decoded = self.statesDecode(states)
        primal_pos = decoded["primal"][:, :, :3]
        debris_pos = decoded["debris"][:, :, :3]
        forecast_pos = decoded["forecast_pos"]
        forecast_vel = decoded["forecast_vel"]
        d2o = torch.linalg.norm(primal_pos, dim=2) # distance to origin
        d2d = torch.linalg.norm(debris_pos-primal_pos, dim=2) # distance to debris
        primal_proj, primal_orth = lineProj(primal_pos, forecast_pos, forecast_vel)
        d2p = torch.linalg.norm(primal_orth, dim=2) # distance to debris' forecast
        return d2o, d2d, d2p