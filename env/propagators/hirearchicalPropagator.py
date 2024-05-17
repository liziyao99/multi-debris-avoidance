import torch
from env.dynamic import matrix
from env.dynamic import cwutils
import typing

class H2Propagator:
    def __init__(self, state_dim:int, obs_dim:int, h1_action_dim:int, h2_action_dim:int, h1_step:int, h2_step:int, device="cpu") -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.h1_action_dim = h1_action_dim
        self.h2_action_dim = h2_action_dim
        self.h1_step = h1_step
        self.h2_step = h2_step
        self.device = device

    def getObss(self, states:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getH1Rewards(self, states:torch.Tensor, h1_actions:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getH2Rewards(self, states:torch.Tensor, h1_actions:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError

    def getTruncatedRewards(self, states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getNextStates(self, states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getDones(self, states:torch.Tensor, require_grad=False) -> typing.Tuple[torch.Tensor]:
        '''
            returns: done and terminal reward.
        '''
        raise NotImplementedError
    
    def propagate(self, states:torch.Tensor, h1_actions:torch.Tensor, actions:torch.Tensor, require_grad=False):
        '''
            returns: `next_states`, `next_obss`, `h1rewards`, `h2rewards`, `dones`, `terminal_rewards`
        '''
        next_states = self.getNextStates(states, actions, require_grad=require_grad)
        h1rewards = self.getH1Rewards(states, h1_actions, actions, require_grad=require_grad)
        h2rewards = self.getH2Rewards(states, h1_actions, actions, require_grad=require_grad)
        dones, terminal_rewards = self.getDones(next_states, require_grad=require_grad)
        next_obss = self.getObss(next_states, require_grad=require_grad)
        return next_states, next_obss, h1rewards, h2rewards, dones, terminal_rewards
    
    def randomInitStates(self, n:int) -> torch.Tensor:
        raise NotImplementedError
    
    def obssNormalize(self, obss:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            return obss
    
    def statesDecode(self, states:torch.Tensor, require_grad=False):
        raise NotImplementedError
    
    def statesEncode(self, datas:dict, require_grad=False):
        raise NotImplementedError
    

class H2CWDePropagator(H2Propagator):
    def __init__(self, n_debris, h1_step=60, h2_step=60, dt=1., orbit_rad=7e6, max_dist=5e3, safe_dist=5e2, device="cpu") -> None:
        state_dim = 6*(1+n_debris) + 7*n_debris + 2
        '''
            `state_dim = 6*(1+n_debris) + 7*n_debris + 2`.
            `6*(1+n_debris)`: states of primal and debris.
            `7*n_debris`: debris' forecast data.
            `2`: h1/h2 time steps.
        '''
        obs_dim = 6*(1+n_debris) + 7*n_debris + 2 # same
        h1_action_dim = 6 # target pos and vel
        h2_action_dim = 3 # thrust applied on debris
        super().__init__(state_dim, obs_dim, h1_action_dim, h2_action_dim, h1_step, h2_step, device)

        self.trans_mat = torch.from_numpy(matrix.CW_TransMat(0, dt, orbit_rad)).float().to(device)
        self.state_mat = torch.from_numpy(matrix.CW_StateMat(orbit_rad)).float().to(device)
        self.dt = dt
        self.orbit_rad = orbit_rad
        self.n_debris = n_debris
        self.max_dist = max_dist
        self.safe_dist = safe_dist

        self.h1r_maxfuel = self.dt*torch.norm(torch.tensor([0.06]*3)).item()

        self.h2r_pos_coef = 1.
        self.h2r_vel_coef = 1.
        self.h2r_pos_err_thrus = 10.

        self.fail_penalty_coef = -0.06
        self.err_penalty_coef = -1/max_dist

    def statesDecode(self, states:torch.Tensor, require_grad=False):
        '''
            return a dict containing copied data.
            "primal": state of primal, shape (batch_size, 1, 6).
            "debris": state of debris, shape (batch_size, n_debris, 6).
            "forecast_time": time of debris' forecast, shape (batch_size, n_debris, 1).
            "forecast_pos": position of debris' forecast, shape (batch_size, n_debris, 3).
            "forecast_vel": velocity of debris' forecast, shape (batch_size, n_debris, 3).
            "h1_step": h1 time step, shape (batch_size, 1).
            "h2_step": h2 time step, shape (batch_size, 1).
        '''
        with torch.set_grad_enabled(require_grad):
            batch_size = states.shape[0]
            primal_states = states[:, :6].reshape((batch_size, 1, 6))
            debris_states = states[:, 6:6*(1+self.n_debris)].reshape((batch_size, self.n_debris, 6))
            debris_forecast = states[:, 6*(1+self.n_debris):6*(1+self.n_debris)+7*self.n_debris]
            forecast_time = debris_forecast[:, :self.n_debris].reshape(batch_size,self.n_debris,1)
            forecast_pos = debris_forecast[:, self.n_debris:4*self.n_debris].reshape(batch_size,self.n_debris,3)
            forecast_vel = debris_forecast[:, 4*self.n_debris:].reshape(batch_size,self.n_debris,3)
            h1_step = states[:, -2].reshape((batch_size, 1))
            h2_step = states[:, -1].reshape((batch_size, 1))
            datas = {
                "primal": primal_states.clone(),
                "debris": debris_states.clone(),
                "forecast_time": forecast_time.clone(),
                "forecast_pos": forecast_pos.clone(),
                "forecast_vel": forecast_vel.clone(),
                "h1_step": h1_step.clone(),
                "h2_step": h2_step.clone(),
            }
            return datas
    
    def statesEncode(self, datas:dict, require_grad=False):
        '''
            `datas`'s keys and items:
            "primal": state of primal, shape (batch_size, 1, 6).
            "debris": state of debris, shape (batch_size, n_debris, 6).
            "forecast_time": time of debris' forecast, shape (batch_size, n_debris, 1).
            "forecast_pos": position of debris' forecast, shape (batch_size, n_debris, 3).
            "forecast_vel": velocity of debris' forecast, shape (batch_size, n_debris, 3).
            "h1_step": h1 time step, shape (batch_size, 1).
            "h2_step": h2 time step, shape (batch_size, 1).
        '''
        with torch.set_grad_enabled(require_grad):
            primal_states = datas["primal"]
            debris_states = datas["debris"]
            forecast_time = datas["forecast_time"]
            forecast_pos = datas["forecast_pos"]
            forecast_vel = datas["forecast_vel"]
            h1_step = datas["h1_step"]
            h2_step = datas["h2_step"]
            states = torch.cat((primal_states.reshape(-1, 6), 
                                debris_states.reshape(-1, 6*self.n_debris),
                                forecast_time.reshape(-1, self.n_debris), 
                                forecast_pos.reshape(-1, 3*self.n_debris),
                                forecast_vel.reshape(-1, 3*self.n_debris),
                                h1_step.reshape(-1, 1),
                                h2_step.reshape(-1, 1)
                            ), dim=1)
            return states
    
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
        forecast_time_dist = torch.distributions.Uniform(low=max_time/3, high=2*max_time/3)

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
            "forecast_vel": forecast_vel,
            "h1_step": torch.zeros((num_states, 1)).to(self.device),
            "h2_step": torch.zeros((num_states, 1)).to(self.device)
        }
        states = self.statesEncode(states_dict)
        return states.to(self.device)
    
    def getObss(self, states:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            obss = states.clone()
            return self.obssNormalize(obss, require_grad)
    
    def getH1Rewards(self, states: torch.Tensor, h1_actions:torch.Tensor, actions: torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            rewards = (self.h1r_maxfuel-torch.norm(actions, dim=-1))/(self.h1r_maxfuel*self.h2_step)
            return rewards
        
    def getH2Rewards(self, states: torch.Tensor, h1_actions: torch.Tensor, actions: torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            h1_actions = h1_actions.detach()
            primal_pos = states[:,  :3]
            primal_vel = states[:, 3:6]
            target_pos = h1_actions[:,  :3]
            target_vel = h1_actions[:, 3:6]
            pos_err = self.h2r_pos_coef * torch.norm(primal_pos-target_pos, dim=-1)
            vel_err = self.h2r_vel_coef * torch.norm(primal_vel-target_vel, dim=-1)
            rewards = -torch.where(pos_err<self.h2r_pos_err_thrus, pos_err, pos_err+vel_err)
            return rewards
        
    def getNextStates(self, states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            batch_size = states.shape[0]
            decoded = self.statesDecode(states, require_grad)
            object_states = torch.concatenate((decoded["primal"], decoded["debris"]), dim=1) # shape: (batch_size, 1+n_debris, 6)
            next_object_states = object_states@self.trans_mat.T
            con_vec = matrix.CW_constConVecsT(0, self.dt, actions, self.orbit_rad).to(self.device)
            next_object_states[:,0,:] = next_object_states[:,0,:]+con_vec
            decoded["primal"] = next_object_states[:,0,:]
            decoded["debris"] = next_object_states[:,1:,:]
            decoded["forecast_time"] = decoded["forecast_time"]-self.dt
            decoded["h2_step"] = decoded["h2_step"]+1
            decoded["h1_step"] = torch.where(decoded["h2_step"]>=self.h2_step, decoded["h1_step"]+1, decoded["h1_step"])
            decoded["h2_step"] = torch.where(decoded["h2_step"]>=self.h2_step, 0, decoded["h2_step"])
            next_states = self.statesEncode(decoded, require_grad)
            return next_states
    
    def getDones(self, states:torch.Tensor, require_grad=False) -> typing.Tuple[torch.Tensor]:
        with torch.set_grad_enabled(require_grad):
            time_out = states[:,-2]>=self.h1_step
            d2o, d2d = self.distances(states, require_grad)
            out_dist = (d2o>self.max_dist).flatten()
            collision = torch.sum(d2d<self.safe_dist, dim=1).to(torch.bool)
            dones = time_out+out_dist+collision
            terminal_rewards = torch.zeros_like(dones, dtype=torch.float32)
            fail_penalty = self.fail_penalty_coef*self.h1_step*self.h2_step*self.dt
            terminal_rewards = torch.where(out_dist+collision, fail_penalty, terminal_rewards)
            err_penalty = self.err_penalty_coef*d2o.flatten()
            terminal_rewards = torch.where(time_out, err_penalty, terminal_rewards)
            return dones, terminal_rewards
    
    def distances(self, states, require_grad=False):
        '''
            returns:
                `d2o`: primal's distance to origin.
                `d2d`: primal's distance to debris.
        '''
        with torch.set_grad_enabled(require_grad):
            decoded = self.statesDecode(states, require_grad)
            primal_pos = decoded["primal"][:, :, :3]
            debris_pos = decoded["debris"][:, :, :3]
            d2o = torch.linalg.norm(primal_pos, dim=2) # distance to origin
            d2d = torch.linalg.norm(debris_pos-primal_pos, dim=2) # distance to debris
            return d2o, d2d