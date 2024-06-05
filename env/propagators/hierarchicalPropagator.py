import torch
import numpy as np
import typing
from rich.progress import Progress

from env.dynamic import matrix
from env.dynamic import cwutils
import utils
import data.dicts as D

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
    

class H2CWDePropagator0(H2Propagator):
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
        self.h1_trans_mat = torch.from_numpy(matrix.CW_TransMat(0, dt*self.h2_step, orbit_rad)).float().to(device)
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

        self.fail_penalty_coef = -2.
        self.err_penalty_coef = -2/max_dist

        self.obss_normalize_scale = self._init_ons()

    def _init_ons(self) -> torch.Tensor:
        obss_normalize_scale = torch.ones((1,self.obs_dim)).float().to(self.device)
        decoded = self.statesDecode(obss_normalize_scale)
        decoded["primal"][...] = self.max_dist
        decoded["debris"][...] = self.max_dist
        decoded["forecast_pos"][...] = self.max_dist
        decoded["forecast_vel"][...] = self.max_dist
        decoded["forecast_time"][...] = self.h1_step*self.h2_step
        decoded["h1_step"][...] = self.h1_step
        decoded["h2_step"][...] = self.h2_step
        obss_normalize_scale = self.statesEncode(decoded)
        return obss_normalize_scale

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

    def obssNormalize(self, obss:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            decoded = self.statesDecode(obss, require_grad)
            max_obs_dist = self.max_dist*10
            decoded["debris"] = torch.clamp(decoded["debris"], -max_obs_dist, max_obs_dist) 
            # debris_pos, NOTE: unhashable type: 'slice'
            obss = self.statesEncode(decoded, require_grad)
            return obss/self.obss_normalize_scale
    
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
            fail_penalty = self.fail_penalty_coef*self.h1_step
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
        
class H2CWDePropagator(H2CWDePropagator0):
    def __init__(self, n_debris, h1_step=60, h2_step=60, dt=1, orbit_rad=7000000, max_dist=5000, safe_dist=500, device="cpu") -> None:
        super().__init__(n_debris, h1_step, h2_step, dt, orbit_rad, max_dist, safe_dist, device)
    
    def getH1Rewards(self, states: torch.Tensor, h1_actions:torch.Tensor, actions: torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):

            decoded = self.statesDecode(states)
            is_approaching = decoded["forecast_time"]>-10
            is_approaching = is_approaching.squeeze(-1)
            d2o, d2d, d2p = self.distances(states, require_grad)

            d2o = d2o/self.max_dist
            d2o = d2o.squeeze(-1)
            d2o_scale= 1.
            d2o_rewards = -d2o_scale*d2o # [-inf, 0]

            d2d = d2d/self.safe_dist
            d2d_scale = 2.
            d2d_rewards = utils.smoothStepFunc(d2d, loc=1, k=30, scale=d2d_scale) # [0, 2]
            d2d_rewards = torch.min(d2d_rewards, dim=1)[0]

            d2p = d2p/self.safe_dist
            d2p_scale = 2.
            d2p_rewards = utils.smoothStepFunc(d2p, loc=1, k=30, scale=d2p_scale) # [0, 2]
            d2p_rewards = torch.where(is_approaching, d2p_rewards, d2p_scale)
            d2p_rewards = torch.min(d2p_rewards, dim=1)[0]

            fuel_rewards = (self.h1r_maxfuel-torch.norm(actions, dim=-1))/self.h1r_maxfuel # [0, 1]

            rewards = (fuel_rewards+d2o_rewards+d2d_rewards+d2p_rewards)/self.h2_step
            return rewards
        
    def getDones(self, states:torch.Tensor, require_grad=False) -> typing.Tuple[torch.Tensor]:
        with torch.set_grad_enabled(require_grad):
            time_out = states[:,-2]>=self.h1_step
            d2o, d2d, _ = self.distances(states, require_grad)
            out_dist = (d2o>self.max_dist).flatten()
            collision = torch.sum(d2d<self.safe_dist, dim=1).to(torch.bool)
            dones = time_out
            terminal_rewards = torch.zeros_like(dones, dtype=torch.float32)
            return dones, terminal_rewards
        
    def distances(self, states, require_grad=False):
        '''
            returns:
                `d2o`: primal's distance to origin.
                `d2d`: primal's distance to debris.
                `d2p`: primal's distance to debris' forecast line.
        '''
        with torch.set_grad_enabled(require_grad):
            decoded = self.statesDecode(states, require_grad)
            primal_pos = decoded["primal"][:, :, :3]
            debris_pos = decoded["debris"][:, :, :3]
            forecast_pos = decoded["forecast_pos"]
            forecast_vel = decoded["forecast_vel"]
            d2o = torch.linalg.norm(primal_pos, dim=2) # distance to origin
            d2d = torch.linalg.norm(debris_pos-primal_pos, dim=2) # distance to debris
            primal_proj, primal_orth = utils.lineProj(primal_pos, forecast_pos, forecast_vel)
            d2p = torch.linalg.norm(primal_orth, dim=2) # distance to debris' forecast
            return d2o, d2d, d2p
        
    def episodeTrack(self, tracker, planner=None, init_states:torch.Tensor=None, targets:torch.Tensor=None, horizon:int=None, batch_size=256) -> torch.Tensor:
        '''
            args:
                `init_states`: shape (batch_size,state_dim)
                `targets`: shape (batch_size,state_dim)
                `tracker`: see `trackNet`.
                `horizon`: step to reach `targets`.
        '''
        init_states = self.randomInitStates(batch_size) if init_states is None else init_states
        horizon = self.h2_step if horizon is None else horizon
        err_thrus = 10.
        states = init_states
        obss = self.getObss(states, require_grad=True)
        if planner is not None:
            h1actions = planner(obss)
            targets = states[:,:6] + h1actions
        else:
            targets = self.randomInitStates(batch_size)[:,:6] if targets is None else targets
            targets = targets*5
        targets = targets.detach()
        actions_list = []
        for i in range(horizon):
            primals = obss[:,:6]
            tracker_input = torch.hstack((primals,targets))
            actions = tracker(tracker_input)
            states, obss, _, _, dones, _ = self.propagate(states, targets, actions, require_grad=True)
            actions_list.append(actions)
        err_loss = torch.norm(states[:,:6]-targets, dim=1)
        actions_list = torch.stack(actions_list)
        act_loss = torch.sum(torch.norm(actions_list, dim=-1), dim=0)
        loss = torch.where(err_loss<err_thrus, err_loss+act_loss, err_loss)
        loss = loss.mean()
        return loss
    
class thrustCWPropagator(H2CWDePropagator):
    def __init__(self, n_debris, h1_step=60, h2_step=60, dt=1, orbit_rad=7000000, max_dist=5000, safe_dist=500, device="cpu") -> None:
        super().__init__(n_debris, h1_step, h2_step, dt, orbit_rad, max_dist, safe_dist, device)
        self.h1_action_dim = 3 # total thrust

    def getH1Rewards(self, states: torch.Tensor, h1_actions:torch.Tensor, actions: torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            decoded = self.statesDecode(states)
            is_approaching = decoded["forecast_time"]>-10
            is_approaching = is_approaching.squeeze(-1)
            d2o, d2d, d2p = self.distances(states, require_grad)

            d2o = d2o/self.max_dist
            d2o = d2o.squeeze(-1)
            d2o_scale= 1.
            d2o_rewards = -d2o_scale*d2o # [-inf, 0]

            d2d = d2d/self.safe_dist
            d2d_scale = 2.
            d2d_rewards = utils.smoothStepFunc(d2d, loc=1, k=30, scale=d2d_scale) # [0, 2]
            d2d_rewards = torch.min(d2d_rewards, dim=1)[0]

            d2p = d2p/self.safe_dist
            d2p_scale = 2.
            d2p_rewards = utils.smoothStepFunc(d2p, loc=1, k=30, scale=d2p_scale) # [0, 2]
            d2p_rewards = torch.where(is_approaching, d2p_rewards, d2p_scale)
            d2p_rewards = torch.min(d2p_rewards, dim=1)[0]

            fuel_rewards = (self.h1r_maxfuel-torch.norm(actions, dim=-1))/self.h1r_maxfuel # [0, 1]

            rewards = (fuel_rewards+d2o_rewards+d2d_rewards+d2p_rewards)/self.h2_step
            return rewards
        
    def getH2Rewards(self, states: torch.Tensor, h1_actions: torch.Tensor, actions: torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            dummy reward.
        '''
        with torch.set_grad_enabled(require_grad):
            return torch.zeros(states.shape[0], device=self.device)


class optCWPropagator(H2CWDePropagator):
    def __init__(self, n_debris, h1_step=60, h2_step=60, dt=1, orbit_rad=7000000, max_dist=5000, safe_dist=500, device="cpu") -> None:
        super().__init__(n_debris, h1_step, h2_step, dt, orbit_rad, max_dist, safe_dist, device)
        self.h1_action_dim = 3 # impulse

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

        primal_pos = torch.zeros((num_states, 1, space_dim)).to(self.device)
        primal_vel = torch.zeros((num_states, 1, space_dim)).to(self.device)
        # primal_pos = primal_pos_dist.sample((num_states, 1, space_dim)).to(self.device)
        # primal_vel = primal_vel_dist.sample((num_states, 1, space_dim)).to(self.device)
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

    def getH1Rewards(self, states: torch.Tensor, h1_actions: torch.Tensor, actions: torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            dummy reward.
        '''
        with torch.set_grad_enabled(require_grad):
            return torch.zeros(states.shape[0], device=self.device)
        
    def getH2Rewards(self, states: torch.Tensor, h1_actions: torch.Tensor, actions: torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            decoded = self.statesDecode(states)
            forecast_time = decoded["forecast_time"].squeeze(-1)
            is_approaching = forecast_time>-10
            is_approaching = is_approaching.squeeze(-1)
            d2o, d2d, d2p = self.distances(states, require_grad)

            d2d = d2d/self.safe_dist
            d2d_rewards = utils.penaltyFuncPosi(d2d) # [-inf, 0]
            d2d_rewards = torch.mean(d2d_rewards, dim=1)

            d2p = d2p/self.safe_dist
            total_time = self.h2_step*self.h1_step*self.dt
            time_scale = ((total_time-forecast_time)/total_time)**2
            d2p_rewards = utils.penaltyFuncPosi(d2p) # [-inf, 0]
            d2p_rewards = d2p_rewards#*time_scale
            d2p_rewards = torch.where(is_approaching, d2p_rewards, 0)
            d2p_rewards = torch.mean(d2p_rewards, dim=1)

            fuel_rewards = (self.h1r_maxfuel-torch.norm(actions, dim=-1))/self.h1r_maxfuel # [0, 1]

            rewards = (fuel_rewards+d2d_rewards+d2p_rewards)/(self.h2_step*self.h1_step)
            return rewards
            # return -d2o.flatten() # for test
        
    def thrustOpt(self, actor:torch.nn.Module, batch_size=128, horizon=None, sigma=0.005):
        horizon = self.h2_step*self.h1_step if horizon is None else horizon
        states = self.randomInitStates(batch_size)
        obss = self.getObss(states, require_grad=True)
        total_rewards = torch.zeros(1, device=self.device)
        trans_dict = D.init_transDictBatch(horizon, batch_size, 
                                           state_dim=self.state_dim,
                                           action_dim=self.h2_action_dim,
                                           obs_dim=self.obs_dim,
                                           struct="torch", device=self.device)
        for i in range(horizon):
            trans_dict["states"][i,...] = states[...]
            trans_dict["obss"][i,...] = obss[...]
            actions = actor(obss)
            noise = torch.randn_like(actions)*sigma
            actions = actions + noise
            trans_dict["actions"][i,...] = actions[...]
            states, obss, _, rewards, dones, _ = self.propagate(states, h1_actions=None, actions=actions, require_grad=True)
            trans_dict["rewards"][i,...] = rewards[...]
            trans_dict["next_states"][i,...] = states[...]
            trans_dict["next_obss"][i,...] = obss[...]
            if dones.all():
                break
            total_rewards = total_rewards + rewards.mean()
        return total_rewards, trans_dict
    
    def impulseOpt(self, actor:torch.nn.Module, batch_size=128, horizon=None, sigma=0.06, impulse_scale=None,
                   same_s0=False):
        horizon = self.h1_step if horizon is None else horizon
        if same_s0:
            s0 = self.randomInitStates(1)
            states = s0.repeat(batch_size,1)
        else:
            states = self.randomInitStates(batch_size)
        obss = self.getObss(states, require_grad=True)
        total_rewards = torch.zeros(1, device=self.device)
        trans_dict = D.init_transDictBatch(horizon, batch_size,
                                           state_dim=self.state_dim,
                                           action_dim=self.h1_action_dim,
                                           obs_dim=self.obs_dim,
                                           struct="torch", device=self.device)
        dummy_h2_actions = torch.zeros((batch_size,self.h2_action_dim), device=self.device)
        for i in range(horizon):
            h1_rewards = torch.zeros(batch_size, device=self.device)
            trans_dict["states"][i,...] = states[...]
            trans_dict["obss"][i,...] = obss[...]
            impulse = actor(obss)
            if sigma:
                noise = torch.randn_like(impulse)*sigma
                impulse = impulse + noise
            trans_dict["actions"][i,...] = impulse[...]
            decoded = self.statesDecode(states, require_grad=True)
            decoded["primal"][:,0,3:] = decoded["primal"][:,0,3:] + impulse
            states = self.statesEncode(decoded, require_grad=True)
            for j in range(self.h2_step):
                states, obss, _, rewards, dones, _ = self.propagate(states, 
                                                                    h1_actions=None, 
                                                                    actions=dummy_h2_actions, 
                                                                    require_grad=True)
                h1_rewards = h1_rewards + rewards
            impulse_n = impulse if impulse_scale is None else impulse/impulse_scale
            impulse_rewards = -torch.sum(impulse_n**2, dim=-1)
            h1_rewards = h1_rewards + impulse_rewards
            trans_dict["rewards"][i,...] = h1_rewards[...]
            total_rewards = total_rewards + h1_rewards.mean()
        return total_rewards, trans_dict
    

class impulsePropagator(optCWPropagator):
    def __init__(self, n_debris, h1_step=60, h2_step=60, dt=1, orbit_rad=7000000, max_dist=5000, safe_dist=500, device="cpu",
                 impulse_bound:float=None) -> None:
        super().__init__(n_debris, h1_step, h2_step, dt, orbit_rad, max_dist, safe_dist, device)
        self.impulse_bound = impulse_bound

    def randomInitStates(self, num_states:int, seed=None) -> torch.Tensor:
        space_dim = 3
        f1 = 10*space_dim
        f2 = 1000*space_dim
        f3 = 10*space_dim
        max_time = 3600.

        if seed is not None:
            torch.manual_seed(seed)

        primal_pos_dist = torch.distributions.Uniform(low=-self.max_dist/f1, high=self.max_dist/f1)
        primal_vel_dist = torch.distributions.Uniform(low=-self.max_dist/f2, high=self.max_dist/f2)
        forecast_pos_dist = torch.distributions.Uniform(low=-self.max_dist/f1, high=self.max_dist/f1)
        forecast_vel_dist = torch.distributions.Uniform(low=-self.max_dist/f3, high=self.max_dist/f3)
        forecast_time_dist = torch.distributions.Uniform(low=max_time/3, high=2*max_time/3)

        primal_pos = torch.zeros((num_states, 1, space_dim)).to(self.device)
        primal_vel = torch.zeros((num_states, 1, space_dim)).to(self.device)
        # primal_pos = primal_pos_dist.sample((num_states, 1, space_dim)).to(self.device)
        # primal_vel = primal_vel_dist.sample((num_states, 1, space_dim)).to(self.device)
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

    def getObss2(self, states: torch.Tensor, require_grad=False):
        '''
            forecast pos and vel data only.
        '''
        with torch.set_grad_enabled(require_grad):
            data = self.obssNormalize(states, require_grad=require_grad)
            # data = states
            decoded = self.statesDecode(data)
            forecast_pos = decoded["forecast_pos"]
            forecast_vel = decoded["forecast_vel"]
            obss = torch.cat((forecast_pos.reshape(-1, 3*self.n_debris),
                              forecast_vel.reshape(-1, 3*self.n_debris),
                            ), dim=1)
            return obss
            
    def getPlanRewards(self, states:torch.Tensor, targets:torch.Tensor, sigma=0.1, require_grad=True):
        with torch.set_grad_enabled(require_grad):
            noise = torch.randn_like(targets)*sigma
            targets_noised = targets + noise
            decoded = self.statesDecode(states)
            forecast_pos = decoded["forecast_pos"]
            forecast_vel = decoded["forecast_vel"]
            def get_penalty(targets_):
                targets_proj, targets_orth = utils.lineProj(targets_.unsqueeze(1), forecast_pos, forecast_vel)
                d2p_normSqar = torch.sum((targets_orth/self.safe_dist)**2, dim=-1)
                penalty = utils.penaltyFuncPosi(d2p_normSqar)
                penalty = torch.sum(penalty, dim=-1)
                return penalty
            penalty0 = get_penalty(targets)
            penalty1 = get_penalty(targets_noised)
            penalty = (penalty0+penalty1)/2
            d2o_normSqar0 = torch.sum((targets/self.max_dist)**2, dim=-1)
            d2o_normSqar1 = torch.sum((targets_noised/self.max_dist)**2, dim=-1)
            rewards0 = penalty0-d2o_normSqar0
            rewards1 = penalty1-d2o_normSqar1
            adv = (rewards1-rewards0).detach()
            adv = torch.clamp(adv, min=0)
            diff = targets_noised.detach()-targets
            adv_loss = adv*torch.norm(diff, dim=-1)
            return rewards0, rewards1, adv_loss
        
    def best_targets(self, states:torch.Tensor, population=1000, lr=1e1, max_loop=20000):
        Rewards = []
        batch_size = states.shape[0]
        states = torch.tile(states, (population, 1, 1))
        states = states.transpose(0,1)
        states = states.reshape((batch_size*population, self.state_dim))
        targets0_dist = torch.distributions.Uniform(low=-self.max_dist, high=self.max_dist)
        targets = targets0_dist.sample((batch_size*population, 3))
        targets = torch.tensor(targets, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([targets], lr=lr)
        for _ in range(max_loop):
            opt.zero_grad()
            rewards0, rewards1, adv_loss = self.getPlanRewards(states, targets, require_grad=True)
            loss = -rewards0.mean()
            loss.backward()
            opt.step()
            rewards0 = rewards0.reshape((batch_size, population))
            best_r, best_i = torch.max(rewards0.detach(), dim=1)
            Rewards.append(best_r)
        Rewards = torch.stack(Rewards, dim=1)
        return targets.detach(), best_i, Rewards

    def best_impulses(self, states:torch.Tensor, population=100, impulse_num=1, lr=1e-1, max_loop=1000, horizon=None):
        horizon = self.h2_step if horizon is None else horizon
        batch_size = states.shape[0] # 1
        states0 = torch.tile(states, (population, 1, 1))
        states0 = states0.transpose(0,1)
        states0 = states0.reshape((batch_size*population, self.state_dim))
        
        impulses0_dist = torch.distributions.Uniform(low=-self.impulse_bound, high=self.impulse_bound)
        _impulses = impulses0_dist.sample((batch_size*population, impulse_num, 3))
        _impulses = torch.tensor(_impulses, requires_grad=True, device=self.device)
        opt = torch.optim.Adam([_impulses], lr=lr)

        dummy_h1actions = torch.zeros((batch_size*population, 3)).to(self.device)

        Best_Rewards_List = []
        with Progress() as progress:
            task = progress.add_task("finding best impulses:", total=max_loop)
            for i in range(max_loop):
                states = states0.clone()
                _impulses_norm = torch.norm(_impulses, dim=2, keepdim=True)
                impulses_unit = _impulses/_impulses_norm
                impulses_norm = torch.tanh(_impulses_norm)*self.impulse_bound
                impulses = impulses_unit*impulses_norm
                # impulses = torch.tanh(_impulses)*self.impulse_bound
                # Rewards = torch.zeros((batch_size*population,), device=self.device, requires_grad=True)
                # Rewards = -torch.sum(torch.norm(impulses/self.impulse_bound, dim=2), dim=1)/horizon
                # Rewards = -torch.sum(torch.abs(impulses/self.impulse_bound), dim=(1,2))/horizon
                Rewards = -torch.sum(impulses_norm.squeeze(-1), dim=1)/horizon
                for j in range(impulse_num):
                    states[:,3:6] = states[:,3:6] + impulses[:,j]
                    for k in range(horizon):
                        states, obss, h1r, h2r, dones, _ = self.propagate(states, h1_actions=dummy_h1actions, actions=None, require_grad=True)
                        Rewards = Rewards + h2r
                        done = torch.all(dones)
                loss = -Rewards.mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                Rewards = Rewards.reshape((batch_size, population))
                best_r, best_i = torch.max(Rewards.detach(), dim=1)
                Best_Rewards_List.append(best_r.detach())
                progress.update(task, advance=1)
        Best_Rewards_List = torch.stack(Best_Rewards_List, dim=1)
        impulses = impulses.detach().reshape((batch_size, population, impulse_num, 3))
        return impulses, best_i, Best_Rewards_List
    
    def impulses_test(self, states:torch.Tensor, impulses:torch.Tensor, horizon=None):
        horizon = self.h2_step if horizon is None else horizon
        batch_size = states.shape[0]
        impulse_num = impulses.shape[1]
        dummy_h1actions = torch.zeros((batch_size, 3)).to(self.device)
        obss = self.getObss(states)
        trans_dict_h1 = D.init_transDictBatch(length=impulse_num, batch_size=batch_size, state_dim=self.state_dim, obs_dim=self.obs_dim, action_dim=3, struct="torch", device=self.device)
        trans_dict_h2 = D.init_transDictBatch(length=horizon*impulse_num, batch_size=batch_size, state_dim=self.state_dim, obs_dim=self.obs_dim, action_dim=3, struct="torch", device=self.device)
        step = 0
        impulses = impulses.detach()
        impulses_norm = torch.norm(impulses, dim=-1)
        Rewards = -torch.sum(impulses_norm, dim=1)/horizon
        for i in range(impulse_num):
            trans_dict_h1["states"][i,...] = states[...]
            trans_dict_h1["obss"][i,...] = obss[...]
            trans_dict_h1["actions"][i,...] = impulses[:,i]
            trans_dict_h2["actions"][step,...] = impulses[:,i]
            for j in range(horizon):
                trans_dict_h2["states"][step,...] = states[...]
                trans_dict_h2["obss"][step,...] = obss[...]
                if j==0:
                    states[:,3:6] = states[:,3:6] + impulses[:,i]
                states, obss, h1r, h2r, dones, _ = self.propagate(states, h1_actions=dummy_h1actions, actions=None, require_grad=False)
                Rewards = Rewards + h2r
                trans_dict_h2["rewards"][step] = h2r
                trans_dict_h2["next_states"][step,...] = states[...]
                trans_dict_h2["obss"][step,...] = obss[...]
                done = torch.all(dones)
                step += 1
            trans_dict_h1["next_states"][i,...] = states[...]
            trans_dict_h1["next_obss"][i,...] = obss[...]
        return trans_dict_h1, trans_dict_h2, Rewards
    
    def impulses_test_actor(self, states:torch.Tensor, actor:torch.nn.Module, horizon=None, impulse_num=None):
        horizon = self.h2_step if horizon is None else horizon
        impulse_num = self.h1_step if impulse_num is None else impulse_num
        batch_size = states.shape[0]
        dummy_h1actions = torch.zeros((batch_size, 3)).to(self.device)
        obss = self.getObss(states)
        trans_dict_h1 = D.init_transDictBatch(length=impulse_num, batch_size=batch_size, state_dim=self.state_dim, obs_dim=self.obs_dim, action_dim=3, struct="torch", device=self.device)
        trans_dict_h2 = D.init_transDictBatch(length=horizon*impulse_num, batch_size=batch_size, state_dim=self.state_dim, obs_dim=self.obs_dim, action_dim=3, struct="torch", device=self.device)
        step = 0
        Rewards = torch.zeros(batch_size, device=self.device)
        for i in range(impulse_num):
            impulse = actor(obss).detach()
            impulse_norm = torch.norm(impulse, dim=-1)
            Rewards = Rewards - impulse_norm/horizon
            trans_dict_h1["states"][i,...] = states[...]
            trans_dict_h1["obss"][i,...] = obss[...]
            trans_dict_h1["actions"][i,...] = impulse[...]
            trans_dict_h2["actions"][step,...] = impulse[...]
            for j in range(horizon):
                trans_dict_h2["states"][step,...] = states[...]
                trans_dict_h2["obss"][step,...] = obss[...]
                if j==0:
                    states[:,3:6] = states[:,3:6] + impulse[...]
                states, obss, h1r, h2r, dones, _ = self.propagate(states, h1_actions=dummy_h1actions, actions=None, require_grad=False)
                Rewards = Rewards + h2r
                trans_dict_h2["rewards"][step] = h2r
                trans_dict_h2["next_states"][step,...] = states[...]
                trans_dict_h2["obss"][step,...] = obss[...]
                done = torch.all(dones)
                step += 1
            trans_dict_h1["next_states"][i,...] = states[...]
            trans_dict_h1["next_obss"][i,...] = obss[...]
        return trans_dict_h1, trans_dict_h2, Rewards

    def propagate(self, states: torch.Tensor, h1_actions: torch.Tensor, actions: torch.Tensor, require_grad=False):
        # impulse
        decoded = self.statesDecode(states, require_grad=require_grad)
        primal = decoded["primal"].squeeze(1)
        h1_actions = torch.where(decoded["h1_step"]==0., h1_actions, torch.zeros_like(h1_actions))
        primal[:,3:] = primal[:,3:]+h1_actions # apply impulse if at beginning of h1 stage 
        decoded["primal"] = primal.unsqueeze(1)
        states = self.statesEncode(decoded, require_grad=require_grad)
        # propagate
        next_states = self.getNextStates(states, require_grad=require_grad)
        h1rewards = torch.zeros(states.shape[0], device=self.device) # dummy
        h2rewards = self.getH2Rewards(states, impulse=h1_actions, require_grad=require_grad)
        dones, terminal_rewards = self.getDones(next_states, require_grad=require_grad)
        next_obss = self.getObss(next_states, require_grad=require_grad)
        return next_states, next_obss, h1rewards, h2rewards, dones, terminal_rewards
      
    def getH2Rewards(self, states: torch.Tensor, impulse:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            decoded = self.statesDecode(states)
            forecast_time = decoded["forecast_time"].squeeze(-1)
            is_approaching = forecast_time>-10
            is_approaching = is_approaching.squeeze(-1)
            d2o, d2d, d2p = self.distances(states, require_grad)

            d2d = (d2d/self.safe_dist)**2
            d2d_rewards = utils.penaltyFuncPosi(d2d) # [-inf, 0]
            d2d_rewards = torch.mean(d2d_rewards, dim=1)

            d2p = (d2p/self.safe_dist)**2
            # total_time = self.h2_step*self.h1_step*self.dt
            # time_scale = ((total_time-forecast_time)/total_time)**2
            d2p_rewards = utils.penaltyFuncPosi(d2p) # [-inf, 0]
            d2p_rewards = d2p_rewards#*time_scale
            d2p_rewards = torch.where(is_approaching, d2p_rewards, 0)
            d2p_rewards = torch.mean(d2p_rewards, dim=1)

            if self.impulse_bound:
                impulse_rewards = -torch.sum((impulse/self.impulse_bound)**2, dim=-1)
            else:
                impulse_rewards = -torch.sum(impulse**2, dim=-1)

            # rewards = (d2d_rewards+d2p_rewards+impulse_rewards)/(self.h2_step*self.h1_step)
            rewards = (d2d_rewards+impulse_rewards)/(self.h2_step*self.h1_step)
            return rewards
    
    def getNextStates(self, states: torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            batch_size = states.shape[0]
            decoded = self.statesDecode(states, require_grad)
            object_states = torch.concatenate((decoded["primal"], decoded["debris"]), dim=1) # shape: (batch_size, 1+n_debris, 6)
            next_object_states = object_states@self.trans_mat.T
            next_object_states[:,0,:] = next_object_states[:,0,:]
            decoded["primal"] = next_object_states[:,0,:]
            decoded["debris"] = next_object_states[:,1:,:]
            decoded["forecast_time"] = decoded["forecast_time"]-self.dt
            decoded["h2_step"] = decoded["h2_step"]+1
            decoded["h1_step"] = torch.where(decoded["h2_step"]>=self.h2_step, decoded["h1_step"]+1, decoded["h1_step"])
            decoded["h2_step"] = torch.where(decoded["h2_step"]>=self.h2_step, 0, decoded["h2_step"])
            next_states = self.statesEncode(decoded, require_grad)
            return next_states
        
    def actor_opt(self, actor, batch_size=128, horizon=None, impulse_num=None):
        '''
            returns: `loss`, `trans_dict`
        '''
        horizon = self.h2_step if horizon is None else horizon
        impulse_num = self.h1_step if impulse_num is None else impulse_num
        states = self.randomInitStates(batch_size)
        dummy_h1actions = torch.zeros((batch_size, 3)).to(self.device)
        obss = self.getObss(states)
        trans_dict_h1 = D.init_transDictBatch(length=impulse_num, batch_size=batch_size, state_dim=self.state_dim, obs_dim=self.obs_dim, action_dim=3, struct="torch", device=self.device)
        trans_dict_h2 = D.init_transDictBatch(length=horizon*impulse_num, batch_size=batch_size, state_dim=self.state_dim, obs_dim=self.obs_dim, action_dim=3, struct="torch", device=self.device)
        step = 0
        Rewards = torch.zeros(batch_size, device=self.device, requires_grad=True)
        for i in range(impulse_num):
            impulse = actor(obss)
            impulse_norm = torch.norm(impulse, dim=-1)
            Rewards = Rewards - impulse_norm/horizon
            trans_dict_h1["states"][i,...] = states[...]
            trans_dict_h1["obss"][i,...] = obss[...]
            trans_dict_h1["actions"][i,...] = impulse[...]
            trans_dict_h2["actions"][step,...] = impulse[...]
            for j in range(horizon):
                trans_dict_h2["states"][step,...] = states[...]
                trans_dict_h2["obss"][step,...] = obss[...]
                if j==0:
                    states[:,3:6] = states[:,3:6] + impulse[...]
                states, obss, h1r, h2r, dones, _ = self.propagate(states, h1_actions=dummy_h1actions, actions=None, require_grad=True)
                Rewards = Rewards + h2r
                trans_dict_h2["rewards"][step] = h2r
                trans_dict_h2["next_states"][step,...] = states[...]
                trans_dict_h2["obss"][step,...] = obss[...]
                done = torch.all(dones)
                step += 1
            trans_dict_h1["next_states"][i,...] = states[...]
            trans_dict_h1["next_obss"][i,...] = obss[...]
        return trans_dict_h1, trans_dict_h2, Rewards
