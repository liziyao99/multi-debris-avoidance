import torch
import torch.nn.functional as F
import numpy as np
from env.dynamic import matrix
from utils import lineProj
from env.dynamic import cwutils

class vdPropagator:
    def __init__(self,
                 max_n_debris:int,
                 max_thrust:float,
                 dt=.1, 
                 orbit_rad=7e6, 
                 max_dist=1e3, 
                 safe_dist=5e1,
                 view_dist=1e4,
                 p_new_debris=0.001,
                 device:str=None) -> None:
        self.max_n_debris = max_n_debris
        self.dt = dt
        self.orbit_rad = orbit_rad
        self.max_dist = max_dist
        self.safe_dist = safe_dist
        self.view_dist = view_dist
        self.p_new_debris = p_new_debris
        self.device = device
        
        trans_mat = matrix.CW_TransMat(0, dt, orbit_rad)
        self.trans_mat = torch.from_numpy(trans_mat).float().to(device)

        state_mat = matrix.CW_StateMat(orbit_rad)
        self.state_mat = torch.from_numpy(state_mat).float().to(device)

        self._obs_noise = torch.zeros(6, device=device)
        self._obs_noise[:3] = 1e1
        self._obs_noise[3:] = 1e-1
        self.obs_noise_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._obs_noise),
            scale=self._obs_noise)
        self._obs_noise = self._obs_noise.reshape((1, 1, 6))

        self._obs_zoom = torch.zeros(6, device=device)
        self._obs_zoom[:3] = 1/max_dist
        self._obs_zoom[3:] = 10/max_dist
        self._obs_zoom = self._obs_zoom.reshape((1, 6))

        self._ps0_scale = torch.zeros(6, device=device)
        self._ps0_scale[:3] = self.max_dist/10
        self._ps0_scale[3:] = self.max_dist/1000
        self.primal_state0_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._ps0_scale),
            scale=self._ps0_scale)
        
        self._dsC_scale = torch.zeros(6, device=device)
        self._dsC_scale[:3] = self.max_dist/10
        self._dsC_scale[3:] = self.max_dist/10
        self.debris_stateC_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._dsC_scale),
            scale=self._dsC_scale)
        
        self.max_thrust = max_thrust
        self.collision_reward = -1000.
        self.area_reward = 1.
        
    def randomPrimalStates(self, n:int) -> torch.Tensor:
        return self.primal_state0_dist.sample((n,))
    
    def randomDebrisStates(self, n:int, old:torch.Tensor=None) -> torch.Tensor:
        stateC = self.debris_stateC_dist.sample((n,))
        states, t2c = cwutils.CW_rInv_batch_torch(self.orbit_rad, stateC, self.view_dist)
        if old is not None:
            states = torch.cat((old, states), dim=0)
        return states
    
    def propagate(self, 
                  primal_states:torch.Tensor, 
                  debris_states:torch.Tensor, 
                  actions:torch.Tensor, 
                  new_debris=True,
                  batch_debris_obss=True,
                  require_grad=False):
        '''
            returns: `(next_primal_states, next_debris_states)`, `rewards`, `dones`, `(next_primal_obss, next_debris_obss)`
        '''
        next_primal_states, next_debris_states = self.getNextStates(primal_states, debris_states, actions, require_grad=require_grad)
        rewards = self.getRewards(primal_states, debris_states, actions, require_grad=require_grad)
        dones = self.getDones(primal_states, debris_states, require_grad=False)
        next_debris_states = self.discard_leaving(next_debris_states)
        if new_debris:
            n_debris = next_debris_states.shape[0]
            if n_debris==0:
                next_debris_states = self.randomDebrisStates(1, old=next_debris_states)
            while n_debris<self.max_n_debris and np.random.rand()<self.p_new_debris:
                next_debris_states = self.randomDebrisStates(1, old=next_debris_states)
        next_primal_obss, next_debris_obss = self.getObss(primal_states, debris_states, batch_debris_obss=batch_debris_obss, require_grad=require_grad)
        return (next_primal_states, next_debris_states), rewards, dones, (next_primal_obss, next_debris_obss)

    def getNextStates(self, primal_states:torch.Tensor, debris_states:torch.Tensor, actions:torch.Tensor, require_grad=False):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
                `actions`: shape: (n_primal, 3)
        '''
        with torch.set_grad_enabled(require_grad):
            next_primal_states = primal_states@self.trans_mat.T
            next_debris_states = debris_states@self.trans_mat.T
            thrust = actions*self.max_thrust
            con_vec = matrix.CW_constConVecsT(0, self.dt, thrust, self.orbit_rad) # shape: (n_primal, 6)
            next_primal_states = next_primal_states + con_vec
            return next_primal_states, next_debris_states
        
    def getNextStatesNominal(self, states:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            next_states = states@self.trans_mat.T
            return next_states
    
    def getObss(self, primal_states:torch.Tensor, debris_states:torch.Tensor, batch_debris_obss=True, require_grad=False) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
            returns:
                `primal_obss`: shape: (n_primal, 6)
                `debris_obss`: shape: (n_primal, n_debris, 6) if `batch_debris_obss`, else (n_debris, 6).
        '''
        with torch.set_grad_enabled(require_grad):
            n_primal = primal_states.shape[0]
            n_debris = debris_states.shape[0]
            obs_noise = self.obs_noise_dist.sample((n_debris,))
            debris_states = debris_states + obs_noise
            primal_obss = primal_states*self._obs_zoom
            debris_obss = debris_states*self._obs_zoom
            if batch_debris_obss:
                debris_obss = torch.stack([debris_obss]*n_primal, dim=0)
            return primal_obss, debris_obss
        
    def getRewards(self, primal_states:torch.Tensor, debris_states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
                `actions`: shape: (n_primal, 3)
            returns:
                `rewards`: shape: (n_primal,)
        '''
        with torch.set_grad_enabled(require_grad):
            distances = self.distances(primal_states, debris_states)
            any_collision = (distances<self.safe_dist).any(dim=-1)
            collision_rewards = any_collision.float()*self.collision_reward

            fuel_rewards = 1 - actions.norm(dim=-1)

            d2o = primal_states[:,:3].norm(dim=-1)
            in_area = d2o<self.max_dist
            # area_rewards = self.area_reward*torch.where(in_area, 1-d2o/self.max_dist, self.collision_reward*torch.ones_like(d2o))
            area_rewards = 1-d2o/self.max_dist

            vel = primal_states[:,3:].norm(dim=-1)
            vel_rewards = -vel/20

            rewards = collision_rewards + fuel_rewards + area_rewards + vel_rewards
            return rewards
        
    def getDones(self, primal_states:torch.Tensor, debris_states:torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
            returns:
                `dones`: shape: (n_primal,)
        '''
        with torch.set_grad_enabled(require_grad):
            dones = torch.zeros(primal_states.shape[0], dtype=torch.bool, device=self.device)
            return dones
        
    def discard_leaving(self, debris_states:torch.Tensor) -> torch.Tensor:
        '''
            args:
                `debris_states`: shape: (n, 6)
        '''
        with torch.no_grad():
            leaving = self.is_leaving(debris_states)
            debris_states = debris_states[~leaving]
            return debris_states
        
    def is_leaving(self, states:torch.Tensor) -> torch.Tensor:
        '''
            args:
                `states`: shape: (n, 6)
        '''
        with torch.no_grad():
            pos = states[:, :3]
            vel = states[:, 3:]
            dist = torch.norm(pos, dim=-1)
            dot = torch.sum(pos*vel, dim=-1)
            leaving = (dot>0) & (dist>self.max_dist)
            return leaving
        
    def distances(self, primal_states:torch.Tensor, debris_states:torch.Tensor) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
            returns:
                `distances`: shape: (n_primal, n_debris)
        '''
        with torch.no_grad():
            primal_pos = primal_states[:, :3]
            debris_pos = debris_states[:, :3]
            primal_pos = primal_pos.unsqueeze(dim=1)
            debris_pos = debris_pos.unsqueeze(dim=0)
            distances = torch.norm(primal_pos-debris_pos, dim=-1) # shape: (n_primal, n_debris)
            return distances