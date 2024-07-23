import torch
import torch.nn.functional as F
import numpy as np
import typing
from env.dynamic import matrix
from utils import lineProj, penaltyFuncPosi, dotEachRow
from env.dynamic import cwutils
import agent.utils as autils

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
                 gamma:float=None,
                 device:str=None) -> None:
        self.max_n_debris = max_n_debris
        self.dt = dt
        self.orbit_rad = orbit_rad
        self.max_dist = max_dist
        self.safe_dist = safe_dist
        self.view_dist = view_dist
        self.p_new_debris = p_new_debris
        self.gamma = gamma
        self.device = device
        
        trans_mat = matrix.CW_TransMat(0, dt, orbit_rad)
        self.trans_mat = torch.from_numpy(trans_mat).float().to(device)

        state_mat = matrix.CW_StateMat(orbit_rad)
        self.state_mat = torch.from_numpy(state_mat).float().to(device)

        self._with_obs_noise = True
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
        self._dsC_scale[:3] = self.max_dist/2
        self._dsC_scale[3:] = self.max_dist/10
        self.debris_stateC_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._dsC_scale),
            scale=self._dsC_scale)
        
        self.max_thrust = max_thrust
        self.collision_reward = -1.
        self.area_reward = 1.
        self.beta_action = 1/self.max_dist
        self.beta_vel = 1/self.max_dist
        
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
                  discard_leaving=True,
                  new_debris=True,
                  batch_debris_obss=True,
                  require_grad=False):
        '''
            returns: `(next_primal_states, next_debris_states)`, `rewards`, `dones`, `(next_primal_obss, next_debris_obss)`
        '''
        next_primal_states, next_debris_states = self.getNextStates(primal_states, debris_states, actions, require_grad=require_grad)
        rewards = self.getRewards(next_primal_states, next_debris_states, actions, require_grad=require_grad)
        dones = self.getDones(next_primal_states, next_debris_states, require_grad=False)
        if discard_leaving:
            next_debris_states = self.discard_leaving(next_debris_states)
        if new_debris:
            if next_debris_states.shape[0]==0:
                next_debris_states = self.randomDebrisStates(1, old=next_debris_states)
            while next_debris_states.shape[0]<self.max_n_debris and np.random.rand()<self.p_new_debris:
                next_debris_states = self.randomDebrisStates(1, old=next_debris_states)
        next_primal_obss, next_debris_obss = self.getObss(next_primal_states, next_debris_states, batch_debris_obss=batch_debris_obss, require_grad=require_grad)
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
            con_vec = self.conVecs(actions, require_grad=require_grad) # shape: (n_primal, 6)
            next_primal_states = next_primal_states + con_vec
            return next_primal_states, next_debris_states
        
    def conVecs(self, actions:torch.Tensor, require_grad=False):
        with torch.set_grad_enabled(require_grad):
            thrust = actions*self.max_thrust
            return matrix.CW_constConVecsT(0, self.dt, thrust, self.orbit_rad) # shape: (n_primal, 6)
        
    def getNextStatesNominal(self, states:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            next_states = states@self.trans_mat.T
            return next_states
    
    def _getObss(self, primal_states:torch.Tensor, debris_states:torch.Tensor, batch_debris_obss=True, require_grad=False):
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
            if self._with_obs_noise:
                debris_states = debris_states + obs_noise
            rel_states = debris_states.unsqueeze(dim=0) - primal_states.unsqueeze(dim=1) # shape: (n_primal, n_debris, 6)
            primal_obss = primal_states*self._obs_zoom
            # debris_obss = debris_states*self._obs_zoom
            # if batch_debris_obss:
            #     debris_obss = torch.stack([debris_obss]*n_primal, dim=0)
            debris_obss = rel_states*self._obs_zoom.unsqueeze(dim=0)
            return primal_obss, debris_obss
        
    def getObss(self, primal_states:torch.Tensor, debris_states:torch.Tensor, batch_debris_obss=True, require_grad=False):
        '''
            returns:
                `primal_obss`: shape: (n_primal, 20)
                `debris_obss`: shape: (n_primal, n_debris, 9)
        '''
        n_primal = primal_states.shape[0]
        n_debris = debris_states.shape[0]
        obs_noise = self.obs_noise_dist.sample((n_debris,))
        if self._with_obs_noise:
            debris_states = debris_states + obs_noise
        rel_states = debris_states.unsqueeze(dim=0) - primal_states.unsqueeze(dim=1) # shape: (n_primal, n_debris, 6)
        rel_pos, rel_vel = rel_states[:,:,:3], rel_states[:,:,3:]
        distance = rel_pos.norm(dim=-1)/self.max_dist # shape: (n_primal, n_debris)
        min_distance, min_distance_idx = distance.min(dim=-1, keepdim=True) # shape: (n_primal, 1)
        min_approach, min_approach_idx, (sin_theta, cos_theta) = self.closet_approach(primal_states, debris_states, require_grad=require_grad)
        # shape: (n_primal, 1), (n_primal, n_debris, 1)
        md_debris_states = debris_states[min_distance_idx.squeeze(dim=-1)] # shape: (n_primal, 6)
        ma_debris_states = debris_states[min_approach_idx.squeeze(dim=-1)] # shape: (n_primal, 6)
        md_debris_obss = md_debris_states*self._obs_zoom
        ma_debris_obss = ma_debris_states*self._obs_zoom
        min_distance, min_approach = min_distance/self.max_dist, min_approach/self.max_dist

        base_primal_obss = primal_states*self._obs_zoom # shape: (n_primal, 6)
        base_debris_obss = rel_states*self._obs_zoom.unsqueeze(dim=0) # shape: (n_primal, n_debris, 6)
        primal_obss = torch.cat((base_primal_obss, md_debris_obss, min_distance, ma_debris_obss, min_approach), dim=-1) # shape: (n_primal, 20)
        debris_obss = torch.cat((base_debris_obss, distance.unsqueeze(-1), sin_theta, cos_theta), dim=-1) # shape (n_primal, n_debris, 9)

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
            distances = self.distances(primal_states, debris_states, require_grad=require_grad)
            any_collision = (distances<self.safe_dist).any(dim=-1)
            collision_rewards = any_collision.float()*self.collision_reward

            fuel_rewards = (1-actions.norm(dim=-1))*self.beta_action

            d2o = primal_states[:,:3].norm(dim=-1)
            in_area = d2o<self.max_dist
            # area_rewards = self.area_reward*torch.where(in_area, 1-d2o/self.max_dist, self.collision_reward*torch.ones_like(d2o))
            area_rewards = 1 - d2o/self.max_dist

            vel = primal_states[:,3:].norm(dim=-1)
            vel_rewards = -vel*self.beta_vel

            rewards = collision_rewards + fuel_rewards + area_rewards + vel_rewards
            if self.gamma is not None:
                if self.gamma>=1 or self.gamma<0:
                    raise ValueError("Invalid gamma")
                rewards = rewards*(1-self.gamma)
            return rewards
        
    def getSmoothRewards(self, primal_states:torch.Tensor, debris_states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
                `actions`: shape: (n_primal, 3)
            returns:
                `rewards`: shape: (n_primal,)
        '''
        with torch.set_grad_enabled(require_grad):
            distances = self.distances(primal_states, debris_states, require_grad=require_grad)
            rel_states = debris_states.unsqueeze(dim=0) - primal_states.unsqueeze(dim=1) # shape: (n_primal, n_debris, 6)            
            rel_pos, rel_vel = rel_states[:,:,:3], rel_states[:,:,3:]
            collision_rewards = penaltyFuncPosi(distances, 1/self.safe_dist)*abs(self.collision_reward)

            fuel_rewards = (1-actions.norm(dim=-1))*self.beta_action

            d2o = primal_states[:,:3].norm(dim=-1)
            in_area = d2o<self.max_dist
            # area_rewards = self.area_reward*torch.where(in_area, 1-d2o/self.max_dist, self.collision_reward*torch.ones_like(d2o))
            area_rewards = 1 - d2o/self.max_dist

            vel = primal_states[:,3:].norm(dim=-1)
            vel_rewards = -vel*self.beta_vel

            rewards = collision_rewards + fuel_rewards + area_rewards + vel_rewards
            if self.gamma is not None:
                if self.gamma>=1 or self.gamma<0:
                    raise ValueError("Invalid gamma")
                rewards = rewards*(1-self.gamma)
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
            distances = self.distances(primal_states, debris_states, require_grad=require_grad)
            dones_collide = (distances<self.safe_dist).any(dim=-1)
            d2o = primal_states[:,:3].norm(dim=-1)
            dones_out = d2o>self.max_dist
            dones = dones_collide | dones_out
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
        
    def is_leaving(self, debris_states:torch.Tensor) -> torch.Tensor:
        '''
            args:
                `states`: shape: (n, 6)
        '''
        with torch.no_grad():
            pos = debris_states[:, :3]
            vel = debris_states[:, 3:]
            dist = torch.norm(pos, dim=-1)
            dot = torch.sum(pos*vel, dim=-1)
            leaving = (dot>0) & (dist>self.max_dist)
            return leaving
        
    def distances(self, primal_states:torch.Tensor, debris_states:torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
            returns:
                `distances`: shape: (n_primal, n_debris)
        '''
        with torch.set_grad_enabled(require_grad):
            if isinstance(debris_states, torch.Tensor):
                primal_pos = primal_states[:, :3]
                debris_pos = debris_states[:, :3]
                primal_pos = primal_pos.unsqueeze(dim=1)
                debris_pos = debris_pos.unsqueeze(dim=0)
                distances = torch.norm(primal_pos-debris_pos, dim=-1) # shape: (n_primal, n_debris)
            return distances
        
    def closet_approach(self, primal_states:torch.Tensor, debris_states:torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
            returns:
                `min_dist`: shape: (n_primal, 1)
                `min_idx`: shape: (n_primal, 1)
                `(sin_theta, cos_theta)`: shape: (n_primal, n_debris, 1)
        '''
        with torch.set_grad_enabled(require_grad):
            rel_states = debris_states.unsqueeze(dim=0) - primal_states.unsqueeze(dim=1)
            rel_pos, rel_vel = rel_states[:,:,:3], rel_states[:,:,3:]
            rpn, rvn = rel_pos.norm(dim=-1, keepdim=True), rel_vel.norm(dim=-1, keepdim=True) # shape: (n_primal, n_debris, 1)
            rpd, rvd = rel_pos/rpn, rel_vel/rvn # shape: (n_primal, n_debris, 3)
            cos_theta = dotEachRow(rpd, rvd, keepdim=True) # shape: (n_primal, n_debris, 1)
            sin_theta = torch.sqrt(1-torch.clip(cos_theta**2, max=1.)+1e-10) # in case sqrt(0), which lead to grad nan.
            closet_dist = rel_pos.norm(dim=-1)*sin_theta.squeeze(-1) # shape: (n_primal, n_debris)
            min_dist, min_idx = closet_dist.min(dim=1, keepdim=True)
            return min_dist, min_idx, (sin_theta, cos_theta)
        
class vdPropagatorPlane(vdPropagator):
    def __init__(self, max_n_debris: int, max_thrust: float, dt=0.1, orbit_rad=7e6, max_dist=1e3, safe_dist=5e1, view_dist=1e4, p_new_debris=0.001, gamma: float=None, device:str=None) -> None:
        super().__init__(max_n_debris, max_thrust, dt, orbit_rad, max_dist, safe_dist, view_dist, p_new_debris, gamma, device)
        self.state_mat = torch.tensor(
            [[0,0,0,1,0,0],
             [0,0,0,0,1,0],
             [0,0,0,0,0,1],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0]], dtype=torch.float32, device=device
        )

        self.trans_mat = torch.tensor(
            [[1,0,0,dt,0,0],
             [0,1,0,0,dt,0],
             [0,0,1,0,0,dt],
             [0,0,0,1,0,0],
             [0,0,0,0,1,0],
             [0,0,0,0,0,1]], dtype=torch.float32, device=device
        )

    def conVecs(self, actions: torch.Tensor, require_grad=False):
        with torch.set_grad_enabled(require_grad):
            thrust = actions*self.max_thrust
            con_vecs = torch.zeros((thrust.shape[0], 6), dtype=torch.float32, device=thrust.device)
            con_vecs[:, 3:] = thrust*self.dt
            con_vecs[:, :3] = thrust*(self.dt**2)/2
            return con_vecs
        
    def randomDebrisStates(self, n:int, old:torch.Tensor=None) -> torch.Tensor:
        p_pertinence = 0.5
        stateC = self.debris_stateC_dist.sample((n,))
        if np.random.rand()<p_pertinence:
            stateC[:,:3] = self.primal_state0_dist.sample((n,))[:,:3]
        states = stateC
        states_0 = torch.zeros_like(stateC)
        t2c = torch.zeros(n, device=self.device)
        flags = torch.zeros(n, dtype=torch.bool, device=self.device,)
        max_loop = 1000
        for k in range(max_loop):
            r = torch.norm(states[:,:3], dim=-1)
            out = r>self.view_dist
            _new = out & ~flags
            states_0[_new] = states[_new]
            t2c[_new] = k*self.dt
            flags = out | flags
            if flags.all():
                break
            states[:,:3] = states[:,:3] - self.dt*states[:,3:]
        if old is not None:
            states = torch.cat((old, states), dim=0)
        return states
    
    def randomDebrisStatesTime(self, n:int, old:torch.Tensor=None) -> torch.Tensor:
        p_pertinence = 0.5
        stateC = self.debris_stateC_dist.sample((n,))
        if np.random.rand()<p_pertinence:
            stateC[:,:3] = self.primal_state0_dist.sample((n,))[:,:3]
        states = stateC
        states_0 = torch.zeros_like(stateC)
        t2c = torch.zeros(n, device=self.device)
        step_span = torch.tensor([10, 100], device=self.device).float()
        step_dist = torch.distributions.Uniform(step_span[0], step_span[1])
        arrive_step = step_dist.sample((n,))
        flags = torch.zeros(n, dtype=torch.bool, device=self.device,)
        max_loop = 1000
        for k in range(max_loop):
            out = arrive_step<=k
            _new = out & ~flags
            states_0[_new] = states[_new]
            t2c[_new] = k*self.dt
            flags = out | flags
            if flags.all():
                break
            states[:,:3] = states[:,:3] - self.dt*states[:,3:]
        if old is not None:
            states = torch.cat((old, states), dim=0)
        return states