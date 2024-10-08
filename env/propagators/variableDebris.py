import torch
import torch.nn.functional as F
import numpy as np
import typing
from env.dynamic import matrix
from utils import lineProj, penaltyFuncPosi, dotEachRow
from env.dynamic import cwutils
import agent.utils as autils

class kineticModel:
    def __init__(self,
                 dim:int,
                 max_thrust:float,
                 dt=.1, 
                 n_substep=1,
                 max_dist=1e3, 
                 gamma:float=None,
                 device:str=None) -> None:
        self.dim = dim
        self.max_thrust = max_thrust
        self.max_dist = max_dist
        self.dt = dt
        self.n_substep = n_substep
        self.gamma = gamma
        self.device = device

        self._ps0_scale = torch.zeros(dim*2, device=device)
        self._ps0_scale[:dim] = self.max_dist/10
        self._ps0_scale[dim:] = self.max_dist/1000
        self.primal_state0_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._ps0_scale),
            scale=self._ps0_scale)
        
        self._obs_zoom = torch.zeros(dim*2, device=device)
        self._obs_zoom[:dim] = 1/max_dist
        self._obs_zoom[dim:] = 10/max_dist
        self._obs_zoom = self._obs_zoom.reshape((1, dim*2))

    def _propagate(self, 
                   primal_states:torch.Tensor, 
                   actions:torch.Tensor, 
                   require_grad=False):
        next_primal_states = self.getNextStates(primal_states, actions, require_grad=require_grad)
        rewards = self.getRewards(next_primal_states, actions, require_grad=require_grad)
        dones = self.getDones(next_primal_states, require_grad=False)
        next_primal_obss = self.getObss(next_primal_states, require_grad=require_grad)
        return next_primal_states, rewards, dones, next_primal_obss

    def randomPrimalStates(self, n:int) -> torch.Tensor:
        return self.primal_state0_dist.sample((n,))
    
    def getNextStates(self, primal_states:torch.Tensor, actions:torch.Tensor, require_grad=False):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
                `actions`: shape: (n_primal, 3)
        '''
        with torch.set_grad_enabled(require_grad):
            next_primal_states = primal_states.clone()
            next_primal_states[:,:self.dim] = next_primal_states[:,:self.dim] + next_primal_states[:,self.dim:]*self.dt
            next_primal_states[:,self.dim:] = next_primal_states[:,self.dim:] + actions*self.max_thrust*self.dt
            return next_primal_states
        
    def getObss(self, primal_states:torch.Tensor, require_grad=False):
        return primal_states*self._obs_zoom
        
    def getRewards(self, primal_states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
                `actions`: shape: (n_primal, 3)
            returns:
                `rewards`: shape: (n_primal,)
        '''
        with torch.set_grad_enabled(require_grad):
            obss = primal_states*self._obs_zoom
            rewards = -obss.norm(dim=-1)
            return rewards
        
    def getDones(self, primal_states:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            d2o = primal_states[:,:self.dim].norm(dim=-1)
            dones_out = d2o>self.max_dist
            dones = dones_out
            return dones

class vdPropagator:
    def __init__(self,
                 max_n_debris:int,
                 max_thrust:float,
                 dt=.1, 
                 n_substep=1,
                 orbit_rad=7e6, 
                 max_dist=1e3, 
                 safe_dist=5e1,
                 view_dist=1e4,
                 p_new_debris=0.001,
                 gamma:float=None,
                 device:str=None) -> None:
        self.max_n_debris = max_n_debris
        self.dt = dt
        self.n_substep = n_substep
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
    
    def _propagate(self, 
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
        if new_debris or next_debris_states.shape[0]==0:
            if next_debris_states.shape[0]==0:
                next_debris_states = self.randomDebrisStates(1, old=next_debris_states)
            while next_debris_states.shape[0]<self.max_n_debris and np.random.rand()<self.p_new_debris:
                next_debris_states = self.randomDebrisStates(1, old=next_debris_states)
        next_primal_obss, next_debris_obss = self.getObss(next_primal_states, next_debris_states, batch_debris_obss=batch_debris_obss, require_grad=require_grad)
        return (next_primal_states, next_debris_states), rewards, dones, (next_primal_obss, next_debris_obss)

    def propagate(self, 
                  primal_states:torch.Tensor, 
                  debris_states:torch.Tensor, 
                  actions:torch.Tensor, 
                  discard_leaving=True,
                  new_debris=True,
                  batch_debris_obss=True,
                  require_grad=False,
                  n_step:int=None):
        '''
            returns: `(next_primal_states, next_debris_states)`, `rewards`, `dones`, `(next_primal_obss, next_debris_obss)`
        '''
        n_step = self.n_substep if n_step is None else n_step
        batch_size = primal_states.shape[0]
        if batch_size!=1:
            raise NotImplementedError("not support batchsize>1 now.")
        Next_primal_states = primal_states.clone()
        Next_debris_states = [debris_states.clone() for _ in range(batch_size)]
        Rewards = torch.zeros(batch_size, device=self.device)
        Dones = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        for step in range(n_step):
            (primal_states, debris_states), rewards, dones, _ = self._propagate(primal_states, debris_states, actions, 
                                                                                discard_leaving=False, 
                                                                                new_debris=False, 
                                                                                batch_debris_obss=batch_debris_obss, 
                                                                                require_grad=require_grad)
            Next_primal_states = torch.where(Dones.unsqueeze(dim=-1), Next_primal_states, primal_states)
            for i in range(batch_size):
                if not Dones[i]:
                    Next_debris_states[i] = debris_states.clone()
            Rewards = Rewards + rewards*(~Dones)
            Dones = Dones | dones
            if Dones.all():
                break
        Next_primal_obss, Next_debris_obss = self.getObss(Next_primal_states, Next_debris_states[0], batch_debris_obss=batch_debris_obss, require_grad=require_grad)
        return (Next_primal_states, Next_debris_states[0]), Rewards, Dones, (Next_primal_obss, Next_debris_obss)

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
        
    def getObss(self, primal_states:torch.Tensor, debris_states:torch.Tensor, batch_debris_obss=True, require_grad=False):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
            returns:
                `primal_obss`: shape: (n_primal, 6)
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
        speed = rel_vel.norm(dim=-1)/(10*self.max_dist)
        min_distance, min_distance_idx = distance.min(dim=-1, keepdim=True) # shape: (n_primal, 1)
        min_approach, min_approach_idx, (sin_theta, cos_theta) = self.closet_approach(primal_states, debris_states, require_grad=require_grad)
        # shape: (n_primal, 1), (n_primal, n_debris, 1)
        md_debris_states = debris_states[min_distance_idx.squeeze(dim=-1)] # shape: (n_primal, 6)
        ma_debris_states = debris_states[min_approach_idx.squeeze(dim=-1)] # shape: (n_primal, 6)
        md_debris_obss = md_debris_states*self._obs_zoom
        ma_debris_obss = ma_debris_states*self._obs_zoom
        min_distance, min_approach = min_distance, min_approach/self.max_dist

        base_primal_obss = primal_states*self._obs_zoom # shape: (n_primal, 6)
        base_debris_obss = rel_states*self._obs_zoom.unsqueeze(dim=0) # shape: (n_primal, n_debris, 6)
        # primal_obss = torch.cat((base_primal_obss, md_debris_obss, min_distance, ma_debris_obss, min_approach), dim=-1) # shape: (n_primal, 20)
        primal_obss = torch.cat((base_primal_obss,), dim=-1) # shape: (n_primal, 6)
        debris_obss = torch.cat((base_debris_obss, distance.unsqueeze(-1), speed.unsqueeze(-1), cos_theta), dim=-1) # shape (n_primal, n_debris, 9)

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
            # distances = self.distances(primal_states, debris_states, require_grad=require_grad)
            # any_collision = (distances<self.safe_dist).any(dim=-1)
            # collision_rewards = any_collision.float()*self.collision_reward

            fuel_rewards = (1-actions.norm(dim=-1))*self.beta_action

            d2o = primal_states[:,:3].norm(dim=-1)
            # area_rewards = 1 - d2o/self.max_dist
            area_rewards = 1

            vel = primal_states[:,3:].norm(dim=-1)
            vel_rewards = -vel*self.beta_vel

            trend_rewards_each, _ = self._trend_analysis(primal_states, debris_states, require_grad=require_grad)
            trend_rewards, _ = torch.min(trend_rewards_each, dim=-1)

            rewards = fuel_rewards + area_rewards + trend_rewards
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
            # distances = self.distances(primal_states, debris_states, require_grad=require_grad)
            # collision_rewards = penaltyFuncPosi(distances, 1/self.safe_dist)*abs(self.collision_reward)

            fuel_rewards = (1-actions.norm(dim=-1))*self.beta_action

            d2o = primal_states[:,:3].norm(dim=-1)
            in_area = d2o<self.max_dist
            # area_rewards = self.area_reward*torch.where(in_area, 1-d2o/self.max_dist, self.collision_reward*torch.ones_like(d2o))
            area_rewards = 1 - d2o/self.max_dist

            vel = primal_states[:,3:].norm(dim=-1)
            vel_rewards = -vel*self.beta_vel
            
            rewards = fuel_rewards + area_rewards
            if self.gamma is not None:
                if self.gamma>=1 or self.gamma<0:
                    raise ValueError("Invalid gamma")
                rewards = rewards*(1-self.gamma)
            return rewards
        
    def _trend_analysis(self, primal_states:torch.Tensor, debris_states:torch.Tensor, require_grad=False):
        with torch.set_grad_enabled(require_grad):
            rel_states = debris_states.unsqueeze(dim=0) - primal_states.unsqueeze(dim=1) # shape: (n_primal, n_debris, 6)            
            rel_pos, rel_vel = rel_states[:,:,:3], rel_states[:,:,3:]
            r, v = torch.norm(rel_pos, dim=-1), torch.norm(rel_vel, dim=-1) # shape: (n_primal, n_debris)
            
            trend_rewards_each = torch.zeros_like(r)

            collided = r<=self.safe_dist
            sin_phi = torch.clip(self.safe_dist/r, max=1-1e-10)
            cos_phi = torch.sqrt(1-sin_phi**2)
            trend_rewards_each = torch.where(collided, -1., trend_rewards_each)
            rpn, rvn = rel_pos.norm(dim=-1, keepdim=True), rel_vel.norm(dim=-1, keepdim=True) # shape: (n_primal, n_debris, 1)
            rpd, rvd = rel_pos/rpn, rel_vel/rvn # shape: (n_primal, n_debris, 3)
            cos_theta = dotEachRow(rpd, rvd) # shape: (n_primal, n_debris)
            sin_theta = torch.sqrt(1-torch.clip(cos_theta**2, max=1.)+1e-10) # in case sqrt(0), which lead to grad nan.
            # cos_psi = -torch.cos(phi)*cos_theta + torch.sin(phi)*sin_theta
            sin_psi = -sin_phi*cos_theta - cos_phi*sin_theta
            separating = ~collided & (cos_theta>=0)
            collision_vel = cos_theta*v
            # trend_rewards_each = torch.where(separating, 0., trend_rewards_each)
            t2c = (r-self.safe_dist)/collision_vel # time to collision, >0
            delta_v = sin_psi*v
            t4a = delta_v/self.max_thrust # time for avoidance
            trend_rewards_each = torch.where(~collided&~separating, torch.clip(-t4a/t2c, min=-1., max=0.), trend_rewards_each)

            info = {
                "collided": collided,
                "separating": separating,
                "sin_phi":sin_phi, "cos_theta":cos_theta, "sin_psi":sin_psi, "delta_v":delta_v,
                "t2c":t2c, "t4a":t4a,
            }

            return trend_rewards_each, info
        
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
    def __init__(self, max_n_debris: int, max_thrust: float, dt=0.1, n_substep=1 , orbit_rad=7e6, max_dist=1e3, safe_dist=5e1, view_dist=1e4, p_new_debris=0.001, gamma: float=None, device:str=None) -> None:
        super().__init__(max_n_debris, max_thrust, dt, n_substep, orbit_rad, max_dist, safe_dist, view_dist, p_new_debris, gamma, device)
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
    
class vdPropagatorDummy(vdPropagatorPlane):
    '''
        debris don't move.
    '''
    def __init__(self, max_n_debris: int, max_thrust: float, dt=0.1, n_substep=1, orbit_rad=7000000, max_dist=1000, safe_dist=50, view_dist=10000, p_new_debris=0.001, gamma: float = None, device: str = None) -> None:
        super().__init__(max_n_debris, max_thrust, dt, n_substep, orbit_rad, max_dist, safe_dist, view_dist, p_new_debris, gamma, device)
    
    def getNextStates(self, primal_states:torch.Tensor, debris_states:torch.Tensor, actions:torch.Tensor, require_grad=False):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6)
                `actions`: shape: (n_primal, 3)
        '''
        with torch.set_grad_enabled(require_grad):
            next_primal_states = primal_states@self.trans_mat.T
            next_debris_states = debris_states.clone()
            con_vec = self.conVecs(actions, require_grad=require_grad) # shape: (n_primal, 6)
            next_primal_states = next_primal_states + con_vec
            return next_primal_states, next_debris_states
        
    def randomDebrisStates(self, n:int, old:torch.Tensor=None) -> torch.Tensor:
        stateC = self.debris_stateC_dist.sample((n,))
        stateC[:,3:] = 0.
        states = stateC
        return states