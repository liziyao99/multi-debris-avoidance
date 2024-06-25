import numpy as np
from env.dynamic import matrix
from utils import lineProj

class Propagator:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def getObss(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getTruncatedRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getDones(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def propagate(self, states:np.ndarray, actions:np.ndarray):
        '''
            returns: `next_states`, `rewards`, `dones`, `next_obss`
        '''
        next_states = self.getNextStates(states, actions)
        rewards = self.getRewards(states, actions)
        dones = self.getDones(states)
        next_obss = self.getObss(next_states)
        return next_states, rewards, dones, next_obss
    
    def randomInitStates(self, num_states:int) -> np.ndarray:
        raise NotImplementedError
    
    def obssNormalize(self, obss:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def obssDenormalize(self, obss:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def statesDecode(self, states:np.ndarray):
        raise NotImplementedError
    
    def statesEncode(self, datas:dict):
        raise NotImplementedError
    

class linearSystem(Propagator):
    def __init__(self,
                 state_mat:np.ndarray, 
                 obs_mat:np.ndarray,
                 control_mat:np.ndarray) -> None:
        if state_mat.shape[0]!=state_mat.shape[1]:
            raise ValueError("state_mat must be a square matrix")
        if obs_mat.shape[1]!=obs_mat.shape[0]:
            raise ValueError("shape incompatible: `state_mat` and `obs_dim`")
        if control_mat.shape[0]!=state_mat.shape[0]:
            raise ValueError("shape incompatible: `state_mat` and `control_mat`")
        state_dim = state_mat.shape[0]
        obs_dim = obs_mat.shape[0]
        action_dim = control_mat.shape[1]
        super().__init__(state_dim, obs_dim, action_dim)
        self.state_mat = state_mat.astype(np.float32)
        self.obs_mat = obs_mat.astype(np.float32)
        self.control_mat = control_mat.astype(np.float32)

    def getObss(self, states:np.ndarray) -> np.ndarray:
        obs = states@self.obs_mat.T
        return obs

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise(NotImplementedError)

    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        next_states = states@self.state_mat.T + actions@self.control_mat.T
        return next_states

    def getDones(self, states:np.ndarray) -> np.ndarray:
        raise(NotImplementedError)
    

class debugPropagator(Propagator):
    def __init__(self, max_dist=10.) -> None:
        state_dim = 6
        obs_dim = 6
        action_dim = 3
        super().__init__(state_dim, obs_dim, action_dim)
        self.max_dist = max_dist
        self.k = 0.2

    def getObss(self, states:np.ndarray) -> np.ndarray:
        return states

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        return (self.max_dist-rad-self.k*vel)/self.max_dist

    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        next_states = np.zeros_like(states)
        next_states[:, 3:] = states[:, 3:] + actions # vel
        # next_states[:, :3] = states[:, :3] + next_states[:, 3:] # pos
        next_states[:, :3] = states[:, :3] + states[:, 3:] # pos
        return next_states

    def getDones(self, states:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        dones = rad>self.max_dist
        return dones
    
    def norms(self, states):
        return np.linalg.norm(states[:, :3], axis=1), np.linalg.norm(states[:, 3:], axis=1)
    
    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.zeros((num_states, self.state_dim), dtype=np.float32)
        states[:,:3] = np.random.uniform(low=-self.max_dist/2, high=self.max_dist/2, size=(num_states,3))
        states[:,3:] = np.random.uniform(low=-self.max_dist/100, high=self.max_dist/100, size=(num_states,3))
        return states

class debugLinearSystem(linearSystem):
    def __init__(self, state_mat:np.ndarray, obs_mat:np.ndarray, control_mat:np.ndarray, 
                 max_state_norm=10., k1=1., k2=.2) -> None:
        super().__init__(state_mat, obs_mat, control_mat)
        self.max_state_norm = max_state_norm
        self.k1 = k1
        self.k2 = k2

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        state_rewards = self.k1*(self.max_state_norm-np.linalg.norm(states[:,:3], axis=1))
        state_rewards2 = -self.k2*np.linalg.norm(states[:,3:], axis=1)
        action_rewards = -self.k2*np.linalg.norm(actions, axis=1)
        return (state_rewards+state_rewards2)/self.max_state_norm

    def getDones(self, states:np.ndarray) -> np.ndarray:
        return np.linalg.norm(states[:,:3], axis=1)>self.max_state_norm
    
    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.random.uniform(low=-self.max_state_norm/self.state_dim/10, 
                                   high=self.max_state_norm/self.state_dim/10, 
                                   size=(num_states,self.state_dim))
        return states
    
class motionSystem(linearSystem):
    def __init__(self, state_mat: np.ndarray, max_dist=10.) -> None:
        state_dim = state_mat.shape[0]
        if state_dim%2!=0:
            raise ValueError("state_mat must be an even integer, representing position and velocity.")
        space_dim = state_dim//2
        obs_mat = np.eye(state_dim)
        control_mat = np.vstack((np.zeros((space_dim,space_dim)),np.eye(space_dim)))
        super().__init__(state_mat, obs_mat, control_mat)
        self.space_dim = space_dim
        
        self.max_dist = max_dist
        self.k=0.2

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        return (self.max_dist-rad-self.k*vel)/self.max_dist

    def getDones(self, states:np.ndarray) -> np.ndarray:
        rad, vel = self.norms(states)
        dones = rad>self.max_dist
        return dones
    
    def norms(self, states):
        return np.linalg.norm(states[:, :self.space_dim], axis=1), np.linalg.norm(states[:, self.space_dim:], axis=1)

    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.zeros((num_states, self.state_dim), dtype=np.float32)
        f1 = self.space_dim
        f2 = 10*self.space_dim
        states[:,:self.space_dim] = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states,self.space_dim))
        states[:,self.space_dim:] = np.random.uniform(low=-self.max_dist/f2, high=self.max_dist/f2, size=(num_states,self.space_dim))
        return states
    

class CWPropagator(motionSystem):
    def __init__(self, dt=1., orbit_rad=7e6, max_dist=5e3) -> None:
        state_mat = matrix.CW_TransMat(0, dt, orbit_rad)
        super().__init__(state_mat, max_dist=max_dist)
        self.dt = dt
        self.orbit_rad = orbit_rad
        self.k = .01

    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        iter_step = 1
        r0, v0 = states[:, :3], states[:, 3:]
        R = np.zeros((iter_step, batch_size, 3), dtype=np.float32)
        V = np.zeros((iter_step, batch_size, 3), dtype=np.float32)
        R[0,...] = r0
        V[0,...] = v0
        for i in range(1, iter_step):
            R[i,...] = R[i-1,...] + V[i-1,...]
            V[i,...] = V[i-1,...] + actions*self.dt
        rads = np.linalg.norm(R, axis=2)
        mean_rads = np.mean(rads, axis=0)
        return (self.max_dist-mean_rads)/self.max_dist - self.k*np.linalg.norm(actions, axis=1)

    def getNextStates(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        con_vec = matrix.CW_constConVecs(0, self.dt, actions, self.orbit_rad)
        next_states = states@self.state_mat.T + con_vec
        return next_states
    
    def randomInitStates(self, num_states: int) -> np.ndarray:
        states = np.zeros((num_states, self.state_dim), dtype=np.float32)
        f1 = self.space_dim
        f2 = 1000*self.space_dim
        states[:,:self.space_dim] = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states,self.space_dim))
        states[:,self.space_dim:] = np.random.uniform(low=-self.max_dist/f2, high=self.max_dist/f2, size=(num_states,self.space_dim))
        return states
    
    def obssNormalize(self, obss: np.ndarray) -> np.ndarray:
        f1 = self.max_dist
        f2 = self.max_dist/100
        obss_n = obss.copy()
        obss_n[:,:self.space_dim] /= f1
        obss_n[:,self.space_dim:] /= f2
        return obss_n
    
    def getObss(self, states: np.ndarray) -> np.ndarray:
        obss = super().getObss(states)
        return self.obssNormalize(obss)
    
from env.dynamic import cwutils

class CWDebrisPropagator(Propagator):
    def __init__(self, n_debris, dt=1., orbit_rad=7e6, max_dist=5e3, safe_dist=5e2) -> None:
        state_dim = 6*(1+n_debris)+7*n_debris # states of primal and debris, and debris' forecast data
        obs_dim = 6*(1+n_debris)+7*n_debris # same
        action_dim = 3 # thrust applied on primal
        super().__init__(state_dim, obs_dim, action_dim)

        self.trans_mat = matrix.CW_TransMat(0, dt, orbit_rad)
        self.state_mat = matrix.CW_StateMat(orbit_rad)
        self.dt = dt
        self.orbit_rad = orbit_rad
        self.n_debris = n_debris
        self.max_dist = max_dist
        self.safe_dist = safe_dist

        self.k = .01
        '''
            parameter of control to reward
        '''

    def statesDecode(self, states:np.ndarray):
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
            "primal": primal_states.copy(),
            "debris": debris_states.copy(),
            "forecast_time": forecast_time.copy(),
            "forecast_pos": forecast_pos.copy(),
            "forecast_vel": forecast_vel.copy(),
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
        states = np.concatenate((primal_states.reshape(-1, 6), 
                                 debris_states.reshape(-1, 6*self.n_debris),
                                 forecast_time.reshape(-1, self.n_debris), 
                                 forecast_pos.reshape(-1, 3*self.n_debris),
                                 forecast_vel.reshape(-1, 3*self.n_debris)
                                ), axis=1)
        return states

    def getObss(self, states:np.ndarray) -> np.ndarray:
        obss = states.copy()
        return self.obssNormalize(obss)
    
    def getRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        d2o, d2d, d2p = self.distances(states)
        # nd2p = d2p/self.safe_dist
        decoded = self.statesDecode(states)
        forecast_time = decoded["forecast_time"].squeeze(axis=-1)
        primal_pos = decoded["primal"][:,:,:3]
        approaching = forecast_time>0
        n_approaching = np.sum(approaching, axis=1)

        forecast_states = np.concatenate((decoded["forecast_pos"],decoded["forecast_vel"]),axis=-1)
        forecast_acc = (forecast_states@self.state_mat.T)[:,:,3:]
        n_acc = forecast_acc/np.linalg.norm(forecast_acc, axis=-1, keepdims=True)
        b = decoded["forecast_pos"]-self.safe_dist*n_acc
        debris_reward_each = -np.sum((primal_pos-b)*n_acc/self.max_dist, axis=-1) # shape (batch_size, n_debris)
        debris_reward_each = np.clip(debris_reward_each, a_max=0, a_min=-np.inf)
        debris_reward_each = debris_reward_each*approaching
        debris_reward = np.sum(debris_reward_each, axis=1)

        primal_reward = (self.max_dist-d2o.flatten())/self.max_dist - self.k*np.linalg.norm(actions, axis=1)
        # debris_reward_each = np.log(nd2p)/nd2p
        # debris_reward_each = -np.where((1-nd2p)>0, 2*(1-nd2p), 0.1*(1-nd2p)/self.n_debris)
        # debris_reward_each = debris_reward_each*approaching
        # debris_reward = np.sum(debris_reward_each, axis=1)
        rewards = (primal_reward+debris_reward)/(1+n_approaching)
        return rewards
    
    def getTruncatedRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        decoded = self.statesDecode(states)
        object_states = np.concatenate((decoded["primal"], decoded["debris"]), axis=1) # shape: (batch_size, 1+n_debris, 6)
        next_object_states = object_states@self.trans_mat.T
        con_vec = matrix.CW_constConVecs(0, self.dt, actions, self.orbit_rad)
        next_object_states[:,0,:] += con_vec
        decoded["primal"] = next_object_states[:,0,:]
        decoded["debris"] = next_object_states[:,1:,:]
        decoded["forecast_time"] -= self.dt
        next_states = self.statesEncode(decoded)
        return next_states
    
    def getDones(self, states:np.ndarray) -> np.ndarray:
        d2o, d2d, _ = self.distances(states)
        out_dist = (d2o>self.max_dist).flatten()
        collision = np.max(d2d<self.safe_dist, axis=1)
        dones = out_dist+collision
        return dones
    
    def randomInitStates(self, num_states:int) -> np.ndarray:
        space_dim = 3
        f1 = 10*space_dim
        f2 = 1000*space_dim
        f3 = 10*space_dim
        max_time = 3600.

        primal_pos = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states, 1, space_dim))
        primal_vel = np.random.uniform(low=-self.max_dist/f2, high=self.max_dist/f2, size=(num_states, 1, space_dim))
        forecast_pos = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states, self.n_debris, space_dim))
        forecast_vel = np.random.uniform(low=-self.max_dist/f3, high=self.max_dist/f3, size=(num_states, self.n_debris, space_dim))
        forecast_time = np.random.uniform(low=max_time/3, high=2*max_time/3, size=(num_states, self.n_debris, 1))

        primal_states = np.concatenate((primal_pos, primal_vel), axis=2)
        forecast_states = np.concatenate((forecast_pos, forecast_vel), axis=2)
        debris_states = cwutils.CW_tInv_batch(a = np.array([self.orbit_rad]*(num_states*self.n_debris)), 
                                                       forecast_states = forecast_states.reshape((num_states*self.n_debris,space_dim*2)), 
                                                       t2c = forecast_time.flatten())
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
    
    def obssNormalize(self, obss:np.ndarray) -> np.ndarray:
        # f1 = self.max_dist
        # f2 = self.max_dist/100
        # states = obss.copy()
        # decoded = self.statesDecode(states) # obss is states
        # primal_pos = decoded["primal"][:, :, :3]
        # debris_pos = decoded["debris"][:, :, :3]
        # primal_vel = decoded["primal"][:, :, 3:]
        # debris_vel = decoded["debris"][:, :, 3:]
        # primal_pos_n = primal_pos/f1
        # debris_pos_n = debris_pos/f1
        # primal_vel_n = primal_vel/f2
        # debris_vel_n = debris_vel/f2
        # decoded["primal"][:, :, :3] = primal_pos_n
        # decoded["debris"][:, :, :3] = debris_pos_n
        # decoded["primal"][:, :, 3:] = primal_vel_n
        # decoded["debris"][:, :, 3:] = debris_vel_n
        # decoded["forecast_pos"] /= f1
        # decoded["forecast_vel"] /= f2
        # states_n = self.statesEncode(decoded)
        # obss_n = states_n
        return obss
    
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
        d2o = np.linalg.norm(primal_pos, axis=2) # distance to origin
        d2d = np.linalg.norm(debris_pos-primal_pos, axis=2) # distance to debris
        primal_proj, primal_orth = lineProj(primal_pos, forecast_pos, forecast_vel)
        d2p = np.linalg.norm(primal_orth, axis=-1) # distance to debris' forecast
        return d2o, d2d, d2p


class CWPlanTrackPropagator(CWDebrisPropagator):
    def __init__(self, 
                 n_debris, 
                 dt=1, 
                 orbit_rad=7000000, 
                 max_dist=5000, 
                 safe_dist=500
                ) -> None:
        super().__init__(n_debris, dt, orbit_rad, max_dist, safe_dist)
        self.obs_dim = 6+self.n_debris*7 # primal state and forecast data, no debris' state

    def randomInitStates(self, num_states:int, 
                         pertinence_pos:np.ndarray=None, 
                         pertinence_states:np.ndarray=None) -> np.ndarray:
        space_dim = 3
        f1 = 1.
        f2 = 1000*space_dim
        f3 = 10*space_dim
        max_time = 3600.

        primal_pos = np.zeros((num_states, 1, space_dim))
        primal_vel = np.zeros((num_states, 1, space_dim))
        # primal_pos = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states, 1, space_dim))
        # primal_vel = np.random.uniform(low=-self.max_dist/f2, high=self.max_dist/f2, size=(num_states, 1, space_dim))
        forecast_pos = np.random.uniform(low=-self.max_dist/f1, high=self.max_dist/f1, size=(num_states, self.n_debris, space_dim))
        forecast_vel = np.random.uniform(low=-self.max_dist/f3, high=self.max_dist/f3, size=(num_states, self.n_debris, space_dim))
        forecast_time = np.random.uniform(low=max_time/3, high=2*max_time/3, size=(num_states, self.n_debris, 1))

        if pertinence_states is not None:
            indices = np.random.choice(self.n_debris, size=pertinence_states.shape[1], replace=False)
            forecast_pos[:,indices] = pertinence_states[:,:,:3]
            forecast_vel[:,indices] = pertinence_states[:,:,3:]

        elif pertinence_pos is not None:
            indices = np.random.choice(self.n_debris, size=pertinence_pos.shape[1], replace=False)
            forecast_pos[:,indices] = pertinence_pos

        primal_states = np.concatenate((primal_pos, primal_vel), axis=2)
        forecast_states = np.concatenate((forecast_pos, forecast_vel), axis=2)
        debris_states = cwutils.CW_tInv_batch(a = np.array([self.orbit_rad]*(num_states*self.n_debris)), 
                                                       forecast_states = forecast_states.reshape((num_states*self.n_debris,space_dim*2)), 
                                                       t2c = forecast_time.flatten())
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

    def getPlanRewards(self, states:np.ndarray, target:np.ndarray):
        batch_size = states.shape[0]
        target = target.reshape((batch_size,1,-1))
        target_pos = target[:, :, :3]
        decoded = self.statesDecode(states)
        primal_pos = decoded["primal"][:,:,:3]
        forecast_pos = decoded["forecast_pos"]
        forecast_vel = decoded["forecast_vel"]
        delta_pos = target_pos-forecast_pos

        forecast_states = np.concatenate((forecast_pos, forecast_vel),axis=-1)
        forecast_acc = (forecast_states@self.state_mat.T)[:,:,3:]
        n_acc = forecast_acc/np.linalg.norm(forecast_acc, axis=-1, keepdims=True)
        n_vel = forecast_vel/np.linalg.norm(forecast_vel, axis=-1, keepdims=True)
        normal_vec = np.cross(n_vel, n_acc, axis=-1)
        normal_vec = normal_vec/np.linalg.norm(normal_vec, axis=-1, keepdims=True)

        pos_vertical, pos_lateral = lineProj(delta_pos, forecast_pos, normal_vec)
        l0, l1 = lineProj(pos_lateral, np.zeros_like(forecast_pos), forecast_vel)
        # suppose acc and vel are vertical
        t = l0/forecast_vel # TODO: acc
        offset1 = t**2*forecast_acc/2
        dist_lateral = np.linalg.norm(l1-offset1, axis=-1)-self.safe_dist

        # dot_lateral = np.sum(delta_pos*(-n_acc), axis=-1) # negative dot product of relative position and debris' acceleration
        # dist_lateral_ex =  dot_lateral-self.safe_dist
        # dist_lateral_in = -dot_lateral-2*self.safe_dist
        # dist_lateral = np.maximum(dist_lateral_ex, dist_lateral_in)

        dot_vertical = np.sum(delta_pos*normal_vec, axis=-1) # dot product of relative position and normal vector of debris' plane
        dist_vertical = np.abs(dot_vertical)-self.safe_dist
        d2d = np.maximum(dist_lateral, dist_vertical) # distance to each debris' forecast
        d2d = np.min(d2d, axis=1) # min distance to each debris' forecast

        nd2t = np.linalg.norm((target_pos-primal_pos), axis=-1)/self.max_dist # normalized distance from primal pos to target
        nd2o = np.linalg.norm(target_pos, axis=-1)/self.max_dist # normalized distance from target to origin

        # rewards_avoid = -(nd2t+nd2o)/2
        rewards_avoid = -nd2o
        rewards_avoid = rewards_avoid.squeeze()
        rewards = np.where(d2d>0, rewards_avoid, -2)
        avoid_rate = (d2d>0).sum()/batch_size
        return rewards, avoid_rate
        
    def getObss(self, states:np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        decoded = self.statesDecode(states)
        primal = decoded["primal"].squeeze(axis=1)
        forecast_time = decoded["forecast_time"].reshape((batch_size, -1))
        forecast_pos = decoded["forecast_pos"].reshape((batch_size, -1))
        forecast_vel = decoded["forecast_vel"].reshape((batch_size, -1))
        obss = np.concatenate((primal, forecast_time, forecast_pos, forecast_vel), axis=1)
        return self.obssNormalize(obss)

    def obssNormalize(self, obss: np.ndarray) -> np.ndarray:
        return obss