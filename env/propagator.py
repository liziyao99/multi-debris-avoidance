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
        return obss
    

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

        self.state_mat = matrix.CW_TransMat(0, dt, orbit_rad)
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
        nd2p = d2p/self.safe_dist
        decoded = self.statesDecode(states)
        forecast_time = decoded["forecast_time"].squeeze(axis=-1)
        approaching = forecast_time>0
        n_approaching = np.sum(approaching, axis=1)

        primal_reward = (self.max_dist-d2o.flatten())/self.max_dist - self.k*np.linalg.norm(actions, axis=1)
        debris_reward_each = np.log(nd2p)/nd2p
        debris_reward_each = debris_reward_each*approaching
        debris_reward = np.sum(debris_reward_each, axis=1)
        rewards = (primal_reward+debris_reward)/(1+n_approaching)
        # print(nd2p)
        # print(primal_reward)
        # print(debris_reward)
        # print(rewards)
        # print("\n")
        return rewards
    
    def getTruncatedRewards(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getNextStates(self, states:np.ndarray, actions:np.ndarray) -> np.ndarray:
        batch_size = states.shape[0]
        decoded = self.statesDecode(states)
        object_states = np.concatenate((decoded["primal"], decoded["debris"]), axis=1) # shape: (batch_size, 1+n_debris, 6)
        next_object_states = object_states@self.state_mat.T
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
        forecast_time = np.random.uniform(low=max_time/2, high=max_time, size=(num_states, self.n_debris, 1))

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
        f1 = self.max_dist
        f2 = self.max_dist/100
        states = obss.copy()
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
        obss_n = states_n.copy()
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
        d2o = np.linalg.norm(primal_pos, axis=2) # distance to origin
        d2d = np.linalg.norm(debris_pos-primal_pos, axis=2) # distance to debris
        # print(forecast_pos)
        # print(forecast_vel)
        # print("\n")
        primal_proj, primal_orth = lineProj(primal_pos, forecast_pos, forecast_vel)
        d2p = np.linalg.norm(primal_orth, axis=-1) # distance to debris' forecast
        return d2o, d2d, d2p
