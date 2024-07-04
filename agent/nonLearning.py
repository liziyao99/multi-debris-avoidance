import torch
import numpy as np
from agent.agent import rlAgent
import typing
import utils

class nlAgent(rlAgent):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 device=None,
                 ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    def act(self, obs:torch.Tensor):
        raise NotImplementedError
    
    def nominal_act(self, obs:torch.Tensor, require_grad=True):
        raise NotImplementedError

    def save(self, path="../model/dicts.ptd"):
        raise NotImplementedError

    def load(self, path="../model/dicts.ptd"):
        raise NotImplementedError
    
class subSeqAgent(nlAgent):
    def __init__(self, 
                 safe_dist:float,
                 safe_dist_buffer=0.1,
                 opt_step_len=10.,
                 max_loop=10000,
                 break_eps=1e-2,
                 device=None,
                 ) -> None:
        self.main_obs_dim = 6
        self.sub_obs_dim = 6
        super().__init__(6, 3, device)
        self._safe_dist = safe_dist
        self._safe_dist_buffer = safe_dist_buffer
        self.safe_dist = safe_dist*(1+safe_dist_buffer)
        self.opt_step_len = opt_step_len
        self.max_loop = max_loop
        self.break_eps = break_eps

    def act(self, main_states:torch.Tensor, sub_seq:torch.Tensor):
        '''
            args:
                `main_states`: shape (batch_size, 6)
                `sub_seq`: shape (batch_size, n_debris, 6)
        '''
        batch_size = sub_seq.shape[0]
        if main_states.dim()==2:
            main_states = main_states.unsqueeze(1)
        action = torch.zeros((batch_size, 3), device=self.device)
        action[...] = main_states[:,0,:3]
        last_action_increment = torch.zeros_like(action)
        sub_seq_sorted = self.sort_sub_seq(main_states, sub_seq)
        r_d = sub_seq_sorted[:,:, :3]
        v_d = sub_seq_sorted[:,:,3:6]
        r_diff = r_d-main_states[:,:,:3] # shape: (batch_size, n_deris, 3)
        terminate_flag = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
        for i in range(self.max_loop):
            esc_dir, pv_proj, pv_orth = self.get_escape_direction(action, sub_seq_sorted) # shape: (batch_size, n_debris, 3)
            pv_dist = pv_orth.norm(dim=-1, keepdim=True) # shape: (batch_size, n_debris, 1)
            alert = pv_dist<self.safe_dist
            alert_any = alert.sum(dim=1)>0
            terminate_flag = terminate_flag + ~alert_any
            if alert.sum()==0:
                break
            esc_dir = esc_dir*alert
            total_dir = esc_dir.sum(dim=1) # shape: (batch_size, 3)
            if i>0:
                _total_dir_p, _total_dir = utils.lineProj(total_dir, torch.zeros_like(total_dir), last_action_increment)
                align = (utils.dotEachRow(total_dir, _total_dir_p, keepdim=True)>0)
                _total_dir = torch.where(align, total_dir, _total_dir)
                _total_dir = torch.where(terminate_flag, 0, _total_dir)
            else:
                _total_dir = total_dir
            action = torch.where(terminate_flag, action, action+_total_dir*self.opt_step_len)
            last_action_increment[...] = _total_dir*self.opt_step_len
        return action

    def sort_sub_seq(self, main_states:torch.Tensor, sub_seq:torch.Tensor):
        sub_seq_sorted = torch.zeros_like(sub_seq)
        batch_size = sub_seq.shape[0]
        if main_states.dim()==2:
            main_states = main_states.unsqueeze(1)
        r_diff = sub_seq[:,:,:3]-main_states[:,:,:3]
        r_diff = r_diff.norm(dim=-1) # shape: (batch_size, n_debris)
        idx = torch.argsort(r_diff, dim=-1) # shape: (batch_size, n_debris)
        for i in range(batch_size):
            sub_seq_sorted[i] = sub_seq[i, idx[i]]
        return sub_seq_sorted
    
    def get_escape_direction(self, main_states:torch.Tensor, sub_seq:torch.Tensor, normalize=True):
        batch_size = sub_seq.shape[0]
        n_debris = sub_seq.shape[1]
        if main_states.dim()==3:
            main_states = main_states.squeeze(1) # shape: (batch_size, 6)
        escape_directions = torch.zeros((batch_size, n_debris, 3), device=self.device)
        pv_projs = torch.zeros((batch_size, n_debris, 3), device=self.device)
        pv_orths = torch.zeros((batch_size, n_debris, 3), device=self.device)
        for i in range(n_debris):
            pv_proj, pv_orth = utils.lineProj(main_states[:,:3], sub_seq[:,i,:3], sub_seq[:,i,3:])
            escape_directions[:,i,:] = pv_orth
            pv_projs[:,i,:] = pv_proj
            pv_orths[:,i,:] = pv_orth

        if normalize:
            escape_directions = escape_directions / escape_directions.norm(dim=-1, keepdim=True)

        return escape_directions, pv_projs, pv_orths