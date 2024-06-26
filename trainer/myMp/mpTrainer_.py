import numpy as np
import torch
import torch.multiprocessing as mp
import rich.progress
import typing

from data.buffer import replayBuffer
import data.dicts as D
import env.propagators
import agent.agent as A
import env.propagators.hierarchicalPropagator
import trainer
from agent.agent import rlAgent
from trainer.trainer import dummyTrainer
from plotting.dataplot import logPlot, dataPlot
import trainer.trainer


class mpWorker(mp.Process):
    '''
        NOTE: a Process can not start twice.
    '''
    def __init__(self, trainer, name:str,
                 inq:mp.Queue=None, outq:mp.Queue=None, sim_kwargs:dict={}) -> None:
        mp.Process.__init__(self)
        self.trainer = trainer
        self.name = name
        self.inq = inq
        self.outq = outq
        self.sim_kwargs = sim_kwargs

    def simulate(self, **kwargs):
        raise NotImplementedError

    def run(self):
        print(f"{self.name} running. agent device:{self.agent.device}.")
        while (flag := self.inq.get()) is not None:
            obj = self.simulate(**self.sim_kwargs)
            self.outq.put(obj)
        self.outq.put(None)
        print(f"{self.name} done.")

    def setQ(self, inq:mp.Queue=None, outq:mp.Queue=None):
        self.inq = inq if inq is not None else self.inq
        self.outq = outq if outq is not None else self.outq

    @property
    def agent(self) -> rlAgent:
        return self.trainer.agent
    
class mpTrainer:
    def __init__(self, n_process:int, buffer:replayBuffer, trainerArgs:dict={}, main_device="cuda",):
        mainTrainerArgs = {}
        trainerArgs["device"] = "cpu" # bug: (Tensor elements become 0) encountered when subprocess run on gpu
        for key in trainerArgs.keys():
            if key!="device":
                mainTrainerArgs[key] = trainerArgs[key]
        mainTrainerArgs["device"] = main_device
        self.main_device = main_device
        self.main_trainer = self._init_trainer(**mainTrainerArgs)
        # TODO: share cuda memory

        self.n_process = n_process
        self.trainerArgs = trainerArgs
        self.workers = []
        self._reset_workers()
        self.buffer = buffer

    def _init_trainer(self) -> dummyTrainer:
        raise NotImplementedError
    
    def _reset_workers(self):
        for w in self.workers:
            w.close()
            del w
        self.workers = []
        for i in range(self.n_process):
            trainer = self._init_trainer(**self.trainerArgs)
            self.workers.append(mpWorker(trainer, f"worker{i}"))

    @property
    def main_agent(self):
        return self.main_trainer.agent
    
    @property
    def agents(self) -> typing.List[rlAgent]:
        return [w.agent for w in self.workers]
    
    def _init_train(self, total:int):
        inq = mp.Queue()
        for _ in range(total):
            inq.put(True)
        for _ in range(self.n_process):
            inq.put(None)
        outq = mp.Queue()
        for i in range(self.n_process):
            self.workers[i].setQ(inq, outq)
        return inq, outq
    
    def pull(self, idx):
        self.agents[idx].actor.load_state_dict(self.main_agent.actor.state_dict())
        self.agents[idx].critic.load_state_dict(self.main_agent.critic.state_dict())
    
    def train(self, n_epoch:int, total_episode:int, folder="../model/", sim_kwargs:dict=None):
        raise NotImplementedError
    
    def debug1p(self):
        self._reset_workers()
        self.pull(0)
        inq = mp.Queue()
        outq = mp.Queue()
        inq.put(1)
        inq.put(None)
        self.workers[0].setQ(inq, outq)
        self.workers[0].run()
        return outq.get()
    

class mpH2TreeWorker(mpWorker):
    def __init__(self, 
                 select_itr:int, select_size:int, trainer:trainer.trainer.H2TreeTrainer, name:str,
                 inq:mp.Queue=None, outq:mp.Queue=None) -> None:
        sim_kwargs = {"select_itr":select_itr, "select_size":select_size}
        mp.Process.__init__(self)
        self.trainer = trainer
        self.name = name
        self.inq = inq
        self.outq = outq
        self.sim_kwargs = sim_kwargs

    def simulate(self, select_itr:int, select_size:int, **kwargs):
        trans_dict, _, v = self.trainer.treeSim(select_itr, select_size, train_h2a=False, **kwargs)
        return trans_dict, v
    
class mpH2TreeTrainer(mpTrainer):
    def __init__(self, n_process:int, buffer:replayBuffer, n_debris:int, agentArgs:dict, 
                 select_itr=10, select_size=32, batch_size=2048, main_device="cuda", mode="default"):
        self._modes = ("default", "alter", "SAC")
        if mode not in self._modes:
            raise ValueError(f"mode must be in {self._modes}.")
        self.mode = mode
        self.n_debris = n_debris
        self.main_device = main_device
        self.agentArgs = agentArgs
        self.select_itr = select_itr
        self.select_size = select_size
        
        self.main_initialized = False
        self.main_trainer = self._init_trainer(main_device)
        self.main_initialized = True

        self.n_process = n_process
        self.workers = []
        self._reset_workers()
        self.buffer = buffer
        
        self.batch_size = batch_size

    def _init_trainer(self, device="cpu",):
        p = env.propagators.hierarchicalPropagator.H2CWDePropagator(self.n_debris, device=device, 
                                                                    h1_step=10, h2_step=360)
        _, tutor = trainer.trainer.CWPTT(self.n_debris, device, "../model/planTrack3.ptd")
        if self.mode=="default":
            agent = A.H2Agent(obs_dim=p.obs_dim,
                          h1obs_dim=p.obs_dim,
                          h2obs_dim=6, 
                          h1out_dim=p.h1_action_dim, 
                          h2out_dim=p.h2_action_dim, 
                          device=device,
                          **self.agentArgs)
            return trainer.trainer.H2TreeTrainer(p, agent)
        if self.mode=="alter":
            agent = A.H2Agent(obs_dim=p.obs_dim,
                          h1obs_dim=p.obs_dim,
                          h2obs_dim=6, 
                          h1out_dim=p.h1_action_dim, 
                          h2out_dim=p.h2_action_dim, 
                          device=device,
                          **self.agentArgs)
            return trainer.trainer.H2TreeTrainerAlter(p, agent, tutor)
        if self.mode=="SAC":
            agent = A.SAC(obs_dim=p.obs_dim,
                          action_dim=p.h1_action_dim,
                          device=device,
                          **self.agentArgs)
            return trainer.trainer.H2TreeTrainerAlter(p, agent, tutor)
    
    def _reset_workers(self):
        for w in self.workers:
            w.close()
            del w
        self.workers = []
        for i in range(self.n_process):
            trainer = self._init_trainer()
            self.workers.append(mpH2TreeWorker(self.select_itr, self.select_size, trainer, f"worker{i}"))

    def pull(self, idx):
        if self.mode=="default":
            self.agents[idx].h1a.load_state_dict(self.main_agent.h1a.state_dict())
            self.agents[idx].h1c.load_state_dict(self.main_agent.h1c.state_dict())
            self.agents[idx].h2a.load_state_dict(self.main_agent.h2a.state_dict())
        if self.mode=="alter":
            self.agents[idx].h1a.load_state_dict(self.main_agent.h1a.state_dict())
            self.agents[idx].h1c.load_state_dict(self.main_agent.h1c.state_dict())
            self.agents[idx].h2a.load_state_dict(self.main_agent.h2a.state_dict())
            self.workers[idx].trainer.tutor.planner.load_state_dict(self.main_trainer.tutor.planner.state_dict())
            self.workers[idx].trainer.tutor.critic.load_state_dict(self.main_trainer.tutor.critic.state_dict())
            self.workers[idx].trainer.tutor.tracker.load_state_dict(self.main_trainer.tutor.tracker.state_dict())
        if self.mode=="SAC":
            self.agents[idx].actor.load_state_dict(self.main_agent.actor.state_dict())
            self.agents[idx].critic1.load_state_dict(self.main_agent.critic1.state_dict())
            self.agents[idx].critic2.load_state_dict(self.main_agent.critic2.state_dict())
            self.agents[idx].target_critic1.load_state_dict(self.main_agent.target_critic1.state_dict())
            self.agents[idx].target_critic2.load_state_dict(self.main_agent.target_critic2.state_dict())
            self.agents[idx].log_alpha = self.main_agent.log_alpha.detach().to(self.agents[idx].log_alpha)
    
    def train(self, n_epoch:int, total_episode:int, folder="../model/", sim_kwargs:dict=None):
        if sim_kwargs is not None:
            for w in self.workers: w.sim_kwargs = sim_kwargs
        plot = dataPlot(["true_values", "critic_loss", "ddpg_loss", "mc_loss"])
        critic_loss, ddpg_loss, mc_loss, true_values = [], [], [], []
        with rich.progress.Progress() as pbar:
            task1 = pbar.add_task("", total=total_episode)
            for i in range(n_epoch):
                pbar.tasks[task1].description = "epoch {0} of {1}".format(i+1, n_epoch)
                pbar.tasks[task1].completed = 0
                [self.pull(idx) for idx in range(self.n_process)] # mp agents pull parameters from main agent
                inq, outq = self._init_train(total_episode)
                [w.start() for w in self.workers]
                count = 0
                while count<self.n_process: # mp workers running
                    if outq.qsize(): # new result in outq
                        if (obj := outq.get()) is not None:
                            trans_dict = obj[0]
                            v = obj[1]
                            self.buffer.from_dict(trans_dict)
                            trans_dicts = D.split_dict(trans_dict, batch_size=self.batch_size)
                            _Q, _mc, _ddpg = [], [], []
                            for dict in trans_dicts:
                                Q_l, mc_l, ddpg_l = self.main_agent.update(dict)
                                _Q.append(Q_l)
                                _mc.append(mc_l)
                                _ddpg.append(ddpg_l)
                            critic_loss.append(np.mean(_Q))
                            ddpg_loss.append(np.mean(_ddpg))
                            mc_loss.append(np.mean(_mc))
                            true_values.append(v)
                            if pbar.tasks[task1].completed > plot.min_window_size:
                                plot.set_data((true_values, critic_loss, ddpg_loss, mc_loss))
                            pbar.update(task1, advance=1)
                        else:
                            count += 1
                    elif self.buffer.size>=self.buffer.minimal_size: # train using buffer
                        trans_dict_b = self.buffer.sample(batch_size=self.batch_size)
                        self.main_agent.update(trans_dict_b)
                [w.join() for w in self.workers]
                np.savez(folder+"log.npz", 
                        true_values = true_values,
                        critic_loss = critic_loss,
                        ddpg_loss = ddpg_loss,
                        mc_loss = mc_loss
                        )
                plot.save_fig(folder+"log.png")
                #self.main_trainer.agent.save(folder+f"check_point{i}.ptd")
                self.main_trainer.agent.save(folder+"check_point.ptd")
                self._reset_workers()
        return
    
    def debug(self):
        trans_dict, v = self.debug1p()
        trans_dicts = D.split_dict(trans_dict, batch_size=self.batch_size)
        _Q, _mc, _ddpg = [], [], []
        for dict in trans_dicts:
            Q_l, mc_l, ddpg_l = self.main_agent.update(dict)
            _Q.append(Q_l)
            _mc.append(mc_l)
            _ddpg.append(ddpg_l)
        return v, np.mean(_Q), np.mean(_mc), np.mean(_ddpg)
    


class impulsesWorker(mpWorker):
    '''
        NOTE: a Process can not start twice.
    '''
    def __init__(self, 
                 prop:env.propagators.hierarchicalPropagator.impulsePropagator, 
                 name:str,
                 batch_size=256,
                 population=100, 
                 impulse_num=20, 
                 prop_lr=1e-1, 
                 max_loop=15,
                 inq:mp.Queue=None, outq:mp.Queue=None,) -> None:
        mp.Process.__init__(self)
        self.prop = prop
        self.name = name
        self.inq = inq
        self.outq = outq
        self.batch_size = batch_size
        self.population = population
        self.impulse_num = impulse_num
        self.prop_lr = prop_lr
        self.max_loop = max_loop

    def simulate(self, **kwargs):
        states = self.prop.randomInitStates(self.batch_size)
        imp, best_i, Best_Rewards_List = self.prop.best_impulses(states, self.population, self.impulse_num, self.prop_lr, self.max_loop)
        best_imp = torch.zeros((imp.shape[0], imp.shape[2], imp.shape[3]), device=imp.device)
        for i in range(self.batch_size):
            best_imp[i,...] = imp[i,best_i[i],...]
        best_imp_ = best_imp.transpose(0,1)
        td1, td2 = self.prop.impulses_test(states, best_imp)
        return td1, best_imp_ # shape (seq_length, batch_size, ...)

    def run(self):
        print(f"{self.name} running. Device:{self.prop.device}.")
        while (flag := self.inq.get()) is not None:
            obj = self.simulate()
            self.outq.put(obj)
        self.outq.put(None)
        print(f"{self.name} done.")

    def setQ(self, inq:mp.Queue=None, outq:mp.Queue=None):
        self.inq = inq if inq is not None else self.inq
        self.outq = outq if outq is not None else self.outq

    @property
    def agent(self) -> rlAgent:
        raise NotImplementedError
    

class mpImpulsesTrainer(mpTrainer):
    def __init__(self, 
                 n_process:int, 
                 model:torch.nn.Module, 
                 n_debris:int,
                 impulse_bound:float,
                 buffer:replayBuffer=None, 
                 batch_size=256,
                 population=100, 
                 impulse_num=20, 
                 prop_lr=1e-1, 
                 lr=1e-2,
                 max_loop=15, main_device="cuda"):
        self.model = model.to(main_device)
        self.main_device = main_device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.prop_args = {
            "n_debris": n_debris,
            "impulse_bound": impulse_bound,
        }
        self.worker_args = {
            "batch_size": batch_size,
            "population": population,
            "impulse_num": impulse_num,
            "prop_lr": prop_lr,
            "max_loop": max_loop,
        }
        
        self.main_initialized = False
        self.main_prop = self._init_prop(main_device)
        self.main_initialized = True

        self.n_process = n_process
        self.workers = []
        self._reset_workers()
        self.buffer = buffer

    def _init_prop(self, device="cpu",):
        kwargs = self.prop_args.copy()
        kwargs["device"] = device
        p = env.propagators.hierarchicalPropagator.impulsePropagator(**kwargs)
        return p
    
    def _reset_workers(self):
        for w in self.workers:
            w.close()
            del w
        self.workers = []
        for i in range(self.n_process):
            prop = self._init_prop()
            self.workers.append(impulsesWorker(prop=prop, name=f"worker{i}", **self.worker_args))

    def pull(self, idx):
        pass
    
    def train(self, n_epoch:int, total_episode:int, folder="../model/", sim_kwargs:dict=None):
        plot = dataPlot(["loss"])
        loss_list = []
        with rich.progress.Progress() as pbar:
            task1 = pbar.add_task("", total=total_episode)
            for i in range(n_epoch):
                pbar.tasks[task1].description = "epoch {0} of {1}".format(i+1, n_epoch)
                pbar.tasks[task1].completed = 0
                [self.pull(idx) for idx in range(self.n_process)] # mp agents pull parameters from main agent
                inq, outq = self._init_train(total_episode)
                [w.start() for w in self.workers]
                count = 0
                while count<self.n_process: # mp workers running
                    if outq.qsize(): # new result in outq
                        if (obj := outq.get()) is not None:
                            td1 = obj[0]
                            obss = td1["obss"].to(self.main_device)
                            best_imp_ = obj[1].to(self.main_device).detach()
                            imp = self.model(obss)
                            loss = torch.nn.functional.mse_loss(imp, best_imp_)
                            self.opt.zero_grad()
                            loss.backward()
                            self.opt.step()
                            loss_list.append(loss.item())
                            if pbar.tasks[task1].completed > plot.min_window_size:
                                plot.set_data((loss_list))
                            pbar.update(task1, advance=1)
                        else:
                            count += 1
                [w.join() for w in self.workers]
                np.savez(folder+"log.npz", 
                        loss = np.array(loss_list)
                        )
                plot.save_fig(folder+"log.png")
                torch.save(self.model.state_dict(), folder+"implusesModel.ptd")
                self._reset_workers()
        return
    
    def debug(self):
        obj = self.debug1p()
        td1 = obj[0]
        states = td1["states"].to(self.main_device)
        obss = td1["obss"].to(self.main_device)
        best_imp_ = obj[1].to(self.main_device).detach()
        imp = self.model(obss)
        loss = torch.nn.functional.mse_loss(imp, best_imp_)
        return td1, best_imp_, loss