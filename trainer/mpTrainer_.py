import numpy as np
import torch.multiprocessing as mp
import rich.progress
import typing

from data.buffer import replayBuffer
import data.dicts as D
import env.propagators
import agent.agent as A
import env.propagators.hirearchicalPropagator
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
        print(f"{self.name} done")

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
    

class mpH2Worker(mpWorker):
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

    def simulate(self, select_itr:int, select_size:int):
        trans_dict, _, v = self.trainer.treeSim(select_itr, select_size, train_h2a=False)
        return trans_dict, v
    
class mpH2Trainer(mpTrainer):
    def __init__(self, n_process:int, buffer:replayBuffer, n_debris:int, agentArgs:dict, 
                 select_itr=10, select_size=32, batch_size=2048, main_device="cuda", mode="default"):
        if mode not in ["default", "alter"]:
            raise ValueError("mode must be \"default\" or \"alter\"")
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
        p = env.propagators.hirearchicalPropagator.H2CWDePropagator(self.n_debris, device=device)
        agent = A.H2Agent(obs_dim=p.obs_dim,
                          h1obs_dim=p.obs_dim,
                          h2obs_dim=6, 
                          h1out_dim=p.h1_action_dim, 
                          h2out_dim=p.h2_action_dim, 
                          device=device,
                          **self.agentArgs)
        if self.mode=="default":
            return trainer.trainer.H2TreeTrainer(p, agent)
        elif self.mode=="alter":
            _, tutor = trainer.trainer.CWPTT(self.n_debris, agent.device, "../model/planTrack3.ptd")
            return trainer.trainer.H2TreeTrainerAlter(p, agent, tutor)
    
    def _reset_workers(self):
        for w in self.workers:
            w.close()
            del w
        self.workers = []
        for i in range(self.n_process):
            trainer = self._init_trainer()
            self.workers.append(mpH2Worker(self.select_itr, self.select_size, trainer, f"worker{i}"))

    def pull(self, idx):
        self.agents[idx].h1a.load_state_dict(self.main_agent.h1a.state_dict())
        self.agents[idx].h1c.load_state_dict(self.main_agent.h1c.state_dict())
        self.agents[idx].h2a.load_state_dict(self.main_agent.h2a.state_dict())
        if self.mode=="alter":
            self.workers[idx].trainer.tutor.planner.load_state_dict(self.main_trainer.tutor.planner.state_dict())
            self.workers[idx].trainer.tutor.critic.load_state_dict(self.main_trainer.tutor.critic.state_dict())
            self.workers[idx].trainer.tutor.tracker.load_state_dict(self.main_trainer.tutor.tracker.state_dict())
    
    def train(self, n_epoch:int, total_episode:int, folder="../model/", sim_kwargs:dict=None):
        if sim_kwargs is not None:
            for w in self.workers: w.sim_kwargs = sim_kwargs
        plot = dataPlot(["true_values", "critic_loss", "ddpg_loss", "mc_loss"])
        with rich.progress.Progress() as pbar:
            task1 = pbar.add_task("", total=total_episode)
            critic_loss, ddpg_loss, mc_loss, true_values = [], [], [], []
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
                                Q_l, mc_l, ddpg_l = self.main_agent.h1update(dict)
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
                        self.main_agent.h1update(trans_dict_b)
                [w.join() for w in self.workers]
                np.savez(folder+"log.npz", 
                        true_values = true_values,
                        critic_loss = critic_loss,
                        ddpg_loss = ddpg_loss,
                        mc_loss = mc_loss
                        )
                plot.save_fig(folder+"log.png")
                self.main_trainer.agent.save(folder+f"check_point{i}.ptd")
                self._reset_workers()
        return
    
    def debug(self):
        trans_dict, v = self.debug1p()
        trans_dicts = D.split_dict(trans_dict, batch_size=self.batch_size)
        _Q, _mc, _ddpg = [], [], []
        for dict in trans_dicts:
            Q_l, mc_l, ddpg_l = self.main_agent.h1update(dict)
            _Q.append(Q_l)
            _mc.append(mc_l)
            _ddpg.append(ddpg_l)
        return v, np.mean(_Q), np.mean(_mc), np.mean(_ddpg)