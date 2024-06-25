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
from trainer.hierarchicalTrainer import H2Trainer
from plotting.dataplot import logPlot, dataPlot
import trainer.trainer


class mpH2Worker(mp.Process):
    '''
        NOTE: a Process can not start twice.
    '''
    def __init__(self, trainer:H2Trainer, name:str,
                 inq:mp.Queue=None, outq:mp.Queue=None, sim_kwargs:dict={}) -> None:
        mp.Process.__init__(self)
        self.trainer = trainer
        self.name = name
        self.inq = inq
        self.outq = outq
        self.sim_kwargs = sim_kwargs

    def simulate(self, **kwargs):
        trans_dict, _ = self.trainer.offPolicySim(**kwargs)
        return trans_dict

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
        return self.trainer.hAgent

class mpH2Trainer:
    def __init__(self, n_process:int, trainerCls, trainerArgs:dict, buffer:replayBuffer, main_device="cuda", worker_batch_size=16):
        trainerArgs["device"] = "cpu" # bug: (Tensor elements become 0) encountered when subprocess run on gpu
        self.trainerCls = trainerCls
        self.trainerArgs = trainerArgs
        self.main_device = main_device
        self.worker_batch_size = worker_batch_size
        self.main_trainer = self._init_trainer({"device": main_device})

        self.n_process = n_process
        self.workers = []
        self._reset_workers()
        self.buffer = buffer

    def _init_trainer(self, trainerArgs:dict={}) -> H2Trainer:
        args = self.trainerArgs.copy()
        args |= trainerArgs
        return self.trainerCls(**args)
    
    def _reset_workers(self):
        for w in self.workers:
            w.close()
            del w
        self.workers = []
        for i in range(self.n_process):
            trainer = self._init_trainer()
            sim_kwargs = {
                "states_num": self.worker_batch_size
            }
            self.workers.append(mpH2Worker(trainer, f"worker{i}", sim_kwargs=sim_kwargs))

    @property
    def main_agent(self):
        return self.main_trainer.hAgent
    
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
        self.agents[idx].copy(self.main_agent)
    
    def train(self, n_epoch:int, total_episode:int, folder="../model/", sim_kwargs:dict=None):
        if sim_kwargs is not None:
            for w in self.workers: w.sim_kwargs = sim_kwargs
        keys = ["total_rewards"] # + self.main_trainer.loss_keys
        plot = dataPlot(keys)
        log_dict = dict(zip( keys, [[] for _ in range(len(keys))] ))
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
                            trans_dict = obj
                            trans_dicts = D.deBatch_dict(trans_dict)
                            self.buffer.from_dicts(trans_dicts)
                            for d in trans_dicts:
                                loss_log = self.main_agent[0].update(d)
                            total_reward = torch.sum(trans_dict["rewards"], dim=0).mean().item()
                            log_dict["total_rewards"].append(total_reward)
                            if pbar.tasks[task1].completed > plot.min_window_size:
                                plot.set_data(list(log_dict.values()))
                            pbar.update(task1, advance=1)
                        else:
                            count += 1
                    elif self.buffer.size>=self.buffer.minimal_size: # train using buffer
                        trans_dict_b = self.buffer.sample()
                        self.main_agent[0].update(trans_dict_b)
                [w.join() for w in self.workers]
                np.savez(folder+"log.npz", 
                        **log_dict
                        )
                plot.save_fig(folder+"log.png")
                #self.main_trainer.agent.save(folder+f"check_point{i}.ptd")
                self.main_trainer.hAgent.save(folder+"h2_check_point/")
                self._reset_workers()
        return
    
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
