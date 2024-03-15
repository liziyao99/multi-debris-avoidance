import numpy as np
import torch.multiprocessing as mp
import rich.progress
import typing

from data.buffer import replayBuffer
import data.dicts as D
from agent.agent import rlAgent
from env.env import treeEnvB
from trainer.trainer import treeTrainer
from plotting.dataplot import logPlot, dataPlot


class mpTreeWorker(mp.Process):
    '''
        NOTE: a Process can not start twice.
    '''
    def __init__(self, trainer:treeTrainer, name:str, global_agent:rlAgent=None,
                 inq:mp.Queue=None, outq:mp.Queue=None, sim_kwargs:dict={}) -> None:
        mp.Process.__init__(self)
        self.trainer = trainer
        self.name = name
        self.inq = inq
        self.outq = outq
        self.sim_kwargs = sim_kwargs
        self.global_agent = global_agent # TODO: cuda share memory

    def simulate(self, **kwargs):
        return self.trainer.simulate(**kwargs)

    def run(self):
        print(f"{self.name} running. agent device:{self.agent.device}")
        while (flag := self.inq.get()) is not None:
            # self.pull(self.global_agent)
            trans_dicts = self.simulate(**self.sim_kwargs)
            trans_dict = D.concat_dicts(trans_dicts)
            self.outq.put(trans_dict)
        self.outq.put(None)
        print(f"{self.name} done")

    def setQ(self, inq:mp.Queue=None, outq:mp.Queue=None):
        self.inq = inq if inq is not None else self.inq
        self.outq = outq if outq is not None else self.outq

    def pull(self, global_agent:rlAgent):
        self.agent.actor.load_state_dict(global_agent.actor.state_dict())
        self.agent.critic.load_state_dict(global_agent.critic.state_dict())

    @property
    def agent(self):
        return self.trainer.agent
    
    @property
    def env(self):
        return self.trainer.env
    
    @property
    def buffer(self):
        return self.trainer.buffer

class mpTreeTrainer:
    def __init__(self, n_process:int, trainerType=treeTrainer, trainerArgs:dict={}, main_device="cuda"):
        mainTrainerArgs = {}
        trainerArgs["device"] = "cpu" # bug (Tensor elements become 0) encountered when subprocess run on gpu
        for key in trainerArgs.keys():
            if key!="device":
                mainTrainerArgs[key] = trainerArgs[key]
        mainTrainerArgs["device"] = main_device
        self.main_trainer = trainerType(**mainTrainerArgs)
        # self.main_agent.share_memory()
        # TODO: share cuda memory

        self.n_process = n_process
        self.trainerType = trainerType
        self.trainerArgs = trainerArgs
        self.workers = []
        self.__reset_workers()

    @property
    def main_env(self):
        return self.main_trainer.env

    @property
    def main_agent(self):
        return self.main_trainer.agent
    
    @property
    def main_buffer(self):
        return self.main_trainer.buffer
    
    @property
    def envs(self) -> typing.List[treeEnvB]:
        return [w.env for w in self.workers]
    
    @property
    def agents(self) -> typing.List[rlAgent]:
        return [w.agent for w in self.workers]
    
    @property
    def buffers(self) -> typing.List[replayBuffer]:
        return [w.buffer for w in self.workers]
    
    def __reset_workers(self):
        for w in self.workers:
            w.close()
            del w
        self.workers = []
        for i in range(self.n_process):
            trainer = self.trainerType(**self.trainerArgs)
            self.workers.append(mpTreeWorker(trainer, f"worker{i}", 
                                         # self.main_agent
                                ))
    
    def __init_train(self, total:int):
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
        if sim_kwargs is not None:
            for w in self.workers: w.sim_kwargs = sim_kwargs
        plot = dataPlot(["total_rewards", "actor_loss", "critic_loss"])
        with rich.progress.Progress() as pbar:
            task1 = pbar.add_task("", total=total_episode)
            actor_loss, critic_loss, total_rewards = [], [], []
            for i in range(n_epoch):
                pbar.tasks[task1].description = "epoch {0} of {1}".format(i+1, n_epoch)
                pbar.tasks[task1].completed = 0
                [self.pull(idx) for idx in range(self.n_process)] # mp agents pull parameters from main agent
                inq, outq = self.__init_train(total_episode)
                [w.start() for w in self.workers]
                count = 0
                while count<self.n_process: # mp workers running
                    if outq.qsize(): # new result in outq
                        if (trans_dict := outq.get()) is not None:
                            trans_dicts = D.batch_dict(trans_dict, batch_size=self.main_trainer.batch_size)
                            als, cls = [], []
                            for d in trans_dicts:
                                al, cl = self.main_trainer.update(d)
                                als.append(al)
                                cls.append(cl)
                            actor_loss.append(np.mean(als))
                            critic_loss.append(np.mean(cls))
                            tr = np.sum(trans_dict["rewards"])/self.main_trainer.tree.population
                            total_rewards.append(tr)
                            if pbar.tasks[task1].completed > plot.min_window_size:
                                plot.set_data((total_rewards, actor_loss, critic_loss))
                            pbar.update(task1, advance=1)
                        else:
                            count += 1
                    elif self.main_buffer.size>=self.main_buffer.minimal_size: # train using buffer
                        trans_dict_batch = self.main_buffer.sample()
                        self.main_trainer.update(trans_dict_batch)
                [w.join() for w in self.workers]
                np.savez(folder+"log.npz", 
                        total_rewards = total_rewards,
                        actor_loss = actor_loss,
                        critic_loss = critic_loss
                        )
                plot.save_fig(folder+"log.png")
                self.main_trainer.agent.save(folder+f"check_point{i}.ptd")
                self.__reset_workers()
        return
    
    def debug(self):
        trans_dicts = self.main_trainer.simulate()
        trans_dict = D.concat_dicts(trans_dicts)
        trans_dicts = D.batch_dict(trans_dict, batch_size=self.main_trainer.batch_size)
        for d in trans_dicts:
            self.main_agent.update(d)
