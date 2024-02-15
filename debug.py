from trainer.trainer import treeTrainer
from agent.agent import debugAgent
from env.env import debugEnv
from data.dicts import concat_dicts

pop = 100
max_gen = 200
env = debugEnv(pop, max_gen)
agent = debugAgent()
T = treeTrainer(env, agent)
dicts = T.simulate()
d = concat_dicts(dicts)
T.update(d)