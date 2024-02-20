from trainer.trainer import treeTrainer
from agent.agent import debugAgent
from env.env import debugEnv

pop = 10
max_gen = 200
env = debugEnv(pop, max_gen)
agent = debugAgent(actor_hiddens=[256]*4, critic_hiddens=[256]*4, actor_lr=1E-4, critic_lr=2E-3)
T = treeTrainer(env, agent, gamma=1)
T.agent.load("../model/check_point9.ptd")
# ds = T.simulate()
_ = T.test(1)