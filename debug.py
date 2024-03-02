from trainer.trainer import treeTrainer
from agent.agent import debugTreeAgent
from env.env import debugTreeEnv

def debug1(pop=20):
    max_gen = 200
    env = debugTreeEnv(pop, max_gen)
    agent = debugTreeAgent(actor_hiddens=[256]*4, critic_hiddens=[256]*4, actor_lr=1E-4, critic_lr=2E-3)
    T = treeTrainer(env, agent, gamma=1)
    # T.agent.load("../model/check_point3.ptd")
    # ds = T.simulate()
    # _ = T.test(1)
    T.train(n_episode=1)
    # _ = T.test(t_max=1,)
    return

def debug2():
    from trainer.trainer import singleTrainer
    from agent.agent import PPOClipAgent
    from env.env import debugSingleEnv

    env = debugSingleEnv(200)
    agent = PPOClipAgent(actor_hiddens=[256]*4, critic_hiddens=[256]*4, actor_lr=1E-4, critic_lr=2E-3)
    T = singleTrainer(env, agent)

    d = T.simulate()
    T.agent.update(d)


if __name__ == "__main__":
    import time
    t0 = time.time()
    g = debug1(100)
    t1 = time.time()
    print(t1-t0)
    print(g)