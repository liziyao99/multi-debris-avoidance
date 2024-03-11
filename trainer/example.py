from trainer.trainer import treeTrainer
from agent.agent import normalDistAgent
from env.env import treeEnvB
from env.propagatorB import CWPropagatorB



class CWTreeTrainer(treeTrainer):
    def __init__(self, population=1024, max_gen=3600, gamma=1, batch_size=1280, 
                 actor_hiddens=[512]*8, critic_hiddens=[512]*8, 
                 actor_lr=1E-5, critic_lr=5E-4, 
                 device="cpu") -> None:
        propB = CWPropagatorB(device=device)
        env = treeEnvB.from_propagator(propB, population=population, max_gen=max_gen, device=device)
        action_bound = 6e-2
        agent = normalDistAgent(obs_dim=propB.obs_dim, action_dim=propB.action_dim,
                                actor_hiddens=actor_hiddens, critic_hiddens=critic_hiddens, 
                                action_lower_bound=-action_bound, action_upper_bound=action_bound, 
                                actor_lr=actor_lr, critic_lr=critic_lr, device=device)
        super().__init__(env, agent, gamma, batch_size)