from tree.tree import GST
gst = GST(10, 10, 6, 6, 3)

from tree.tree import stateDict
from agent.agent import debugAgent
from env.propagator import debugPropagator
sd = stateDict(6, 6, 3)
gst.reset(sd)
A = debugAgent()
P = debugPropagator()

gst.step(A, P)
gst.select()
