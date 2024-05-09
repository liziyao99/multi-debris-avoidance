import matplotlib.pyplot as plt
import collections

from tree.geneticTree import stateIndexNode

class treeRenderer:
    def __init__(self, population:int, max_gen:int):
        self.population = population
        self.max_gen = max_gen

        self.__qr = collections.deque() # queue for render
        self.__qc = collections.deque() # queue for children

    def nodePosMap(self, node:stateIndexNode):
        if node.depth==0: # root
            return ((self.population-1)/2, 0.)
        else:
            return (self.population-len(self.__qr)-1, node.depth)

    def nodeColorMap(self, node:stateIndexNode):
        return 'b'
    
    def edgeColorMap(self, node:stateIndexNode):
        return 'b'
    
    def render(self, root:stateIndexNode):
        nodePos = []
        parentPos = [None,]
        nodeColor = []
        edgeColor = []
        self.__qr.clear()
        self.__qc.clear()
        self.__qr.append(root)
        while len(self.__qr):
            node = self.__qr.popleft()
            pos = self.nodePosMap(node)
            nodePos.append(pos)
            nodeColor.append(self.nodeColorMap(node))
            edgeColor.append(self.edgeColorMap(node))
            self.__qc.extend(node.children)
            parentPos += [pos]*len(node.children)
            if not len(self.__qr):
                self.__qr.extend(self.__qc)
                self.__qc.clear()
        # TODO