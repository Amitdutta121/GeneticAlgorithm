import math
import random

from Options import Options
import EvrpGraph as Graph


class Evaluator:

    def evaluate(self, chromosome):
        return Options.EVALUATOR(chromosome)
    def evrpProblem(self, chromosome):
        Graph.clearTour()
        for i in range(0, len(chromosome), 3):
            action = chromosome[i:i + 3]
            # convert binary to decimal
            action = int(''.join(map(str, action)), 2)
            Graph.addToTourBasedOnHuristics(action)
        Graph.addSourceNodeToAllTours()
        maxCostTour = Graph.getTheMaxCostTour()
        return 1/(maxCostTour+1)



