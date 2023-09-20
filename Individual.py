import random

from Options import Options


class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0

    def get_genes(self):
        return self.genes

    def set_fitness(self, fitness):
        self.fitness = fitness

    def one_point_crossover(self, other):
        crossover_point = random.randint(0, len(self.genes))

        child1_genes = self.genes[:crossover_point] + other.get_genes()[crossover_point:]
        child2_genes = other.get_genes()[:crossover_point] + self.genes[crossover_point:]

        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)

        return [child1, child2]

    def mutate(self):
        for i in range(len(self.genes)):
            if random.random() < Options.P_MUT:
                self.genes[i] = 1 - self.genes[i]
