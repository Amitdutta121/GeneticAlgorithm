import random
from Options import Options
import numpy as np

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0

    def get_genes(self):
        return self.genes

    def set_fitness(self, fitness):
        self.fitness = fitness

    def PMX_crossover_final(self, parent1, parent2):
        '''
        parent1 and parent2 are 1D np.array
        '''
        rng = np.random.default_rng()

        cutoff_1, cutoff_2 = np.sort(rng.choice(np.arange(len(parent1) + 1), size=2, replace=False))

        def PMX_one_offspring(p1, p2):
            offspring = np.zeros(len(p1), dtype=p1.dtype)

            # Copy the mapping section (middle) from parent1
            offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]

            # copy the rest from parent2 (provided it's not already there
            for i in np.concatenate([np.arange(0, cutoff_1), np.arange(cutoff_2, len(p1))]):
                candidate = p2[i]
                while candidate in p1[cutoff_1:cutoff_2]:  # allows for several successive mappings
                    candidate = p2[np.where(p1 == candidate)[0][0]]
                offspring[i] = candidate
            return offspring

        offspring1 = PMX_one_offspring(parent1, parent2)
        offspring2 = PMX_one_offspring(parent2, parent1)

        return offspring1, offspring2

    def one_point_crossover(self, other):
        crossover_point = random.randint(0, len(self.genes))

        child1_genes = self.genes[:crossover_point] + other.get_genes()[crossover_point:]
        child2_genes = other.get_genes()[:crossover_point] + self.genes[crossover_point:]

        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)

        return [child1, child2]

    def pmx_crossover(self, other):
        crossover_point = random.randint(0, len(self.genes))

        parent1 = np.array(self.genes)
        parent2 = np.array(other.get_genes())
        child1_genes, child2_genes = self.PMX_crossover_final(parent1, parent2)

        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)

        return [child1, child2]

    def mutate(self):
        if random.random() < Options.P_MUT:
            # Swap mutation: Randomly select two positions and swap the values
            mutation_positions = random.sample(range(len(self.genes)), 2)
            self.genes[mutation_positions[0]], self.genes[mutation_positions[1]] = (
                self.genes[mutation_positions[1]],
                self.genes[mutation_positions[0]],
            )

