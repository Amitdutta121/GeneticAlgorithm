import random

from Evaluator import Evaluator
from Options import Options


class Population:
    def __init__(self, popSize):
        self.individuals = [None] * popSize

    def add_individual(self, individual):
        for i in range(len(self.individuals)):
            if self.individuals[i] is None:
                self.individuals[i] = individual
                break

    def calculate_fitness(self):
        fitness_values = []
        for i in range(len(self.individuals)):
            single_fitness = 0
            if self.individuals[i] is None:
                single_fitness = 0
            else:
                single_fitness = Evaluator().evaluate(self.individuals[i].get_genes())
                fitness_values.append(single_fitness)
                self.individuals[i].set_fitness(single_fitness)

        return fitness_values

    def roulette_wheel_selection(self, fitness_values):
        total_fitness = sum(fitness_values)
        random_value = random.random() * total_fitness
        cumulative_sum = 0

        for i in range(len(self.individuals)):
            cumulative_sum += fitness_values[i]
            if cumulative_sum >= random_value:
                return self.individuals[i]

        # Fallback to the last individual
        return self.individuals[-1]

    def mutate_population(self):
        for i in range(len(self.individuals)):
            genes = self.individuals[i].get_genes()
            for j in range(len(genes)):
                if random.random() < Options.P_MUT:
                    genes[j] = 1 - genes[j]  # Flip the bit

    def get_individual(self, index):
        return self.individuals[index]

    def get_best_individual(self):
        best_individual = self.individuals[0]
        best_fitness = Evaluator().evaluate(best_individual.get_genes())

        for i in range(1, len(self.individuals)):
            fitness = Evaluator().evaluate(self.individuals[i].get_genes())
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = self.individuals[i]
        return best_individual
