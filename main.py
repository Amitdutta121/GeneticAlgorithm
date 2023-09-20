import random

from Evaluator import Evaluator
from Individual import Individual
from Options import Options
from Population import Population
import matplotlib.pyplot as plt


class GeneticAlgorithm:

    def __init__(self):
        self.average_fitness = []
        self.max_fitness = []

    def clear_fitness(self):
        self.average_fitness = []
        self.max_fitness = []

    def simple_genetic_algorithm(self):
        random.seed(Options.RANDOM_SEED)
        self.clear_fitness()
        population = self.initialize_population(Options.POPULATION_SIZE)
        generation = 0

        while generation < Options.MAX_GENERATION:
            fitness_values = population.calculate_fitness()
            new_population = Population(Options.POPULATION_SIZE)

            max_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)

            print(f"Generation {generation}:")
            print(f"Max Fitness: {max_fitness}")
            print(f"Average Fitness: {avg_fitness}")
            self.average_fitness.append(avg_fitness)
            self.max_fitness.append(max_fitness)

            for i in range(0, Options.POPULATION_SIZE, 2):
                parent1 = population.roulette_wheel_selection(fitness_values)
                parent2 = population.roulette_wheel_selection(fitness_values)
                children1 = None
                children2 = None
                if random.random() < Options.P_CROSS:  # Perform crossover with probability
                    children = parent1.one_point_crossover(parent2)
                    children1 = children[0]
                    children2 = children[1]
                    new_population.add_individual(children[0])
                    new_population.add_individual(children[1])
                else:
                    # If no crossover, simply copy parents to the next generation
                    new_population.add_individual(parent1)
                    new_population.add_individual(parent2)

                if children1 is not None and children2 is not None:
                    children1.mutate()
                    children2.mutate()

            population = new_population

            generation += 1

        best_individual = population.get_best_individual()
        print(f"Best individual after {Options.MAX_GENERATION} generations:")
        print(best_individual.get_genes())
        ones_count = Evaluator().evaluate(best_individual.get_genes())
        print(f"Number of 1's: {ones_count}")

    def print_population(self, population):
        print("=====================================")
        for i in range(len(population.individuals)):
            print(f"Individual {i}: {population.individuals[i].get_genes()}")
        print("=====================================")

    def plot_average_fitness_and_max_fitness(self):
        plt.plot(self.average_fitness, label="Average Fitness")
        plt.plot(self.max_fitness, label="Max Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

    def chc_genetic_algorithm(self):
        random.seed(Options.RANDOM_SEED)
        self.clear_fitness()
        current_population = self.initialize_population(Options.POPULATION_SIZE * Options.CHC_LAMDA)
        generation = 0

        while generation < Options.MAX_GENERATION:
            fitness_values = current_population.calculate_fitness()
            new_population = Population(Options.POPULATION_SIZE * Options.CHC_LAMDA)

            max_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)

            print(f"Generation {generation}:")
            print(f"Max Fitness: {max_fitness}")
            print(f"Average Fitness: {avg_fitness}")
            self.average_fitness.append(avg_fitness)
            self.max_fitness.append(max_fitness)

            for i in range(0, Options.POPULATION_SIZE, 2):
                parent1 = current_population.roulette_wheel_selection(fitness_values)
                parent2 = current_population.roulette_wheel_selection(fitness_values)
                new_population.add_individual(parent1)
                new_population.add_individual(parent2)
                children1 = None
                children2 = None
                if random.random() < Options.P_CROSS:  # Perform crossover with probability
                    children = parent1.one_point_crossover(parent2)
                    children1 = children[0]
                    children2 = children[1]
                    new_population.add_individual(children[0])
                    new_population.add_individual(children[1])

                if children1 is not None and children2 is not None:
                    children1.mutate()
                    children2.mutate()

            # CHC
            new_population.calculate_fitness()

            new_population.individuals.sort(key=lambda x: x.fitness if x is not None else float('-inf'), reverse=True)
            new_population.individuals = new_population.individuals[:Options.POPULATION_SIZE]

            current_population = new_population

            generation += 1

        best_individual = current_population.get_best_individual()
        print(f"Best individual after {Options.MAX_GENERATION} generations:")
        print(best_individual.get_genes())
        result = Evaluator().evaluate(best_individual.get_genes())
        print(f"Best Result: {result}")

    def initialize_population(self, size):
        population = Population(size)

        for i in range(size):
            genes = [random.choice((0, 1)) for _ in range(Options.CHROMOSOME_LENGTH)]
            individual = Individual(genes)
            population.add_individual(individual)
        return population


# Include your Population, Individual, Options, Evaluator, and RandomGenerator classes here.

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.simple_genetic_algorithm()
    ga.plot_average_fitness_and_max_fitness()

    Options.P_MUT = 0.05
    Options.P_CROSS = 0.99
    ga.chc_genetic_algorithm()
    ga.plot_average_fitness_and_max_fitness()
