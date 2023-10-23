import random

import Objective
from Evaluator import Evaluator
from Individual import Individual
from Options import Options
from Population import Population
import matplotlib.pyplot as plt


# Assuming you have a function to read TSP data from a file
def read_tsp_data(file_name):
    cities = []
    with open(file_name, 'r') as file:
        read_coordinates = False
        for line in file:
            if "NODE_COORD_SECTION" in line:
                read_coordinates = True
                continue
            elif "EOF" in line:
                break
            if read_coordinates:
                parts = line.strip().split()
                city_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                cities.append((city_id, x, y))
    return cities


class GeneticAlgorithm:

    def __init__(self):
        self.average_fitness = []
        self.max_fitness = []
        self.min_fitness = []
        self.objective = []
        self.max_objective = []
        self.tsp_data = read_tsp_data(Options.FILE_NAME)

    def clear_fitness(self):
        self.average_fitness = []
        self.max_fitness = []
        self.min_fitness = []
        self.objective = []
        self.max_objective = []

    def simple_genetic_algorithm(self):
        # random.seed(Options.RANDOM_SEED)
        self.clear_fitness()
        population = self.initialize_population(Options.POPULATION_SIZE)
        generation = 0

        while generation < Options.MAX_GENERATION:
            fitness_values = population.calculate_fitness()
            new_population = Population(Options.POPULATION_SIZE)

            max_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)
            obj = Options.OBJECTIVE(max_fitness)
            min_f = min(fitness_values)
            max_obj = Options.OBJECTIVE(min_f)

            print(f"Generation {generation}:")
            print(f"Max Fitness: {max_fitness}")
            print(f"Average Fitness: {avg_fitness}")
            self.average_fitness.append(avg_fitness)
            self.max_fitness.append(max_fitness)
            self.objective.append(obj)
            self.min_fitness.append(min_f)
            self.max_objective.append(max_obj)

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
        return self.average_fitness, self.max_fitness, self.min_fitness, self.objective, self.max_objective

    def print_population(self, population):
        print("=====================================")
        for i in range(len(population.individuals)):
            print(f"Individual {i}: {population.individuals[i].get_genes()}")
        print("=====================================")

    def plot_average_fitness_and_max_fitness(self, algo_type="simple"):
        plt.plot(self.average_fitness, label="Average Fitness")
        plt.plot(self.max_fitness, label="Max Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        # set title
        if algo_type == "simple":
            plt.title("Average Fitness and Max Fitness (Simple Genetic Algorithm)")
        else:
            plt.title("Average Fitness and Max Fitness (CHC Genetic Algorithm)")
        plt.legend()
        plt.show()

    def chc_genetic_algorithm(self, max_generations_without_improvement=10):
        # random.seed(Options.RANDOM_SEED)
        self.clear_fitness()
        current_population = self.initialize_population(Options.POPULATION_SIZE * Options.CHC_LAMDA)
        generation = 0
        prev_avg_fitness = None
        generations_without_improvement = 0

        while generation < Options.MAX_GENERATION:
            fitness_values = current_population.calculate_fitness()
            new_population = Population(Options.POPULATION_SIZE * Options.CHC_LAMDA)

            max_fitness = max(fitness_values)
            avg_fitness = sum(fitness_values) / len(fitness_values)
            obj = Options.OBJECTIVE(avg_fitness)
            min_f = min(fitness_values)
            max_obj = Options.OBJECTIVE(max_fitness)

            print(f"Generation {generation}:")
            print(f"Max Fitness: {max_fitness}")
            print(f"Average Fitness: {avg_fitness}")
            self.average_fitness.append(avg_fitness)
            self.max_fitness.append(max_fitness)
            self.objective.append(obj)
            self.min_fitness.append(min_f)
            self.max_objective.append(max_obj)

            # if prev_avg_fitness is not None and avg_fitness <= prev_avg_fitness:
            #     generations_without_improvement += 1
            #     if generations_without_improvement >= max_generations_without_improvement:
            #         print(f"Terminating due to lack of improvement for {generations_without_improvement} generations.")
            #         break
            # else:
            #     generations_without_improvement = 0

            prev_avg_fitness = avg_fitness

            for i in range(0, Options.POPULATION_SIZE, 2):
                parent1 = current_population.roulette_wheel_selection(fitness_values)
                parent2 = current_population.roulette_wheel_selection(fitness_values)
                new_population.add_individual(parent1)
                new_population.add_individual(parent2)
                children1 = None
                children2 = None
                if random.random() < Options.P_CROSS:  # Perform crossover with probability
                    children = parent1.pmx_crossover(parent2)
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
        print(f"Objective: {Options.OBJECTIVE(Evaluator().evaluate(best_individual.get_genes()))}")
        result = Evaluator().evaluate(best_individual.get_genes())
        print(f"Best Result: {result}")
        return self.average_fitness, self.max_fitness, self.min_fitness, self.objective, self.max_objective

    def initialize_population(self, size):
        population = Population(size)

        for i in range(size):
            # Randomly shuffle the city order
            random.shuffle(self.tsp_data)
            genes = [city[0] for city in self.tsp_data]  # Extract city IDs in shuffled order
            individual = Individual(genes)
            population.add_individual(individual)
        return population


def plot_final_runs(runs_and_avg_fitness, runs_and_max_fitness, runs_and_min_fitness, runs_and_objective,
                    algo_type="SGA",
                    evaluator="deJongFunction1", withObjective=False):
    avg_fitness = [sum(x) / len(x) for x in zip(*runs_and_avg_fitness)]
    max_fitness = [sum(x) / len(x) for x in zip(*runs_and_max_fitness)]
    min_fitness = [sum(x) / len(x) for x in zip(*runs_and_min_fitness)]
    obj = [sum(x) / len(x) for x in zip(*runs_and_objective)]
    if withObjective:
        plt.plot(obj, label="Max Objective")
    plt.xlabel("Generation")
    plt.plot(avg_fitness, label="Average Fitness")
    plt.plot(max_fitness, label="Max Fitness")
    plt.plot(min_fitness, label="Min Fitness")

    y_axis = "Fitness"
    if withObjective:
        y_axis = "Objective"
    else:
        y_axis = "Fitness"
    plt.ylabel(y_axis)
    # set title
    if algo_type == "SGA":
        plt.title(f"(Simple Genetic Algorithm) for {evaluator}")
    else:
        plt.title(f"Average and Max Fitness (CHC Genetic Algorithm) for {evaluator}")
    plt.legend()
    plt.show()


def plot_final_runs_objective(runs_avg_objective, runs_max_objective,
                              algo_type="SGA",
                              evaluator="deJongFunction1"):
    avg_obj = [sum(x) / len(x) for x in zip(*runs_avg_objective)]
    max_obj = [sum(x) / len(x) for x in zip(*runs_max_objective)]
    plt.xlabel("Generation")
    plt.plot(avg_obj, label="Average Objective")
    plt.plot(max_obj, label="Max Objective")

    y_axis = "Objective"
    plt.ylabel(y_axis)
    # set title
    if algo_type == "SGA":
        plt.title(f"(Simple Genetic Algorithm) for {evaluator}")
    else:
        plt.title(f"Average and Max Objective (CHC Genetic Algorithm) for {evaluator}")
    plt.legend()
    plt.show()


import multiprocessing

multiprocessing.log_to_stderr()


def run_genetic_algorithm(run_id, results_queue):
    try:
        ga = GeneticAlgorithm()
        Options.EVALUATOR = Evaluator().tsp_fitness
        Options.P_MUT = 0.9
        Options.P_CROSS = 0.9
        Options.CHROMOSOME_LENGTH = 318
        Options.OBJECTIVE = Objective.dejongReverse
        average_fitness, max_fitness, min_fitness, obj, m_obj = ga.chc_genetic_algorithm()
        results_queue.put((average_fitness, max_fitness, min_fitness, obj, m_obj))
        print(f"Run {run_id} completed.")
    except Exception as e:
        print(f"Error in run_genetic_algorithm: {e}")


if __name__ == "__main__":
    total_runs = Options.TOTAL_RUNS

    # Create a multiprocessing Queue to collect results
    results_queue = multiprocessing.Queue()

    # Create a list to store process objects
    processes = []

    # Launch multiple processes to run the genetic algorithm
    for i in range(total_runs):
        process = multiprocessing.Process(target=run_genetic_algorithm, args=(i, results_queue))
        process.start()
        processes.append(process)

    print("ALL PROCESSES STARTED")

    try:
        # Wait for all processes to complete
        for process in processes:
            process.join(timeout=10)
    except Exception as e:
        print(f"Error waiting for processes to complete: {e}")

    print("ALL PROCESSES COMPLETED")

    # Collect results from the Queue
    runs_and_avg_fitness = list()
    runs_and_max_fitness = list()
    runs_and_min_fitness = list()
    runs_and_objective = list()
    runs_max_objective = list()



    for _ in range(total_runs):
        avg, max_f, min_f, obj, m_obj = results_queue.get()
        runs_and_avg_fitness.append(avg)
        runs_and_max_fitness.append(max_f)
        runs_and_min_fitness.append(min_f)
        runs_and_objective.append(obj)
        runs_max_objective.append(m_obj)

    plot_final_runs(runs_and_avg_fitness, runs_and_max_fitness, runs_and_min_fitness, runs_and_objective,
                    algo_type="CHC",
                    evaluator="lin318")

    plot_final_runs_objective(runs_and_objective, runs_max_objective, algo_type="CHC", evaluator="lin318")

# if __name__ == "__main__":
#     ga = GeneticAlgorithm()
#
#     runs_and_avg_fitness = list()
#     runs_and_max_fitness = list()
#     runs_and_min_fitness = list()
#     runs_and_objective = list()
#     runs_max_objective = list()
#
#     for i in range(Options.TOTAL_RUNS):
#         # random.seed(Options.RANDOM_SEED)
#         Options.EVALUATOR = Evaluator().tsp_fitness
#         Options.P_MUT = 0.9
#         Options.P_CROSS = 0.9
#         Options.CHROMOSOME_LENGTH = 52
#         Options.OBJECTIVE = Objective.dejongReverse
#         average_fitness, max_fitness, min_fitness, obj, m_obj = ga.chc_genetic_algorithm()
#         runs_and_avg_fitness.append(average_fitness)
#         runs_and_max_fitness.append(max_fitness)
#         runs_and_objective.append(obj)
#         runs_and_min_fitness.append(min_fitness)
#         runs_max_objective.append(m_obj)
#
#
#     plot_final_runs(runs_and_avg_fitness, runs_and_max_fitness, runs_and_min_fitness, runs_and_objective,
#                     algo_type="CHC",
#                     evaluator="eil76")
#
#     # plot_final_runs(runs_and_avg_fitness, runs_and_max_fitness, runs_and_min_fitness, runs_and_objective,
#     #                 algo_type="CHC",
#     #                 evaluator="eil76", withObjective=True)
#
#     plot_final_runs_objective(runs_and_objective, runs_max_objective, algo_type="CHC", evaluator="eil76")
