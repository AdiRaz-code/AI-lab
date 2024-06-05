import random
from typing import Union, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

GA_MUTATIONRATE = 0.50  # Mutation rate
GA_TARGET = [
    [0, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0]]

pop_size = 2048
num_genes = 13
max_generations = 100
k = 2

# 0 - PMX, 1 - CX
rep_operator = 0

# 0 - show plot every gen, 1 - show plot only at the end
want_plot = 0

# parent selection: 1 = RWS + Linear scaling, 2 = SUS + Linear scaling, 3 = RWS + RANKING, 4 = non deterministic tournament
select_parents = 1

##############################################    Hello, world! - LAB A PART 1    #########################################
# def mutate(individual):
#     tsize = len(GA_TARGET)
#     ipos = random.randint(0, tsize - 1)
#     delta = random.randint(32, 121)  # rand() % 90 + 32 produces a number in the range [32, 121]
#
#     individual[ipos] = chr((ord(individual[ipos]) + delta) % 122)


# Define a function that get the reproduction operator - SINGLE, TWO or UNIFORM and returns the new child
# def single_two_uniform_cross(choice, parent1, parent2):
#     if choice == 0:
#         # SINGLE = Choose a random crossover point
#         crossover_point = random.randint(1, len(parent1) - 1)
#
#         # Perform crossover
#         child = parent1[:crossover_point] + parent2[crossover_point:]
#
#         return child
#
#     elif choice == 1:
#         # TWO = Choose two random points for crossover
#         point1 = random.randint(0, len(parent1) - 1)
#         point2 = random.randint(point1 + 1, len(parent1))
#
#         # Perform crossover
#         child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
#
#         return child
#
#     else:
#         # UNIFORM =  Perform uniform crossover based on the crossover rate
#         child = [parent1[i] if random.random() < 0.5 else parent2[i] for i in range(len(parent1))]
#
#         return child
#

# Define the fitness function with the addition of " bulls and cows"
# def fitness(individual):
#     target = list(GA_TARGET)
#     score = 0
#     for i, char in enumerate(individual):
#         if char in GA_TARGET:
#             if individual[i] == target[i]:
#                 score += 5
#             else:
#                 score += 1
#     return score


# Define the fitness function
# def fitness(individual):
#     target = list("Hello, world!")
#     score = 0
#     for i in range(len(individual)):
#         if individual[i] == target[i]:
#             score += 1
#     return score

# def plot_func(fitnesses, want_plot):
#     # Normalization of the fitness values
#     min_fitness = np.min(fitnesses)
#     max_fitness = np.max(fitnesses)
#     normalized_fit = []
#     for i, value in enumerate(fitnesses):
#         if max_fitness == min_fitness:
#             normalized_fit.append(100)
#         else:
#             normalized_value = 100 * (value - min_fitness) / (max_fitness - min_fitness)
#             normalized_fit.append(normalized_value)
#
#     counter = Counter(normalized_fit)
#     x_values = []
#     y_values = []
#     gap = 0.1
#
#     for value, count in counter.items():
#         x_values.extend([value] * count)
#         y_values.extend(np.arange(count) * gap)
#
#     if want_plot == 0:
#         plt.scatter(x_values, y_values)
#         plt.xticks(np.arange(0, 101, 10))
#         plt.yticks([])
#         plt.xlabel('fitness values')
#         plt.title('The normalized density of the fitness values ')
#         plt.grid(True)
#         plt.show()
###########################################################################################################################


##########################################   SUDUKU + BIN PACKING  - LAB A PART 2   #######################################


# def RWS(population, fitness_values, offspring_size):
#     total_fitness = sum(fitness_values)
#     selection_probs = [fitness / total_fitness for fitness in fitness_values]
#     cum_probs = [sum(selection_probs[:i + 1]) for i in range(len(selection_probs))]
#     selected = []
#     for _ in range(offspring_size):
#         r = random.random()
#         for i in range(len(cum_probs)):
#             if r <= cum_probs[i]:
#                 selected.append(population[i])
#                 break
#
#     return selected

def RWS(population, fitness_values, offspring_size):
    total_fitness = sum(fitness_values)

    if total_fitness == 0:
        # Handle the case where total_fitness is zero
        return random.choices(population, k=offspring_size)

    selection_probs = [fitness / total_fitness for fitness in fitness_values]
    cum_probs = [sum(selection_probs[:i + 1]) for i in range(len(selection_probs))]
    selected = []

    for _ in range(offspring_size):
        r = random.random()
        for i in range(len(cum_probs)):
            if r <= cum_probs[i]:
                selected.append(population[i])
                break

    return selected


def SUS(population, fitness_values, num_to_select):
    """
    Perform Stochastic Universal Sampling (SUS) to select individuals from the population.

    Parameters:
    - population: List of individuals
    - fitnesses: List of fitness values corresponding to each individual
    - num_to_select: Number of individuals to select

    Returns:
    - selected: List of selected individuals
    """
    total_fitness = sum(fitness_values)
    pointer_distance = total_fitness / num_to_select
    start_point = random.uniform(0, pointer_distance)
    pointers = [start_point + i * pointer_distance for i in range(num_to_select)]

    selected = []
    current_member = 0
    cumulative_fitness = fitness_values[current_member]

    for pointer in pointers:
        while cumulative_fitness < pointer:
            current_member += 1
            cumulative_fitness += fitness_values[current_member]
        selected.append(population[current_member])

    return selected


def linear_scaling(fitness_values):
    scaled_fitnesses = []
    k = 5
    best_fitness = max(fitness_values)
    avg_fitness = np.nanmean(fitness_values)  # Use np.nanmean to handle NaN values in the average

    if np.isnan(avg_fitness) or best_fitness == avg_fitness:
        # Handle cases where the average is NaN or best_fitness equals avg_fitness
        return [0] * len(fitness_values)

    scaling_factor = (k - 1) / (best_fitness - avg_fitness)
    offset = 1 - (scaling_factor * avg_fitness)

    for fitness in fitness_values:
        scaled_fitness = scaling_factor * fitness + offset
        if np.isnan(scaled_fitness):
            scaled_fitnesses.append(0)  # Assign a default value for NaN
        else:
            scaled_fitnesses.append(int(round(scaled_fitness)))

    return scaled_fitnesses


def RANKING(fitness_values, s=1.7):
    new_fitness = []
    """
    Rank-based linear scaling of fitnesses.
    :param fitnesses: List of fitness values.
    :param s: Selection pressure parameter (1.0 < s <= 2.0).
    :return: Scaled probabilities for selection.
    """
    if not 1.0 < s <= 2.0:
        raise ValueError("s must be between 1.0 and 2.0")

    mu = len(fitness_values)
    # Rank individuals based on fitness (higher fitness gets higher rank)
    sorted_indices = np.argsort(fitness_values)[::-1]
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, mu + 1)  # Ranks from 1 to mu

    # Calculate selection probabilities based on ranks
    probabilities = (2 - s) / mu + (2 * ranks * (s - 1)) / (mu * (mu - 1))

    # return probabilities # return ndarray

    # Convert to list before returning
    return probabilities.tolist()


def tournament(population, fitness_values, num_to_select):
    selected_parents = []
    for i in range(num_to_select):
        # Step 1: Select k random individuals
        selected_indices = random.sample(range(len(population)), k)
        selected_individuals = [population[i] for i in selected_indices]
        selected_fitnesses = [fitness_values[i] for i in selected_indices]

        # Step 2: Sort the selected individuals by their fitness values
        sorted_individuals = [x for _, x in sorted(zip(selected_fitnesses, selected_individuals), reverse=True)]
        sorted_fitnesses = sorted(selected_fitnesses, reverse=True)

        temp = k

        while temp > 1:
            p = random.uniform(0.5, 1)
            x = random.uniform(0, 1)
            if p >= x:
                selected_parents.append(sorted_individuals[0])
                break
            else:
                temp = temp - 1
                sorted_individuals = sorted_individuals[1:]

        selected_parents.append(sorted_individuals[0])  # only 1 left

    return selected_parents


def Aging(inidividual):
    age = 5
    return age


###################################################    SUDUKU    ##########################################################

def PMX(parents1, parents2):
    size = len(parents1[0][0])
    children = []

    # index i is the next child we create
    for i in range(len(parents1)):
        child = parents1[i]

        # index j is a row in child[i]
        for j in range(len(child)):
            respect = False
            mutate = random.random() < GA_MUTATIONRATE
            while not respect:
                # index is a random index for crossover
                index = random.randint(0, size - 1)

                # Swap the values at the chosen index
                value1, value2 = child[j][index], parents2[i][j][index]
                child[j][index] = value2
                # value2 is the value inserted to the child from parent2
                # value1 is the missing value need to be fixed

                # fix the permutation
                for m in range(len(child[j])):
                    if child[j][m] == value2:
                        if m != index:
                            child[j][m] = value1

                # here we want to see if the new row child[j] is with respect to the given grid
                if respects_given(child[j], GA_TARGET[j]):
                    if mutate:
                        inversion_mutation(child[j], GA_TARGET[j])
                    respect = True
                else:
                    child[j] = parents1[i][j]
        children.append(child)
    return children


def respects_given(child_row, given_row):
    for index in range(len(given_row)):
        if given_row[index] != 0 and child_row[index] != given_row[index]:
            return False
    return True


def cx(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size

    def create_child(parent1, parent2):
        child = [-1] * size
        start = 0

        while -1 in child:
            if child[start] == -1:
                current_pos = start
                while True:
                    child[current_pos] = parent1[current_pos]
                    current_pos = parent1.index(parent2[current_pos])
                    if current_pos == start:
                        break
                start = child.index(-1)

        return child

    child1 = create_child(parent1, parent2)
    child2 = create_child(parent2, parent1)

    return child1, child2


def inversion_mutation(list_to_mutate, list_to_respect):
    respect = False
    while not respect:
        idx1, idx2 = sorted(random.sample(range(len(list_to_mutate)), 2))

        reversed_subarray = list_to_mutate[idx1:idx2 + 1][::-1]
        # temp is the array without the reversed part
        temp = list_to_mutate[:idx1] + list_to_mutate[idx2 + 1:]
        insert_idx = random.randint(0, len(temp))

        new_array = temp[:insert_idx] + reversed_subarray + temp[insert_idx:]

        if respects_given(new_array, list_to_respect):
            return new_array


def scramble_mutation(array):
    # Select two random indices for the scramble
    idx1, idx2 = sorted(random.sample(range(len(array)), 2))

    # Shuffle the elements between the two indices
    subset = array[idx1:idx2 + 1]
    random.shuffle(subset)
    array[idx1:idx2 + 1] = subset

    return array


def is_valid(sudoku, row, col, num):
    # Check if num is not in the current row, column, and 3x3 sub-grid
    sub_row, sub_col = 3 * (row // 3), 3 * (col // 3)
    return (num not in sudoku[row] and
            num not in (sudoku[r][col] for r in range(9)) and
            num not in (sudoku[sub_row + r][sub_col + c] for r in range(3) for c in range(3)))


def count_conflicts(sudoku):
    conflicts = 0
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != 0:
                if not is_valid(sudoku, i, j, sudoku[i][j]):
                    conflicts += 1
    return conflicts


def count_duplicates_subgrid(subgrid):
    count = 0
    temp = []
    for num in subgrid:
        if num not in temp:
            temp.append(num)
        else:
            count += 1
    return count


def fitness_evaluation(grid):
    fitness = 0

    # Check columns
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        fitness -= len(column) - len(set(column))

    # Check 3x3 subgrids
    for row in range(0, 9, 3):
        for col in range(0, 9, 3):
            subgrid = []
            for i in range(3):
                for j in range(3):
                    subgrid.append(grid[row + i][col + j])
            fitness -= count_duplicates_subgrid(subgrid)

    return fitness


def plot_func(fitnesses):
    # Normalization of the fitness values
    min_fitness = np.min(fitnesses)
    max_fitness = np.max(fitnesses)
    normalized_fit = []
    for i, value in enumerate(fitnesses):
        if max_fitness == min_fitness:
            normalized_fit.append(100)
        else:
            normalized_value = 100 * (value - min_fitness) / (max_fitness - min_fitness)
            normalized_fit.append(normalized_value)

    counter = Counter(normalized_fit)
    x_values = []
    y_values = []
    gap = 0.1

    for value, count in counter.items():
        x_values.extend([value] * count)
        y_values.extend(np.arange(count) * gap)

    if want_plot == 0:
        plt.scatter(x_values, y_values)
        plt.xticks(np.arange(0, 101, 10))
        plt.yticks([])
        plt.xlabel('fitness values')
        plt.title('The normalized density of the fitness values ')
        plt.grid(True)
        plt.show()


def init_pop(pop_size):
    population = []
    for _ in range(pop_size):
        individual = []
        for row in GA_TARGET:
            fixed = [num for num in row if num != 0]
            available_numbers = [num for num in range(1, 10) if num not in fixed]
            random.shuffle(available_numbers)
            individual_row = [
                num if num != 0 else available_numbers.pop() for num in row
            ]
            individual.append(individual_row)
        population.append(individual)
    return population


def grid_to_string(grid):
    return '\n'.join([' '.join(map(str, row)) for row in grid])

###################################################   BIN PACKING     #####################################################


###################################################    def genetic_algorithm   ############################################


# Define the genetic algorithm
def genetic_algorithm(pop_size, num_genes, fitness_func, max_generations):
    # Initialize the population with random individuals
    population = init_pop(pop_size)

    # for i in range(pop_size):
    #     individual = [chr(random.randint(32, 126)) for j in range(num_genes)]
    #     population.append(individual)

    target_gen = -1  # indicator for converging
    global_cpu_time = 0
    global_time = 0

    # asking from the user to choose the reproduction operator
    # rep_operator = int(input("Please enter your decision for the reproduction operator : SINGLE enter 0 , TWO enter 1 , UNIFORM enter 2: "))

    # Evolve the population for a fixed number of generations
    for generation in range(max_generations):

        start_time = time.time()
        start_CPU_time = time.perf_counter_ns()

        # Evaluate the fitness of each individual
        fitnesses = [fitness_func(individual) for individual in population]

        # Select the best individuals for reproduction
        elite_size = int(pop_size * 0.1)
        elite_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)[:elite_size]
        elites = [population[i] for i in elite_indices]

        offspring_size = pop_size - elite_size

        selection_methods = {
            1: lambda: RWS(population, linear_scaling(fitnesses), offspring_size),
            2: lambda: SUS(population, linear_scaling(fitnesses), offspring_size),
            3: lambda: RWS(population, RANKING(fitnesses), offspring_size),
            4: lambda: tournament(population, fitnesses, offspring_size)
        }

        parents1 = selection_methods[select_parents]()
        parents2 = selection_methods[select_parents]()

        # parants1 and 2 are lists of all parents in size offspring from them we will create new children

        offspring = []

        selection_cross = {
            0: lambda: PMX(parents1, parents2),
            1: lambda: cx(parents1, parents2)
        }

        #children: Union[list[Any], tuple[list[int], list[int]]] = selection_cross[rep_operator]()
        children = selection_cross[rep_operator]()

        # for i in range(len(children)):
        #     if random.random() < GA_MUTATIONRATE:
        #         inversion_mutation(children[i])
        #         # scramble_mutation(children[i])
        #         offspring.append(children[i])
        #     else:
        #         offspring.append(children[i])

        population = elites + children

        # population = elites + offspring[:offspring_size]   chatGPT said so, dont konw why in conversion : "Crossover Methods in GA

        gen_CPU_time = time.perf_counter_ns() - start_CPU_time
        gen_time = time.time() - start_time

        global_cpu_time += gen_CPU_time
        global_time += gen_time

        avg_fit = np.mean(fitnesses)
        std_fit = np.std(fitnesses)
        best_individual = max(population, key=lambda individual: fitness_func(individual))
        # Find the individual with the highest fitness
        best_fitness = fitness_func(best_individual)
        best_individual_str = grid_to_string(best_individual)

        if best_individual == GA_TARGET:
            want_plot = 0
            target_gen = generation + 1
            print(
                f"Generation {generation + 1}:Best individual = {''.join(best_individual_str)}, Avg Fitness = {avg_fit}, Std Dev = {std_fit}, Gen CPU time = {gen_CPU_time / 1000000:.4f} ms , Gen real time = {gen_time * 1000:.4f} ms")
            print(
                f"GA_TARGET achieved at generation: {target_gen}. Global CPU time elapsed : {global_cpu_time / 1000000:.4f} ms, Global real time = = {global_time * 1000:.4f} ms ")

        # if std_fit == 0 and target_gen == -1:
        #     want_plot = 0
        #     target_gen = generation + 1
        #     print(
        #         f"Generation {generation + 1}:Best individual = {''.join(best_individual_str)}, Avg Fitness = {avg_fit}, Std Dev = {std_fit}, Gen CPU time = {gen_CPU_time / 1000000:.4f} ms , Gen real time = {gen_time * 1000:.4f} ms")
        #     print(
        #         f"Global CPU time elapsed : {global_cpu_time / 1000000:.4f} ms, Global real time =  {global_time * 1000:.4f} ms ")

        if target_gen == -1:
            print(
                f"Generation {generation + 1}:Best individual = {''.join(best_individual_str)}, Avg Fitness = {avg_fit}, Std Dev = {std_fit}, Gen CPU time = {gen_CPU_time / 1000000:.4f} ms , Gen real time = {gen_time * 1000:.4f} ms")

        if target_gen == -1 and generation == max_generations - 1:
            print("the individuals did not converge")
            print(
                f"Global CPU time elapsed : {global_cpu_time / 1000000:.4f} ms, Global real ime =  {global_time * 1000:.4f} ms")
            want_plot = 0

        plot_func(fitnesses)

        if target_gen != -1:
            break

    return best_individual, best_fitness


###################################################    MAIN    ############################################################


# Run the genetic algorithm and print the result
best_individual, best_fitness = genetic_algorithm(pop_size=pop_size, num_genes=num_genes, fitness_func=fitness_evaluation,
                                                  max_generations=max_generations)
best_individual_str = grid_to_string(best_individual)
print("Best individual:", ''.join(best_individual_str))
print("Best fitness:", best_fitness)
