import random
import neural

def generate_individual():
    # Primero, el numero de capas ocultas se genera aleatoreamente
    # Se considero un maximo de 10
    hidden_layers = random.randint(0, 10)
    hidden_layer_sizes = ()
    if hidden_layers > 0:
        i = 0
        while i < hidden_layers:
            # Si es mayor a 0, para cada capa se genera su cantidad de neuronas
            # Se considero un maximo de 100
            neurons = random.randint(1, 100)
            # Se agrega el valor a la tupla que recibe como parametro la red neuronal
            hidden_layer_sizes += (neurons,)
            i += 1
    activation = random.randint(0, 3)
    solver = random.randint(0, 2)
    alpha = random.uniform(0, 1)
    learning_rate = random.uniform(0, 1)
    momentum = random.uniform(0, 1)
    # Se retorna el arreglo de parametros
    return [
        hidden_layer_sizes,
        activation,
        solver,
        alpha,
        learning_rate,
        momentum
    ]

def generate_population(n):
    population = []
    i = 0
    while i < n:
        population.append(
            generate_individual()
        )
        i += 1
    return population

def fitness(individual):
    correlation = neural.run(
        individual[0],
        individual[1],
        individual[2],
        individual[3],
        individual[4],
        individual[5]
    )
    return correlation

def selection(population, fitness_results, fitness_mean):
    selected = []
    for i in range(0, len(population)):
        if fitness_results[i] > fitness_mean:
            selected.append(population[i])
    return selected

def crossover(individual1, individual2):
    # Intercambiar solo valores de alpha, learning_rate o momentum (entre 0 y 1)
    rand1 = random.randint(3, 5)
    rand2 = random.randint(3, 5)
    while rand1 == rand2:
        rand2 = random.randint(3, 5)
    randoms = [rand1, rand2]
    start = min(randoms)
    end = max(randoms)
    p1 = individual1[start:end]
    p2 = []
    for x in individual2:
        if x not in p1:
            p2.append(x)
    result = []
    for i in range(0, start):
        result.append(p2[i])
    result.extend(p1)
    for i in range(start, len(p2)):
        result.append(p2[i])
    return result

def reproduce(n, selected):
    new_population = []
    i = 0
    while i < n:
        result = crossover(
            selected[random.randint(0, len(selected)-1)],
            selected[random.randint(0, len(selected)-1)]
        )
        new_population.append(result)
        i += 1
    return new_population

def mutation(individual, mutation_prob):
    prob = random.randrange(100)
    if prob <= mutation_prob:
        # Intercambiar el valor de alpha con el de learning_rate
        alpha = individual[3]
        individual[3] = individual[4]
        individual[4] = alpha
    return individual

def run(n, max_iter, mutation_prob):
    population = generate_population(n)
    for i in range(0, max_iter):
        print("({})=======================================".format(i))
        print('Population size:', len(population))
        fitness_results = []
        for individual in population:
            fitness_results.append(fitness(individual))
        fitness_total = 0
        for j in range(0, len(population)):
            fitness_total += fitness_results[j]
        fitness_mean = fitness_total / len(fitness_results)
        print("Fitness mean: {}".format(fitness_mean))

        selected = selection(population, fitness_results, fitness_mean)
        print('Selected size:', len(selected))

        if not selected:
            break

        new_population = reproduce(n, selected)

        mutated_population = []
        for individual in new_population:
            mutated = mutation(individual, mutation_prob)
            mutated_population.append(mutated)
        population = mutated_population

    print("=======================================")
    print('FINAL POPULATION:', population)
    print('FINAL FITNESS MEAN:', fitness_mean)
