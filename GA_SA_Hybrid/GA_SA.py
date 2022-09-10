import random
import statistics
import numpy as np
from math import exp
from datetime import datetime
from typing import Dict, List, Tuple


def flatten(array):
    return [item for sublist in array for item in sublist]


class Chromosome:

    def __init__(self, chromosome: np.ndarray, fitness: float):
        self.chromosome = chromosome
        self.fitness = fitness


class Genome:

    def __init__(self, genome: List[Chromosome], fitness_mean_data: Dict,
                 best: Chromosome, error: float):
        self.genome = genome
        self.best_fitness = best
        self.fitness_mean_data = fitness_mean_data
        self.error = error


class GeneticSimulatedAnnealing:

    best: Chromosome = None
    data = {
        "fitness": list(),
        "mean": list()
    }

    def __init__(self, city, parameters: Dict):

        self.coords: np.ndarray = city.coords  # this list should never change
        self.optimal_distance: int = city.optimal_distance
        self.chromosome_n: int = parameters["general"]["chromosome_n"]
        self.generation_n: int = parameters["general"]["generation_n"]
        self.annealing_schedule = parameters["general"]["annealing_schedule"]

        self.selection_parameters = parameters["selection"]
        self.crossover_parameters = parameters["crossover"]
        self.mutation_parameters = parameters["mutation"]

        self.genome: List[Chromosome] = self.__create_chroms()

    def run_genetic_simulated_annealing(self) -> Genome:

        assert self.chromosome_n % 2 == 0, "Chromosome_n must be even"
        begin = datetime.now()

        self.__append_data()

        for temperature in self.__exponential_decay_annealing_schedule():
            print(f"temperature:: {temperature} Energy:: {round(self.genome[0].fitness, 2)}\r", end="")

            # ### SELECTION ###
            elite, selected_pop = Selection(self.genome, self.chromosome_n, temperature,
                                            self.selection_parameters).select()

            # ### CROSSOVER ###
            selected_pop_crossed = Crossover(self.coords, selected_pop, self.chromosome_n, temperature,
                                             self.crossover_parameters).crossover()

            # ### MUTATION ###
            mutated_selected_pop_crossed = Mutation(selected_pop_crossed, self.coords, temperature,
                                                    self.mutation_parameters).mutate()

            self.genome: List[Chromosome] = sorted(
                [*elite, *mutated_selected_pop_crossed],
                key=lambda x: x.fitness
            )

            self.__append_data()

            if self.genome[0].fitness == self.optimal_distance:
                break

        error = round(
            (100 * (self.genome[0].fitness - self.optimal_distance) / self.optimal_distance), 2)

        delta = datetime.now() - begin
        print(f"\nERROR          :: {error}%")
        print(f"LOWEST ENERGY  :: {round(self.genome[0].fitness, 2)}")
        print(f"OPTIMAL ENERGY :: {self.optimal_distance}")
        print(f"TOTAL TIME :: {delta.total_seconds()}")

        return Genome(self.genome, self.data, self.best, error)

    def __create_chroms(self) -> List[Chromosome]:
        """return list of Chromosomes sorted by fitness"""

        chromosomes_raw: List[np.ndarray] = [
            np.random.choice(
                len(self.coords),
                len(self.coords),
                replace=False
            ) for _ in range(self.chromosome_n)
        ]

        return sorted(map(
            lambda x: Chromosome(x, self.__eval_distance(x)), chromosomes_raw),
            key=lambda y: y.fitness)

    def __eval_distance(self, chromosome: np.ndarray) -> float:
        distance = 0
        # This is to add the return to the start point
        c = np.append(chromosome, chromosome[0])

        for i in range(len(c) - 1):
            distance += np.linalg.norm(self.coords[c[i]] - self.coords[c[i + 1]])

        return round(distance, 3)

    def __append_data(self) -> None:
        """appends best_fitness and mean"""
        self.data["fitness"].append(self.genome[0].fitness)
        self.data["mean"].append(
            statistics.mean(map(
                lambda chromosome: chromosome.fitness, self.genome
            ))
        )

    def __exponential_decay_annealing_schedule(self):
        """return temperature which is a probability (0, 1]"""
        j, k = self.annealing_schedule["coefficient"], self.annealing_schedule["exponent"]
        return map(
            lambda x: round(j * exp(k * x), 3), map(
                lambda y: round(y/self.generation_n, 3), range(self.generation_n)))


def eval_distance(chromosome: np.ndarray, coords: np.array) -> float:
    distance = 0
    # This is to add the return to the start point
    c = np.append(chromosome, chromosome[0])

    for i in range(len(c) - 1):
        distance += np.linalg.norm(coords[c[i]] - coords[c[i + 1]])

    return round(distance, 3)


class Selection:

    def __init__(self, genome: List[Chromosome], chromosome_n: int,
                 temperature: float, selection_params):
        self.elite_n: int = 0
        self.elite: List[Chromosome] = list()

        self.genome = genome
        self.chromosome_n = chromosome_n
        self.length_to_append = chromosome_n
        self.temperature = temperature
        self.elite_p: float = selection_params["elite_p"]
        self.truncation_p: float = selection_params["truncation_p"]
        self.truncation_amount: float = selection_params["truncation_amount"]

    def select(self) -> [List[Chromosome], List[Chromosome]]:
        """return elite and selected chromosomes NOT sorted by fitness"""
        assert self.truncation_amount == 0.5, "Truncation amount must be 50%"

        # ELITE SELECTION
        if random.random() < self.elite_p:
            # remember elite_n % 2 MUST == 0  TODO why is this? I forget... crossover?
            self.elite_n = random.choice((2, 4, 6))
            self.elite = self.genome[:self.elite_n]
            # new pop will be recombined with elite after crossover
            self.length_to_append = self.chromosome_n - self.elite_n

        # TRUNCATION
        if random.random() < self.truncation_p:
            self.__truncation()

        # CHOOSE SELECTION FUNCTION
        selection_functions = (
            self._linear_rank, self._exponential_rank, self._tournament,
            self._roulette, self._boltzmann
        )
        w = (.1, .3, .2, .4, 0) if self.temperature >= 0.3 else (0, .5, .1, .4, 0)
        selection_function = random.choices(population=selection_functions, weights=w)[0]

        return self.elite, selection_function()

    def __truncation(self) -> None:
        assert self.chromosome_n % 2 == 0, "CANNOT TRUNCATE UNEVEN DNA"

        dna = self.genome[:int(self.chromosome_n * self.truncation_amount)]
        # this doubles and maintains fitness ordering of chromosomes
        self.genome = [*dna, *dna]  # TODO maybe just better with the zip function
        self.__sort_genome_fitness()

    # ### SELECTION FUNCTIONS ###

    def _linear_rank(self) -> List[Chromosome]:
        ns = list(range(1, self.chromosome_n + 1))[::-1]
        probabilities = list(map(lambda x: x / sum(ns), ns))  # normalizes

        return random.choices(self.genome, probabilities, k=self.length_to_append)

    def _exponential_rank(self) -> List[Chromosome]:
        ns = list(range(1, self.chromosome_n + 1))[::-1]
        # e = list(map(lambda x: .5 ** (5 - x), ns))
        # probabilities = list(map(lambda x: x / sum(ns), ns))
        e = list(map(lambda x: x**2, ns))
        probabilities = list(map(lambda x: x / sum(e), e))

        return random.choices(self.genome, probabilities, k=self.length_to_append)

    def _tournament(self) -> List[Chromosome]:
        new_chromosomes: List[Chromosome] = list()
        for _ in range(self.length_to_append):
            # TODO this may or may not be a good idea
            size = random.choice(range(2, 4))
            chromosomes_tournament = random.sample(self.genome, k=size)
            # TODO perhaps this could also be subject to the Annealing Schedule
            new_chromosomes.append(max(chromosomes_tournament, key=lambda x: x.fitness))

        return new_chromosomes

    def _roulette(self) -> List[Chromosome]:
        fitnesses: List[float] = list(map(lambda chromosome: chromosome.fitness, self.genome))
        total: float = round(sum(fitnesses), 3)
        probabilities = list(map(lambda x: x / total, fitnesses))

        return random.choices(self.genome, probabilities, k=self.length_to_append)

    def _boltzmann(self) -> List[Chromosome]:
        new_chromosomes: List[Chromosome] = list()
        for _ in range(self.length_to_append):
            cs = random.sample(self.genome, k=2)
            if cs[0].fitness == max(cs, key=lambda x: x.fitness):
                new_chromosomes.append(cs[0])
            else:
                if random.random() < exp((cs[0].fitness - cs[1].fitness) / self.temperature):  # TODO temp always less than 1? and dividing by 0
                    new_chromosomes.append(cs[0])
                else:
                    new_chromosomes.append(cs[1])

        return new_chromosomes

    # ### HELPER FUNCTIONS ###

    def __sort_genome_fitness(self) -> None:
        self.genome = sorted(self.genome, key=lambda y: y.fitness)


class Crossover:

    def __init__(self, coords: np.array, genome: List[Chromosome], chromosome_n: int,
                 temperature: float, crossover_params: Dict):
        self.coords = coords
        self.genome = genome
        self.chromosome_n = chromosome_n
        self.temperature = temperature
        self.chromosome_l: int = len(coords)
        self.crossover_p: float = crossover_params["crossover_p"]

    def crossover(self) -> List[Chromosome]:
        random.shuffle(self.genome)
        half = len(self.genome) // 2

        return sorted(flatten(list(map(
            lambda x, y: self.pmx_crossover(x, y),
            self.genome[:half], self.genome[half:]  # TODO how should chromosomes be selected for crossover?
        ))), key=lambda x: x.fitness)

    def pmx_crossover(self, lover_1: Chromosome, lover_2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        # https://www.researchgate.net/figure/Partially-mapped-crossover-operator-PMX_fig1_226665831
        if random.random() < self.crossover_p:
            indices: List[int] = sorted(random.sample(range(self.chromosome_l), 2))
            c1 = self._pmx_function(lover_1.chromosome, lover_2.chromosome, indices)
            c2 = self._pmx_function(lover_2.chromosome, lover_1.chromosome, indices)
            c1 = Chromosome(c1, eval_distance(c1, self.coords))
            c2 = Chromosome(c2, eval_distance(c2, self.coords))

            if random.random() < self.temperature:
                return c1, c2

            alles = sorted([c1, c2, lover_1, lover_2], key=lambda x: x.fitness)

            return alles[0], alles[1]
        else:
            return lover_1, lover_2

    @staticmethod
    def _pmx_function(c1: np.array, c2: np.array, indices: List[int]) -> np.array:
        splice2 = c2[indices[0]:indices[1]]
        c1 = np.setdiff1d(c1, splice2)
        return np.concatenate([c1[:indices[0]], splice2, c1[indices[0]:]])


class Mutation:

    def __init__(self, genome: List[Chromosome], coords: np.array, temperature: float, parameters):
        self.genome = genome
        self.coords = coords
        self.chromosome_l: int = len(coords)
        self.temperature = temperature
        self.mutation_rate: float = parameters["rate"]

    def mutate(self) -> List[Chromosome]:
        return list(map(
            lambda chromosome: self.__mutation_control(
                chromosome),
            self.genome)
        )

    def __mutation_control(self, chromosome_whole: Chromosome) -> Chromosome:
        """70% chance of switching two indices; 15% shuffle; and 15% reverse"""

        chromosome = chromosome_whole.chromosome
        fitness = chromosome_whole.fitness

        if random.random() < self.mutation_rate:
            new_chromosome = np.copy(chromosome)
            random_p = random.random()
            indices: List[int] = sorted(random.sample(range(self.chromosome_l), 2))
            if random_p < 0.7:
                # switch two indices
                new_chromosome[[indices[0], indices[1]]] = new_chromosome[[indices[1], indices[0]]]
            else:
                splice2 = new_chromosome[indices[0]:indices[1]]
                # scramble
                if random_p < 0.85:
                    np.random.shuffle(splice2)
                # reverse
                else:
                    splice2 = splice2[::-1]
                new_chromosome[indices[0]:indices[1]] = splice2

            new_fitness = eval_distance(new_chromosome, self.coords)
            if new_fitness < fitness or random.random() < self.temperature:
                return Chromosome(new_chromosome, new_fitness)
            else:
                return chromosome_whole
        else:
            return chromosome_whole
