import random
import statistics
from copy import deepcopy
from math import exp

import numpy as np
from typing import Dict, List, Tuple

# TODO class for genome so it can have best fitness and mean and shit at all times
# also put create new or fill to max pop or whatever


class Chromosome:

    def __init__(self, chromosome: np.ndarray, fitness: float):
        self.chromosome: np.ndarray = chromosome
        self.fitness: float = fitness


class Genome:

    best_fitness: float = None
    mean: float = None
    standard_deviation: float = None

    def __init__(self, chromosome_n):
        self.chromosome_n = chromosome_n
        self.genome: List[Chromosome] = self.__create_chromosomes(chromosome_n)

    def __create_chromosomes(self, chromosome_n):
        pass


class GeneticSimulatedAnnealing:

    data = {
        "fitness": list(),
        "mean": list()
    }
    temperature: float = None

    def __init__(self, city, parameters: Dict):
        self.coords: np.ndarray = city.coords  # this list should never change
        self.optimal_distance: int = city.optimal_distance
        self.chromosome_n: int = parameters["chrom_n"]
        self.generation_n: int = parameters["gen_n"]
        self.crossover_p: float = parameters["cross_p"]
        self.elite_p: float = parameters["elite_p"]
        self.truncation_p: float = parameters["truncation_p"]

        self.selection_parameters = parameters["selection_parameters"] # TODO change yaml and update init
        self.crossover_parameters = parameters["crossover_parameters"]

        self.genome = self.__create_chroms()

    def run_genetic_simulated_annealing(self):
        self.__append_data()
        for generation in range(self.generation_n):
            self.temperature = self.__annealing_schedule(generation)

            # SELECTION
            elite, selected_pop = Selection(self.genome, self.chromosome_n, self.temperature,
                                            self.selection_parameters).select()

            # CROSSOVER
            selected_pop_crossed = Crossover(selected_pop, self.chromosome_n, self.temperature, len(self.coords),
                                             self.crossover_parameters).crossover()

            # MUTATION
            mutated_selected_pop_crossed = Mutation(selected_pop_crossed).mutate()

            self.genome = elite + mutated_selected_pop_crossed

            self.__append_data()
            quit()

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

    def __eval_distance(self, chrom: np.ndarray) -> float:
        distance = 0
        # This is to add the return to the start point
        c = np.append(chrom, chrom[0])

        for i in range(len(c) - 1):
            distance += np.linalg.norm(self.coords[c[i]] - self.coords[c[i+1]])

        return round(distance, 3)

    def __annealing_schedule(self, generation: int) -> float:
        gens_a = self.generation_n * .25
        eighth = (len(self.coords)/8)/100
        p = round(eighth * np.exp(((-generation/gens_a) / (len(self.coords)/30))), 2)  # 30 instead of 80

        return max(int(p * 100), 1) + 1

    def __append_data(self) -> None:
        """appends best_fitness and mean"""

        self.data["fitness"].append(self.genome[0].fitness)
        self.data["mean"].append(
            statistics.mean(map(
                lambda chromosome: chromosome.fitness, self.genome
            ))
        )


class Selection:

    length_to_append: int = 0
    elite_n: int = 0
    elite: List[Chromosome] = list()

    def __init__(self, genome: List[Chromosome], chromosome_n: int, temperature: float, selection_params):
        self.genome: List[Chromosome] = genome
        self.chromosome_n: int = chromosome_n
        self.temperature: float = temperature
        self.elite_p: float = selection_params["elite_p"]
        self.truncation_p: float = selection_params["truncation_p"]
        self.truncation_amount: float = selection_params["truncation_amount"]

    def select(self) -> [List[Chromosome], List[Chromosome]]:
        """return selected chromosomes NOT sorted by fitness"""

        # ELITE SELECTION
        if random.random() < self.elite_p:
            # remember elite_n % 2 MUST == 0  TODO why is this? I forget... crossover?
            self.elite_n = random.choice((2, 4, 6))
            self.elite = self.genome[:self.elite_n]
            # new pop will be recombined with elite after crossover
            self.length_to_append = self.chromosome_n - self.elite_n

        # TRUNCATION
        if random.random() < self.truncation_p:
            # truncates by 50%
            self.__truncation()

        # CHOOSE SELECTION FUNCTION
        selection_functions = (
            self._linear_rank, self._exponential_rank, self._tournament,
            self._roulette, self._boltzmann
        )
        w = (.1, .3, .2, .4, 0) if self.temperature >= 3 else (0, .5, .1, .4, 0)
        selection_function = random.choices(population=selection_functions, weights=w)[0]

        return self.elite, selection_function()

    def __truncation(self) -> None:
        assert self.chromosome_n % 2 == 0, "CANNOT TRUNCATE UNEAVEN DNA"

        # TODO can truncation amount be any number?
        dna = self.genome[:int(self.chromosome_n*self.truncation_amount)]
        # this doubles and maintains fitness ordering of chromosomes
        self.genome = [*dna, *dna]

    # ### SELECTION FUNCTIONS ###

    def _linear_rank(self):
        ns = list(range(1, self.chromosome_n+1))[::-1]
        probabilities = list(map(lambda x: x/sum(ns), ns))  # normalizes

        return random.choices(self.genome, probabilities, k=self.length_to_append)

    def _exponential_rank(self):
        ns = list(range(1, self.chromosome_n+1))[::-1]
        e = list(map(lambda x: .5**(5-x), ns))
        probabilities = list(map(lambda x: x/sum(e), e))

        return random.choices(self.genome, probabilities, k=self.length_to_append)

    def _tournament(self):
        new_chromosomes = list()
        for _ in range(self.length_to_append):
            # this may or may not be a good idea
            size = random.choice(range(2, 4))
            cs = random.sample(self.genome, k=size)
            # TODO how to get max from list of chroms?
            new_chromosomes.append(max(cs))

        return new_chromosomes

    def _roulette(self):
        # strip to get fitnesses
        fitnesses = map(lambda chromosome: chromosome.fitness, self.genome)
        if len(set(fs)) == 1:
            fs = [1 for _ in range(len(fs))]
        else:
            diff = fs[-1]
            # last chrom has 0 prob of selection, but...
            fs = list(map(lambda x: x - diff, fs))
            # here, I bump up last fitness by a notch so not zero
            fs[-1] = fs[-2]/4
            fs = list(map(lambda x: x**2, fs))

        total = sum(fs)
        probabilities = list(map(lambda x: x/total, fs))

        return random.choices(self.genome, probabilities, k=self.length_to_append)

    def _boltzmann(self):
        new_chromosomes = list()
        for _ in range(self.length_to_append):
            cs = random.sample(self.genome, k=2)
            if cs[0] == max(cs):  # TODO
                new_chromosomes.append(cs[0])
            else:
                if random.random() < exp((cs[0][0]-cs[1][0])/self.temperature):
                    new_chromosomes.append(cs[0])
                else:
                    new_chromosomes.append(cs[1])

        return new_chromosomes


class Crossover:

    def __init__(self, genome: List[Chromosome], chromosome_n: int, temperature: float,
                 chromosome_l: int, crossover_params: Dict):
        self.genome = genome
        self.chromosome_n = chromosome_n
        self.temperature = temperature
        self.chromosome_l = chromosome_l
        self.cross_p: float = crossover_params["cross_p"]

    def crossover(self) -> List[Chromosome]:
        random.shuffle(self.genome)
        half = int(len(self.genome)/2)

        cross_dna = flatten(list(map(
            lambda x, y: self.pmx_crossover(x, y, self.chromosome_l, self.cross_p, data),
            self.genome[:half], self.genome[half:]
        )))

        return self.genome

    def pmx_crossover(self, lover_1, lover_2, local_array, cross_prob, data):
        # https://www.researchgate.net/figure/Partially-mapped-crossover-operator-PMX_fig1_226665831
        if random.random() < cross_prob:  # Crossover probability
            idxs = sorted(random.sample(range(local_array), 2))
            c1 = self._pmx_function(lover_1, lover_2, idxs)
            c2 = self._pmx_function(lover_2, lover_1, idxs)

            if random.random() < .3:
                return c1, c2
            c1 = eval_distance(c1, data)
            c2 = eval_distance(c2, data)
            l1 = eval_distance(lover_1, data)
            l2 = eval_distance(lover_2, data)
            alles = [c1, c2, l1, l2]
            alles.sort()

            return alles[0][1], alles[1][1]
        else:
            return lover_1, lover_2

    @staticmethod
    def _pmx_function(c1, c2, idxs):
        copy_1 = deepcopy(c1)
        splice2 = c2[idxs[0]:idxs[1]]
        for i in splice2:
            copy_1.remove(i)
        return copy_1[:idxs[0]] + splice2 + copy_1[idxs[0]:]


class Mutation:

    def __init__(self, genome: List[Chromosome]):
        self.genome = genome

    def mutate(self) -> List[Chromosome]:
        return self.genome
