import numpy as np
from typing import Dict, List


class Chromosome:

    def __init__(self, chromosome: np.ndarray, fitness: float):
        self.chromosome: np.ndarray = chromosome
        self.fitness: float = fitness


class GeneticSimulatedAnnealing:

    data = {
        "fitness": list(),
        "mean": list()
    }

    def __init__(self, city, parameters: Dict):
        self.coords: np.ndarray = city.coords
        self.optimal_distance: int = city.optimal_distance
        self.chromosome_n: int = parameters["chrom_n"]
        self.generation_n: int = parameters["gen_n"]
        self.crossover_prob: float = parameters["crossP"]
        self.genome = self.__create_chroms()

    def run_genetic_simulated_annealing(self):
        for generation in range(self.generation_n):
            pass

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
        c = chrom + [chrom[0]]
        for i in range(len(c) - 1):
            distance += np.linalg.norm(self.coords[c[i]] - self.coords[c[i]])

        return round(distance, 3)
