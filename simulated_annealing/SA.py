import random
import numpy as np
from math import exp
from typing import Dict, List

# import numpy.typing as npt
# npt.NDArray[np.int_]


def random_prob(precision=3) -> float:
    return random.randint(0, 10 ** precision) / float(10 ** precision)


class SimulatedAnnealing:

    restart_locations: List[int] = list()  # for graphing purposes
    temperature: float = None
    percentage_error: float = None

    def __init__(self, city, parameters: Dict):
        self.coords: np.ndarray = city.coords
        self.iterations: int = parameters["iterations"]
        self.restart_threshold: int = int(self.iterations * parameters["restart_threshold"])
        self.annealing_schedule = parameters["annealing_schedule"]
        self.energy: float = self.__evaluate(city.coords)
        self.data: List[float] = [round(self.energy, 3)]
        self.global_lowest_energy: float = self.energy
        self.global_lowest_energy_coords = city.coords
        self.optimal_distance: int = city.optimal_distance

    def run_annealing_schedule(self) -> None:

        suboptimal_iterations: int = 0

        for iteration, temperature in enumerate(self.__exponential_decay_annealing_schedule()):  # range(self.iterations - 1):
            print(f"iteration:: {iteration} Energy:: {round(self.global_lowest_energy, 2)}\r", end="")
            self.temperature = temperature  # self.__annealing_schedule(iteration)
            new_config = self.__swap()
            new_config_energy = self.__evaluate(new_config)

            if self.__accept_state_change(new_config_energy):
                self.coords = new_config
                self.energy = new_config_energy
                if new_config_energy < self.global_lowest_energy:
                    self.global_lowest_energy_coords = new_config  # for restart function
                    self.global_lowest_energy = new_config_energy

            if self.global_lowest_energy < self.energy:
                suboptimal_iterations += 1
                if suboptimal_iterations > self.restart_threshold:
                    self.__restart()
                    self.restart_locations.append(iteration)
                    suboptimal_iterations = 0
            else:
                suboptimal_iterations = 0

            self.data.append(round(self.energy, 3))

            # Break for loop if optimum is reached
            if self.global_lowest_energy == self.optimal_distance:
                break

        self.percentage_error = round(((
            (self.global_lowest_energy - self.optimal_distance)
        ) / self.optimal_distance) * 100, 1)

        print(f"\nERROR          :: {self.percentage_error}%")
        print(f"LOWEST ENERGY  :: {round(self.global_lowest_energy, 2)}")
        print(f"OPTIMAL ENERGY :: {self.optimal_distance}")

    def __exponential_decay_annealing_schedule(self):
        """return temperature which is a probability (0, 1]"""
        j, k = self.annealing_schedule["coefficient"], self.annealing_schedule["exponent"]
        return map(
            lambda x: round(j * exp(k * x), 3), map(
                lambda y: round(y/self.iterations, 3), range(self.iterations)))

    '''
    def __annealing_schedule(self, i):
        return round(1 - ((i + 1) / self.iterations), 3)
    '''

    def __restart(self) -> None:
        self.coords = self.global_lowest_energy_coords
        self.energy = self.global_lowest_energy

    def __swap(self) -> np.ndarray:
        new_coords = np.copy(self.coords)
        j, k = random.sample(range(len(self.coords)), 2)
        new_coords[[j, k]] = new_coords[[k, j]]
        return new_coords

    def __accept_state_change(self, new_energy_level: float) -> bool:
        accept_prob = self.__acceptance_probability(new_energy_level)
        if accept_prob >= random_prob():  # random.random is unfortunately [0, 1)
            return True
        else:
            return False

    def __acceptance_probability(self, e_prime: float) -> float:
        if e_prime < self.energy:
            return 1
        elif self.temperature == 0:
            return 0
        else:
            return np.exp((0 - (e_prime - self.energy)) / self.temperature)

    @staticmethod
    def __evaluate(configuration: np.ndarray) -> float:
        distance = 0
        for i in range(len(configuration) - 1):
            distance += np.linalg.norm(configuration[i] - configuration[i+1])
        return distance
