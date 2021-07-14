import random
import numpy as np
from typing import List

# import numpy.typing as npt
# npt.NDArray[np.int_]


def random_prob(precision=3) -> float:
    return random.randint(0, 10 ** precision) / float(10 ** precision)


class SimulatedAnnealing:

    restart_locations: List[int] = list()  # for graphing purposes
    temperature: float = None

    def __init__(self, coords: np.ndarray, iterations=1000):
        self.coords: np.ndarray = coords
        self.iterations: int = iterations
        self.energy: float = self.__evaluate(coords)
        self.data: List[float] = [round(self.energy, 3)]
        self.global_lowest_energy: float = self.energy
        self.global_lowest_energy_coords = coords

    def run_annealing_schedule(self) -> None:
        restart_threshold: int = int(self.iterations * 0.1)
        suboptimal_iterations: int = 0

        for iteration in range(self.iterations - 1):
            # print(f"iteration:: {iteration} Energy:: {self.energy}\r", end="")
            self.temperature = round(1 - ((iteration + 1) / self.iterations), 3)
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
                if suboptimal_iterations > restart_threshold:
                    self.__restart()
                    self.restart_locations.append(iteration)
                    suboptimal_iterations = 0
            else:
                suboptimal_iterations = 0

            self.data.append(round(self.energy, 3))

    def __restart(self) -> None:
        self.coords = self.global_lowest_energy_coords
        self.energy = self.global_lowest_energy

    def __swap(self):
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
        else:
            return np.exp((0 - (e_prime - self.energy)) / self.temperature)

    @staticmethod
    def __evaluate(configuration: np.ndarray) -> float:
        distance = 0
        for i in range(len(configuration) - 1):
            distance += np.linalg.norm(configuration[i] - configuration[i+1])
        return distance


class Coordinates:

    def __init__(self, n=100, h=100, w=100):
        self.n: int = n
        self.h: int = h
        self.w: int = w
        self.coords = self.__generate_coords()

    def __generate_coords(self) -> np.ndarray:
        coords_list = list()
        for _ in range(self.n):
            coords_list.append((random.randrange(self.w), random.randrange(self.w)))
        return np.array(coords_list)


def plot_data(data, restarts: List[int]) -> None:
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    t = np.arange(len(data))
    fig, ax = plt.subplots()
    ax.plot(t, data, color='limegreen')
    ax.set(
        xlabel='time (iterations)',
        ylabel='energy (E)',
        title='Energy over Annealing Schedule')
    if len(restarts) > 0:
        for restart_x_value in restarts:
            ax.axvline(x=restart_x_value, color="red")
    # ax.grid()
    # fig.savefig("test.png")
    plt.show()


def main() -> None:
    simulated_annealing = SimulatedAnnealing(Coordinates().coords)
    simulated_annealing.run_annealing_schedule()
    plot_data(simulated_annealing.data, simulated_annealing.restart_locations)


if __name__ == "__main__":
    main()
