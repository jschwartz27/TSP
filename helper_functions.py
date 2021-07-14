import csv
import yaml
import numpy as np
from typing import List


def load_yaml(filename: str):
    with open(f"{filename}.yaml", 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


class Coordinates:

    def __init__(self, coords: np.ndarray, optimal_distance: int):
        self.coords: np.ndarray = coords
        self.optimal_distance: int = optimal_distance


def load_coordinates(city_data_file="xqf131") -> Coordinates:
    coords = list()
    with open(f'cities_data/{city_data_file}.tsp', newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in file:
            if row[0].isdigit() is False:
                if row[0] == "OPTIMAL_DISTANCE":
                    optimal = int(row[2])
                else:
                    continue
            else:
                coord = [int(row[1]), int(row[2])]
                coords.append(coord)

    return Coordinates(np.array(coords), optimal)


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
