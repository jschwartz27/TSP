import helper_functions
from simulated_annealing.SA import SimulatedAnnealing


def main():
    city_coordinates = helper_functions.load_coordinates()
    sa = True

    if sa is True:
        parameters = helper_functions.load_yaml("simulated_annealing/parameters")
        simulated_annealing = SimulatedAnnealing(city_coordinates, parameters)
        simulated_annealing.run_annealing_schedule()
        print(f"\nERROR        :: {simulated_annealing.percentage_error}%")
        print(f"LOWEST ENERGY  :: {simulated_annealing.global_lowest_energy}")
        print(f"OPTIMAL ENERGY :: {simulated_annealing.optimal_distance}")
        helper_functions.plot_data(simulated_annealing.data, simulated_annealing.restart_locations)


if __name__ == "__main__":
    main()
