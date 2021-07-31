import helper_functions
from simulated_annealing.SA import SimulatedAnnealing
from GA_SA_Hybrid.GA_SA import GeneticSimulatedAnnealing


def main():
    city_coordinates = helper_functions.load_coordinates()
    sa = False
    ga_sa = True
    if sa is True:
        parameters = helper_functions.load_yaml("simulated_annealing/parameters")
        simulated_annealing = SimulatedAnnealing(city_coordinates, parameters)
        simulated_annealing.run_annealing_schedule()
        print(f"\nERROR          :: {simulated_annealing.percentage_error}%")
        print(f"LOWEST ENERGY  :: {round(simulated_annealing.global_lowest_energy, 2)}")
        print(f"OPTIMAL ENERGY :: {simulated_annealing.optimal_distance}")
        helper_functions.plot_data(simulated_annealing.data, simulated_annealing.restart_locations)
    elif ga_sa is True:
        parameters = helper_functions.load_yaml("GA_SA_Hybrid/parameters")
        genetic_simulated_annealing = GeneticSimulatedAnnealing(city_coordinates, parameters)
        genetic_simulated_annealing.run_genetic_simulated_annealing()

        # print(genetic_simulated_annealing.genome[:2])


if __name__ == "__main__":
    main()
