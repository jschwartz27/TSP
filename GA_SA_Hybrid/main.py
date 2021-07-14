import functions as f
import genetic_functions as genetik


def main():
    data, optimal = f.load_data()
    params = f.load_parameters()
    print("Optimal:: {}\n".format(optimal))
    der_uebermensch, fitness_dictionary, error = genetik.evolve(
        len(data), params, data, optimal)
    print(der_uebermensch)

    if params["graph_data"] is True:
        f.show_graph(fitness_dictionary, params, error)
    else:
        quit()


if __name__ == '__main__':
    main()
