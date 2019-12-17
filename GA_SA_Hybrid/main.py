import functions as f
import genetic_functions as genteik


def main():
    data, optimal = f.load_data()
    params = f.load_parameters()
    print("Optimal:: {}\n".format(optimal))
    der_Übermensch, fitsDict = genteik.evolve(
        len(data), params, data, optimal)
    print(der_Übermensch)

    if params["graph_data"] is True:
        f.show_graph(fitsDict, params)
    else:
        quit()

if __name__ == '__main__':
    main()
