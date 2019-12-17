import functions as f
import genetic_functions as genteik


def main():
    data, optimal = f.load_data()
    params = f.load_parameters()
    # chrom_n = 500
    # gen_n = 600
    print("Optimal:: {}\n".format(optimal))
    der_Übermensch = genteik.evolve(
        len(data), params["chrom_n"], params["gen_n"], data, optimal)
    print(der_Übermensch)

if __name__ == '__main__':
    main()
