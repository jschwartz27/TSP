import csv
import genetic_functions as genteik


def load_test():
    with open('test.csv', newline='') as csvfile:
        coords = csv.reader(csvfile, delimiter=',', quotechar='|')
        return list(map(lambda x: list(map(lambda y: int(y), x)), coords))


def main():
    ML_data = load_test()
    N = len(ML_data)
    # print(ML_data[:10]) # .columns.values)
    chrom_n = 200
    gen_n = 200

    der_Übermensch = genteik.evolve(N, chrom_n, gen_n, ML_data)
    print(der_Übermensch)

if __name__ == '__main__':
    main()
