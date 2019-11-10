import csv
import genetic_functions as genteik

'''
def load_test():
    with open('Mona_Lisa_1000.csv', newline='') as csvfile:
        coords = csv.reader(csvfile, delimiter=',', quotechar='|')
        return list(map(lambda x: list(map(lambda y: int(y), x)), coords))
'''


def load():
    coords = list()
    with open('xqf131.csv', newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in file:
            if row[0].isdigit() is False:
                if row[0] == "OPTIMAL":
                    optimal = int(row[2])
                else:
                    continue
            else:
                coord = (int(row[1]), int(row[2]))
                coords.append(coord)

    return coords, optimal


def main():
    data, optimal = load()
    N = len(data)
    # print(ML_data[:10]) # .columns.values)
    chrom_n = 400
    gen_n = 400

    der_Übermensch = genteik.evolve(N, chrom_n, gen_n, data, optimal)
    print(der_Übermensch)

if __name__ == '__main__':
    main()
