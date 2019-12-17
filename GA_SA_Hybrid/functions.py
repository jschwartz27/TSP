import csv
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_parameters(file_name="parameters.json"):
    with open(file_name, "r") as file:
        data = json.load(file)

    return data


def load_data():
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


def show_graph(data, params):
    d = {
        "Type": list(),
        "Generation": list(),
        "Fitness": list()
    }
    for i in data:
        index = 0
        for j in data[i]:
            d["Type"].append(i)
            d["Generation"].append(index)
            d["Fitness"].append(j)
            index += 1

    dF = pd.DataFrame(d)
    plt.title('TSP Fitness vs. Generation (Pop: {}, Gens: {})'.format(
        params["chrom_n"], params["gen_n"]))

    sns.lineplot(
        x="Generation", y="Fitness",
        hue="Type", #style="type", 
        data=dF)

    plt.show()


def flatten(l):
    return [item for sublist in l for item in sublist]


def reverse(l):
    return l[::-1]
