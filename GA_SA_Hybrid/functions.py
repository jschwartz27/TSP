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


def show_graph(data, params, error):
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

    data_frame = pd.DataFrame(d)
    with plt.style.context('dark_background'):
        plt.title('TSP Fitness vs. Generation (Pop: {}, Gens: {}, Error: {}%;564goal)'.format(
            params["chrom_n"], params["gen_n"], error))
        flatui = ['#32cd32', "#FF00FF", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        sns.set_palette(flatui)  # "husl")
        sns.lineplot(
            x="Generation", y="Fitness",
            hue="Type",  # style="type",
            data=data_frame)

        plt.show()


def flatten(array):
    return [item for sublist in array for item in sublist]


def reverse(array):
    return array[::-1]
