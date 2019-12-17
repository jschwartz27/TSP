import csv
import json


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
