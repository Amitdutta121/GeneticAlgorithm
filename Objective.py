import math
import numpy as np


def dejongReverse(x):
    return (1 / x) - 1

def stepdejongReverse(x):
    return (30 / x) - 30


def read_city_coordinates_from_file(file_path):
    city_coordinates = {}
    in_coordinates_section = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_coordinates_section = True
                continue

            if in_coordinates_section and line:
                parts = line.split()
                if len(parts) == 3:
                    city_name = int(parts[0])  # Assuming city names are numbers
                    x = float(parts[1])
                    y = float(parts[2])
                    city_coordinates[city_name] = (x, y)
                else:
                    break  # Exit the loop if there is an invalid line

    return city_coordinates
def tsp_actual_distance(chromosome):
    # Calculate the total distance of the TSP tour
    city_coordinates = read_city_coordinates_from_file("berlin52.tsp")

    global current_city
    total_distance = 0
    num_cities = len(city_coordinates)

    for i in range(num_cities - 1):
        current_city = chromosome[i]
        next_city = chromosome[i + 1]
        x1, y1 = city_coordinates[current_city]
        x2, y2 = city_coordinates[next_city]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance

    # Add the distance from the last city back to the starting city
    first_city = chromosome[0]
    x1, y1 = city_coordinates[current_city]
    x2, y2 = city_coordinates[first_city]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    total_distance += distance

    return total_distance
