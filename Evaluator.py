import math
import random

from Options import Options


class Evaluator:
    def evaluate(self, chromosome):
        return Options.EVALUATOR(chromosome)

    def calculate_distance(self,x_i, y_i, x_j, y_j):
        RRR = 6378.388
        PI = 3.141592
        deg_x_i = int(x_i)
        min_x_i = x_i - deg_x_i
        latitude_i = (PI * (deg_x_i + 5.0 * min_x_i / 3.0)) / 180.0

        deg_y_i = int(y_i)
        min_y_i = y_i - deg_y_i
        longitude_i = (PI * (deg_y_i + 5.0 * min_y_i / 3.0)) / 180.0

        deg_x_j = int(x_j)
        min_x_j = x_j - deg_x_j
        latitude_j = (PI * (deg_x_j + 5.0 * min_x_j / 3.0)) / 180.0

        deg_y_j = int(y_j)
        min_y_j = y_j - deg_y_j
        longitude_j = (PI * (deg_y_j + 5.0 * min_y_j / 3.0)) / 180.0

        q1 = math.cos(longitude_i - longitude_j)
        q2 = math.cos(latitude_i - latitude_j)
        q3 = math.cos(latitude_i + latitude_j)
        dij = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

        return dij

    def read_city_coordinates_from_file(self, file_path):
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
    def tsp_fitness(self, chromosome):
        # Calculate the total distance of the TSP tour
        city_coordinates = self.read_city_coordinates_from_file(Options.FILE_NAME)

        global current_city
        total_distance = 0
        num_cities = len(city_coordinates)

        for i in range(num_cities - 1):
            current_city = chromosome[i]
            next_city = chromosome[i + 1]
            x1, y1 = city_coordinates[current_city]
            x2, y2 = city_coordinates[next_city]
            distance = self.calculate_distance(x1, y1, x2, y2)
            total_distance += distance

        # Add the distance from the last city back to the starting city
        first_city = chromosome[0]
        x1, y1 = city_coordinates[current_city]
        x2, y2 = city_coordinates[first_city]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        total_distance += distance

        return 1 / (total_distance + 1)
