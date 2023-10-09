import math
import random

from Options import Options


class Evaluator:

    def calculate_cost(self, room_type, area):
        # Define cost constants for each room type
        cost_constants = {
            'living': 1.0,  # Cost per unit area for living room
            'kitchen': 2.0,  # Cost per unit area for kitchen (twice the cost)
            'bedroom': 1.0,  # Cost per unit area for bedroom
            'hall': 1.5,  # Cost per unit area for hall
            'bathroom': 2.0,  # Cost per unit area for bathroom (twice the cost)
        }

        # Calculate the cost for the given room type
        cost = cost_constants[room_type] * area

        return cost


    def sushilEvaluation(self, chromosome):
        num_bits_per = 4
        livingLength = chromosome[0:4 * num_bits_per]
        livingHeight = chromosome[4 * num_bits_per:8 * num_bits_per]
        kitchenLength = chromosome[8 * num_bits_per:12 * num_bits_per]
        kitchenHeight = chromosome[12 * num_bits_per:15 * num_bits_per]
        hallWidth = chromosome[15 * num_bits_per:17 * num_bits_per]
        bed1Length = chromosome[17 * num_bits_per:20 * num_bits_per]
        bed1Height = chromosome[20 * num_bits_per:23 * num_bits_per]
        bed2Length = chromosome[23 * num_bits_per:27 * num_bits_per]
        bed2Height = chromosome[27 * num_bits_per:31 * num_bits_per]
        bed3Length = chromosome[31 * num_bits_per:35 * num_bits_per]
        bed3Height = chromosome[35 * num_bits_per:39 * num_bits_per]

        living_length = self.map_10bit_binary_to_value_without_index(chromosome, 8, 20, 4 * num_bits_per)
        living_width = self.map_10bit_binary_to_value_without_index(chromosome, 8, 20, 4 * num_bits_per)
        kitchen_length = self.map_10bit_binary_to_value_without_index(chromosome, 6, 18, 4 * num_bits_per)
        kitchen_width = self.map_10bit_binary_to_value_without_index(chromosome, 6, 18, 4 * num_bits_per)
        hall_width = self.map_10bit_binary_to_value_without_index(chromosome, 3.5, 6, 2 * num_bits_per)
        berdroom1_length = self.map_10bit_binary_to_value_without_index(chromosome, 10, 17, 3 * num_bits_per)
        berdroom1_width = self.map_10bit_binary_to_value_without_index(chromosome, 10, 17, 3 * num_bits_per)
        berdroom2_length = self.map_10bit_binary_to_value_without_index(chromosome, 9, 20, 4 * num_bits_per)
        berdroom2_width = self.map_10bit_binary_to_value_without_index(chromosome, 9, 20, 4 * num_bits_per)
        berdroom3_length = self.map_10bit_binary_to_value_without_index(chromosome, 8, 18, 4 * num_bits_per)
        berdroom3_width = self.map_10bit_binary_to_value_without_index(chromosome, 8, 18, 4 * num_bits_per)


        reward = living_length*living_width + 2*kitchen_length*kitchen_width + 93.5*(46.75) + 5.5*(living_width-8.5)+ kitchen_width*berdroom1_width+ berdroom2_length* berdroom2_width+ berdroom2_length*berdroom3_width
        return 1/(reward+1)

    def decode_floor_planning(self, chromosome):

        num_bits_per = 1
        livingLength = chromosome[0:4 * num_bits_per]
        livingHeight = chromosome[4 * num_bits_per:8 * num_bits_per]
        kitchenLength = chromosome[8 * num_bits_per:12 * num_bits_per]
        kitchenHeight = chromosome[12 * num_bits_per:15 * num_bits_per]
        hallWidth = chromosome[15 * num_bits_per:17 * num_bits_per]
        bed1Length = chromosome[17 * num_bits_per:20 * num_bits_per]
        bed1Height = chromosome[20 * num_bits_per:23 * num_bits_per]
        bed2Length = chromosome[23 * num_bits_per:27 * num_bits_per]
        bed2Height = chromosome[27 * num_bits_per:31 * num_bits_per]
        bed3Length = chromosome[31 * num_bits_per:35 * num_bits_per]
        bed3Height = chromosome[35 * num_bits_per:39 * num_bits_per]

        living_length = self.map_10bit_binary_to_value_without_index(livingLength, 8, 20, 4 * num_bits_per)
        living_width = self.map_10bit_binary_to_value_without_index(livingHeight, 8, 20, 4 * num_bits_per)
        kitchen_length = self.map_10bit_binary_to_value_without_index(kitchenLength, 6, 18, 4 * num_bits_per)
        kitchen_width = self.map_10bit_binary_to_value_without_index(kitchenHeight, 6, 18, 4 * num_bits_per)
        hall_width = self.map_10bit_binary_to_value_without_index(hallWidth, 3.5, 6, 2 * num_bits_per)
        berdroom1_length = self.map_10bit_binary_to_value_without_index(bed1Length, 10, 17, 3 * num_bits_per)
        berdroom1_width = self.map_10bit_binary_to_value_without_index(bed1Height, 10, 17, 3 * num_bits_per)
        berdroom2_length = self.map_10bit_binary_to_value_without_index(bed2Length, 9, 20, 4 * num_bits_per)
        berdroom2_width = self.map_10bit_binary_to_value_without_index(bed2Height, 9, 20, 4 * num_bits_per)
        berdroom3_length = self.map_10bit_binary_to_value_without_index(bed3Length, 8, 18, 4 * num_bits_per)
        berdroom3_width = self.map_10bit_binary_to_value_without_index(bed3Height, 8, 18, 4 * num_bits_per)

        # Calculate the areas for each room
        living_area = living_length * living_width
        kitchen_area = kitchen_length * kitchen_width
        hall_area = 5.5 * hall_width
        berdroom1_area = berdroom1_length * berdroom1_width
        berdroom2_area = berdroom2_length * berdroom2_width
        berdroom3_area = berdroom3_length * berdroom3_width

        # Calculate the total area of the rooms
        total_area = living_area + 2 * kitchen_area + hall_area + berdroom1_area + berdroom2_area + berdroom3_area + 5.5 * 8.5 * 2

        # Calculate the required space for the doorway between bed2 and bed3

        # Calculate the total area including the doorway space
        total_area_with_doorway = total_area

        living_ratio = living_length / living_width
        bed1_ratio = berdroom1_length / berdroom1_width
        bed2_ratio = berdroom2_length / berdroom2_width
        bed3_ratio = berdroom3_length / berdroom3_width

        print("Living Room: ", living_length, "x", living_width, " = ", living_area, "Proportion: ", living_ratio)
        print("Kitchen: ", kitchen_length, "x", kitchen_width, " = ", kitchen_area,)
        print("Hall: ", 5.5, "x", hall_width, " = ", hall_area)
        print("Bedroom 1: ", berdroom1_length, "x", berdroom1_width, " = ", berdroom1_area, "Proportion: ", bed1_ratio)
        print("Bedroom 2: ", berdroom2_length, "x", berdroom2_width, " = ", berdroom2_area, "Proportion: ", bed2_ratio)
        print("Bedroom 3: ", berdroom3_length, "x", berdroom3_width, " = ", berdroom3_area, "Proportion: ", bed3_ratio)
        print("Total Area: ", total_area)
        print("COST: ", total_area_with_doorway)

        return total_area_with_doorway

    def floorPlanning(self, chromosome):
        num_bits_per = 1
        livingLength = chromosome[0:4*num_bits_per]
        livingHeight = chromosome[4*num_bits_per:8*num_bits_per]
        kitchenLength = chromosome[8*num_bits_per:12*num_bits_per]
        kitchenHeight = chromosome[12*num_bits_per:15*num_bits_per]
        hallWidth = chromosome[15*num_bits_per:17*num_bits_per]
        bed1Length = chromosome[17*num_bits_per:20*num_bits_per]
        bed1Height = chromosome[20*num_bits_per:23*num_bits_per]
        bed2Length = chromosome[23*num_bits_per:27*num_bits_per]
        bed2Height = chromosome[27*num_bits_per:31*num_bits_per]
        bed3Length = chromosome[31*num_bits_per:35*num_bits_per]
        bed3Height = chromosome[35*num_bits_per:39*num_bits_per]

        living_length = self.map_10bit_binary_to_value_without_index(livingLength,  8, 20, 4*num_bits_per)
        living_width = self.map_10bit_binary_to_value_without_index(livingHeight,  8, 20, 4*num_bits_per)
        kitchen_length = self.map_10bit_binary_to_value_without_index(kitchenLength, 6, 18, 4*num_bits_per)
        kitchen_width = self.map_10bit_binary_to_value_without_index(kitchenHeight,  6, 18, 4*num_bits_per)
        hall_width = self.map_10bit_binary_to_value_without_index(hallWidth, 3.5, 6, 2*num_bits_per)
        berdroom1_length = self.map_10bit_binary_to_value_without_index(bed1Length,  10, 17, 3*num_bits_per)
        berdroom1_width = self.map_10bit_binary_to_value_without_index(bed1Height,  10, 17, 3*num_bits_per)
        berdroom2_length = self.map_10bit_binary_to_value_without_index(bed2Length,  9, 20, 4*num_bits_per)
        berdroom2_width = self.map_10bit_binary_to_value_without_index(bed2Height,  9, 20, 4*num_bits_per)
        berdroom3_length = self.map_10bit_binary_to_value_without_index(bed3Length,  8, 18, 4*num_bits_per)
        berdroom3_width = self.map_10bit_binary_to_value_without_index(bed3Height, 8, 18, 4*num_bits_per)

        # Define the minimum and maximum areas for each room
        living_area_min = 120
        living_area_max = 300
        kitchen_area_min = 50
        kitchen_area_max = 120
        hall_area_min = 19
        hall_area_max = 72
        berdroom_area_min = 100
        berdroom_area_max = 180

        # Calculate the areas for each room
        living_area = living_length * living_width
        kitchen_area = kitchen_length * kitchen_width
        hall_area = 5.5 * hall_width
        berdroom1_area = berdroom1_length * berdroom1_width
        berdroom2_area = berdroom2_length * berdroom2_width
        berdroom3_area = berdroom3_length * berdroom3_width

        # Calculate the total area of the rooms
        total_area = living_area + 2 * kitchen_area + hall_area + berdroom1_area + berdroom2_area + berdroom3_area

        # Calculate the required space for the doorway between bed2 and bed3
        doorway_space = 3.0
        cost = 0.0

        # Calculate the total area including the doorway space
        total_area_with_doorway = total_area

        # Define the desired proportions for specific rooms
        living_proportion = 1.5
        bed1_proportion = 1.5
        bed2_proportion = 1.5
        bed3_proportion = 1.5

        # Calculate the proportions of specific rooms
        living_ratio = living_length / living_width
        bed1_ratio = berdroom1_length / berdroom1_width
        bed2_ratio = berdroom2_length / berdroom2_width
        bed3_ratio = berdroom3_length / berdroom3_width

        # if living_ratio != living_proportion:
        #     cost = cost + 30000
        # if bed1_ratio != bed1_proportion:
        #     cost = cost + 30000
        # if bed2_ratio != bed1_proportion:
        #     cost = cost + 30000
        # if bed3_ratio != bed1_proportion:
        #     cost = cost + 30000

        if living_area < living_area_min:
            cost = cost + 100
        if living_area > living_area_max:
            cost = cost + 100
        if kitchen_area < kitchen_area_min:
            cost = cost + 100
        if kitchen_area > kitchen_area_max:
            cost = cost + 100
        if hall_area < hall_area_min:
            cost = cost + 100
        if hall_area > hall_area_max:
            cost = cost + 100
        if berdroom1_area < berdroom_area_min:
            cost = cost + 100
        if berdroom1_area > berdroom_area_max:
            cost = cost + 100
        if berdroom2_area < berdroom_area_min:
            cost = cost + 100
        if berdroom2_area > berdroom_area_max:
            cost = cost + 100
        if berdroom3_area < berdroom_area_min:
            cost = cost + 100
        if berdroom3_area > berdroom_area_max:
            cost = cost + 100

        # Calculate penalties based on the difference from desired proportions
        proportion_penalty = (
                abs(living_ratio - living_proportion) ** 2 +
                abs(bed1_ratio - bed1_proportion) ** 2 +
                abs(bed2_ratio - bed2_proportion) ** 2 +
                abs(bed3_ratio - bed3_proportion) ** 2
        )

        # Calculate penalties based on the deviation from desired area ranges
        # area_penalty = (
        #         abs((living_area - living_area_max) ** 2) +
        #         abs((kitchen_area - kitchen_area_max) ** 2) +
        #         abs((hall_area - hall_area_max) ** 2) +
        #         abs((berdroom1_area - berdroom_area_max) ** 2) +
        #         abs((berdroom2_area - berdroom_area_max) ** 2) +
        #         abs((berdroom3_area - berdroom_area_max) ** 2) +
        #         abs((living_area_min - living_area) ** 2) +
        #         abs((kitchen_area_min - kitchen_area) ** 2) +
        #         abs((hall_area_min - hall_area) ** 2) +
        #         abs((berdroom_area_min - berdroom1_area) ** 2) +
        #         abs((berdroom_area_min - berdroom2_area) ** 2) +
        #         abs((berdroom_area_min - berdroom3_area) ** 2)
        # )

        # Calculate the cost based on the total area
        cost = total_area_with_doorway + 5.5 * 8.5 * 2  # Assuming this is a fixed cost

        # Add proportion and area penalties to the cost
        # cost +=  proportion_penalty + area_penalty
        cost +=  proportion_penalty

        reward = 1 / (cost + 1)

        return reward

    def evaluate(self, chromosome):
        return Options.EVALUATOR(chromosome)

    def function_5(self, chromosome):
        decodedValues = list()

        partLength = int(len(chromosome) / 9)

        for i in range(partLength):
            fromIndex = i * 13
            toIndex = (i + 1) * 13
            d_value = self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -65.536, 65.536, 13)
            decodedValues.append(d_value)

        a = [
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32,
             -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16,
             16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
        ]

        K = 500.0
        n = 1
        sum = 1.0 / K

        for j in range(25):
            Val = j + 1
            for i in range(n):
                Dif = decodedValues[i] - a[i][j]
                Val += Dif ** 6
            sum += 1.0 / Val

        finalSum = 1.0 / sum
        return 1 / (finalSum + 1)

    def quartic(self, chromosome):
        decodedValues = list()

        partLength = int(len(chromosome) / 9)

        for i in range(partLength):
            fromIndex = i * 9
            toIndex = (i + 1) * 9
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -1.28, 1.28, 9))

        sum = 30
        n = 1

        for i in range(n):
            sum += (i + 1) * (decodedValues[i] ** 4)

        Prd = 0.0  # Initialize Prd to 0.0

        for i in range(12):
            Prd += random.random()  # Add random numbers between 0 and 1
        Prd -= 6.0  # Adjust Prd to have a mean of 0

        sum += Prd

        # for i in range(len(decodedValues) - 1):
        #     sum += pow(decodedValues[i], 4) + random.random()
        return 1 / (sum + 1)

    def step_function(self, chromosome):
        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -5.12, 5.12, 10))

        sum = 0

        for i in range(len(decodedValues)):
            sum += int(decodedValues[i])
        return 30 / (sum + 30)

    def dejongFunction2(self, chromosome):

        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -2.048, 2.048, 10))
        # print(decodedValues)

        sum = 0

        sum += 100 * ((decodedValues[1] - decodedValues[0] ** 2) ** 2) + ((decodedValues[0] - 1) ** 2)

        if sum < 0:
            print("======================GOING NEGATIVE======================")
        return 1 / (sum + 1)

    def map_10bit_binary_to_value_without_index(self, chromosome, min_value, max_value, bits):
        binary_str = ''.join(map(str, chromosome))
        decimal_value = int(binary_str, 2)

        # Calculate the mapped value using linear scaling
        # mapped_value = min_value + (decimal_value / ((2**13) - 1)) * (max_value - min_value)

        mapped_value = min_value + decimal_value * ((max_value - min_value) / ((2 ** bits) - 1))
        return mapped_value

    def map_10bit_binary_to_value(self, chromosome, fromIndex, toIndex, min_value, max_value, bits):
        binary_str = ''.join(map(str, chromosome[fromIndex:toIndex]))
        decimal_value = int(binary_str, 2)

        # Calculate the mapped value using linear scaling
        # mapped_value = min_value + (decimal_value / ((2**13) - 1)) * (max_value - min_value)

        mapped_value = min_value + decimal_value * ((max_value - min_value) / ((2 ** bits) - 1))
        return mapped_value

    def deJongFunction1(self, chromosome):
        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -5.12, 5.12, 10))

        sum = 0

        for value in decodedValues:
            sum += value * value

        # print("=======", sum)
        return 1 / (sum + 1)

    def maxOnes(self, chromosome):
        return sum(chromosome)
