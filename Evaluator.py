import math
import random

from Options import Options


class Evaluator:

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
        n = 5

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

        sum = 6 * 5

        for i in range(len(decodedValues)):
            sum += abs(decodedValues[i])
        return 1 / (sum + 1)

    def dejongFunction2(self, chromosome):

        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -5.11, 5.12, 10))
        # print(decodedValues)

        sum = 0

        for i in range(len(decodedValues) - 1):
            sum += 100 * ((decodedValues[i + 1] - decodedValues[i] ** 2) ** 2) + ((decodedValues[i] - 1) ** 2)

        if sum < 0:
            print("======================GOING NEGATIVE======================")
        return 1 / (sum + 1)

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
