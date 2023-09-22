import random

from Options import Options


class Evaluator:

    def evaluate(self, chromosome):
        return Options.EVALUATOR(chromosome)

    def quartic(self, chromosome):
        decodedValues = list()

        partLength = int(len(chromosome) / 9)

        for i in range(partLength):
            fromIndex = i * 9
            toIndex = (i + 1) * 9
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -1.28, 1.28, 9))

        sum = 0

        for i in range(len(decodedValues) - 1):
            sum += pow(decodedValues[i], 4) + random.random()
        return 1 / (sum + 1)

    def step_function(self, chromosome):
        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -5.12, 5.12, 10))

        sum = 6 * 5

        for i in range(len(decodedValues) - 1):
            sum += abs(decodedValues[i])
        return 1 / (sum + 1)

    def dejongFunction2(self, chromosome):

        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex, -5.12, 5.12, 10))

        sum = 0

        for i in range(len(decodedValues) - 1):
            sum += 100 * ((decodedValues[i + 1] - decodedValues[i] ** 2) ** 2) + ((decodedValues[i] - 1) ** 2)

        # print("=======", sum)
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
