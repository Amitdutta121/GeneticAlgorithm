from Options import Options


class Evaluator:

    def evaluate(self, chromosome):
        return Options.EVALUATOR(chromosome)

    def dejongFunction2(self, chromosome):

        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.map_10bit_binary_to_value(chromosome, fromIndex, toIndex))

        sum = 0

        for i in range(len(decodedValues) - 1):
            sum += 100 * ((decodedValues[i + 1] - decodedValues[i] ** 2) ** 2) + ((decodedValues[i] - 1) ** 2)

        # print("=======", sum)
        return 1 / (sum + 1)

    def map_10bit_binary_to_value(self, chromosome, fromIndex, toIndex):
        binary_str = ''.join(map(str, chromosome[fromIndex:toIndex]))
        decimal_value = int(binary_str, 2)

        # Define the range and precision
        min_value = -5.12
        max_value = 5.12

        # Calculate the mapped value using linear scaling
        # mapped_value = min_value + (decimal_value / ((2**13) - 1)) * (max_value - min_value)

        mapped_value = min_value + decimal_value * ((max_value - min_value) / ((2 ** 10) - 1))

        return mapped_value

    def deJongFunction1(self, chromosome):
        decodedValues = list()

        partLength = int(len(chromosome) / 10)

        for i in range(partLength):
            fromIndex = i * 10
            toIndex = (i + 1) * 10
            decodedValues.append(self.convertFromBinary(chromosome, fromIndex, toIndex, -5.12, 0.01))

        sum = 0

        for value in decodedValues:
            sum += value * value

        # print("=======", sum)
        return 1 / (sum + 1)

    def convertFromBinary(self, chromosome, fromIndex, to, min_value, precision):
        binary = ''.join(map(str, chromosome[fromIndex:to]))
        int_value = int(binary, 2)
        return min_value + int_value * precision

    def maxOnes(self, chromosome):
        return sum(chromosome)
