import argparse
import seaborn as seaborn
import matplotlib.pyplot as pyplot
from functools import reduce

import knapsackParser
import GeneticAlgorithm


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--knapsack-file', required=True, help='file containing knapsack configuration')

    return parser.parse_args()


def plotGenerations(generations):
    # format generations for plotting
    generationsFitness = {'fitness': [], 'generation': []}
    for idx, generation in enumerate(generations):
        generationNum = idx + 1
        for knapsackConf in generation:
            generationsFitness['fitness'].append(knapsackConf['fitness'])
            generationsFitness['generation'].append(generationNum)

    seaborn.set_theme(style='darkgrid')
    seaborn.relplot(x='generation', y='fitness', data=generationsFitness)
    pyplot.show()


def main():
    parsedArgs = parseArgs()
    availableItems = knapsackParser.parseFile(parsedArgs.knapsack_file)
    maxFitness = reduce(lambda maxFitness, item: maxFitness + item.value, availableItems, 0)
    geneticAlgorithm = GeneticAlgorithm.GeneticAlgorithm(5, 15, availableItems, 20, maxFitness, 5, 3)

    generations = []
    for generation in geneticAlgorithm.advance():
        generations.append(generation)

    plotGenerations(generations)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
