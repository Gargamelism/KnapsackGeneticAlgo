import argparse
import seaborn as seaborn
import matplotlib.pyplot as pyplot
from functools import reduce

import knapsackParser
import GeneticAlgorithm

PARENTS_COUNT = 20
MAX_GENERATIONS = 500
NO_IMPROVEMENT_GENERATIONS_COUNT = 15
MUTATIONS_COUNT = 5


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight-limit', required=True, type=int, help='knapsack weight limit')
    parser.add_argument('-f', '--knapsack-file', required=True, help='file containing knapsack configuration')
    parser.add_argument('-p', '--parents-count', default=PARENTS_COUNT, type=int, help='size of generation')
    parser.add_argument('-g', '--max-generations', default=MAX_GENERATIONS, type=int,
                        help='no matter result, stop at N generations')
    parser.add_argument('-s', '--potential-fitness', required=False, type=int,
                        help='if known, the best fitness possible')
    parser.add_argument('-t', '--no-improvements-count', default=NO_IMPROVEMENT_GENERATIONS_COUNT, type=int,
                        help='no matter result, stop at N generations')
    parser.add_argument('-m', '--mutations-count', default=MUTATIONS_COUNT, type=int, help='genetic mutations count')

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
    maxFitness = parsedArgs.potential_fitness if parsedArgs.potential_fitness else reduce(
        lambda maxFitness, item: maxFitness + item.value, availableItems, 0)
    geneticAlgorithm = GeneticAlgorithm.GeneticAlgorithm(parsedArgs.parents_count, parsedArgs.weight_limit,
                                                         availableItems, parsedArgs.max_generations, maxFitness,
                                                         parsedArgs.no_improvements_count, parsedArgs.mutations_count)

    generations = []
    for generation in geneticAlgorithm.advance():
        generations.append(generation)

    plotGenerations(generations)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
