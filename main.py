import argparse
import seaborn as seaborn
import matplotlib.pyplot as pyplot
from functools import reduce
import pandas

import knapsackParser
import GeneticAlgorithm

PARENTS_COUNT = 20
MAX_GENERATIONS = 500
NO_IMPROVEMENT_GENERATIONS_COUNT = 15
MUTATIONS_COUNT = 5
GRAPH_ROLLING_WINDOW = 50
CROSS_OVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.001


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
    parser.add_argument('-r', '--graph-rolling-window', default=GRAPH_ROLLING_WINDOW, type=int,
                        help='chunk graphs to keep it clear')
    parser.add_argument('-c', '--cross-over-probability', default=CROSS_OVER_PROBABILITY, type=float,
                        help='random draw has to be higher than this to cross over')
    parser.add_argument('-u', '--mutation-probability', default=MUTATION_PROBABILITY, type=float,
                        help='random draw has to be lower than this to mutate')

    return parser.parse_args()


def plotGenerations(generationsStats):
    formattedGenerations = {
        'fitness': [generationStats.get('max') for generationStats in generationsStats],
        'generation': list(range(1, len(generationsStats) + 1))
    }

    generationsStatsDataFrame = pandas.DataFrame(formattedGenerations)

    seaborn.set_theme(style='darkgrid')
    plot = seaborn.lineplot(x='generation', y='fitness', data=generationsStatsDataFrame.rolling(50).mean())
    plot.set_xlabel('Generations')
    plot.set_ylabel('Fitness Max')

    pyplot.show()


def main():
    parsedArgs = parseArgs()
    availableItems = knapsackParser.parseFile(parsedArgs.knapsack_file)
    maxFitness = parsedArgs.potential_fitness if parsedArgs.potential_fitness else reduce(
        lambda maxFitness, item: maxFitness + item.value, availableItems, 0)
    geneticAlgorithm = GeneticAlgorithm.GeneticAlgorithm(parsedArgs.parents_count, parsedArgs.weight_limit,
                                                         availableItems, parsedArgs.max_generations, maxFitness,
                                                         parsedArgs.no_improvements_count, parsedArgs.mutations_count,
                                                         parsedArgs.cross_over_probability,
                                                         parsedArgs.mutation_probability)

    generationsStats = []
    for generationStats in geneticAlgorithm.advance():
        generationsStats.append(generationStats)

    plotGenerations(generationsStats)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
