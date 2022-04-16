import argparse
import seaborn as seaborn
import matplotlib.pyplot as pyplot
from functools import reduce
import numpy
from statistics import mean

import knapsackParser
import GeneticAlgorithm

PARENTS_COUNT = 20
MAX_GENERATIONS = 500
NO_IMPROVEMENT_GENERATIONS_COUNT = 15
MUTATIONS_COUNT = 5
GRAPH_CHUNK_SIZE = 50


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
    parser.add_argument('-a', '--chunk-generations-size', default=GRAPH_CHUNK_SIZE, type=int,
                        help='chunk graphs to keep it clear')

    return parser.parse_args()


def chunks(list, chunkSize):
    for i in range(0, len(list), chunkSize):
        yield list[i:i + chunkSize]


def extractSortedFitness(generation):
    return sorted(map(lambda knapsackConf: knapsackConf['fitness'], generation))


def chunkFitnesses(generationChunk):
    chunkSortedFitnesses = list(map(lambda generation: extractSortedFitness(generation), generationChunk))
    cleanChunkGroups = map(lambda fitnessesChunk:
                           filter(lambda fitness: fitness > 0, fitnessesChunk),
                           numpy.array(chunkSortedFitnesses).transpose())
    return list(
        map(lambda cleanFitnessesChunk: mean(cleanFitnessesChunk), cleanChunkGroups)
    )


def plotGenerations(generations, chunkSize=1):
    if (chunkSize >= len(generations)):
        chunkSize = 1

    # format generations for plotting
    generationsFitness = {'fitness': [], 'generation': []}
    for idx, generationChunk in enumerate(chunks(generations, chunkSize)):
        generationNum = idx * chunkSize

        for fitness in chunkFitnesses(generationChunk):
            generationsFitness['generation'].append(f'{generationNum}-{generationNum + chunkSize}')
            generationsFitness['fitness'].append(fitness)

    seaborn.set_theme(style='darkgrid')
    plot = seaborn.scatterplot(x='generation', y='fitness', data=generationsFitness)
    plot.set_xlabel('Generations Range')
    plot.set_ylabel('Fitness Mean')

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

    plotGenerations(generations, parsedArgs.chunk_generations_size)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
