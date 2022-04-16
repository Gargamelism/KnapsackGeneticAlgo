from random import randint, sample
from functools import partial
from typing import List

from knapsackParser import Item


class GeneticAlgorithm:
    def __init__(self, parentsCount, weightLimit, items, maxGenerations, potentialFitness,
                 noImprovementGenerationsCount, mutationsCount):
        self.parentsCount = parentsCount
        self.weightLimit = weightLimit
        self.mutationsCount = mutationsCount

        self.maxGenerations = maxGenerations
        self.potentialFitness = potentialFitness
        self.noImprovementGenerations = noImprovementGenerationsCount
        self.noImprovementGenerationsCount = 0

        self.items = items
        self.itemsCount = len(items)

        self.generations = []
        self.__initFirstGeneration()

    def __toKnapsack(self, knapsackConfig):
        fitness = GeneticAlgorithm.__calcKnapsackFitness(knapsackConfig, self.items, self.weightLimit)

        return {
            'fitness': fitness,
            'knapsack': list(
                map(lambda isIncluded, item: str(item) if isIncluded else None, knapsackConfig, self.items))
        }

    def generation(self, idx=-1):
        try:
            return list(
                map(self.__toKnapsack, self.generations[idx])
            )
        except IndexError:
            return None

    def __createRandomSack(self):
        return [True if randint(0, 1) == 1 else False for _ in range(self.itemsCount)]

    def __initFirstGeneration(self):
        self.generations.append([self.__createRandomSack() for _ in range(self.parentsCount)])

    def __hasExceededPotentialFitness(self):
        latestGenerationFitnesses = self.__getGenerationFitnesses(self.generations[-1])
        latestGenerationHighestFitness = sorted(latestGenerationFitnesses, reverse=True)[0]

        # user can stop before "best possible result"
        return latestGenerationHighestFitness >= self.potentialFitness

    def __hasExceededNoImprovementLimit(self):
        return len(self.generations) > self.noImprovementGenerations <= self.noImprovementGenerationsCount

    def __shouldStopGenerationsCondition(self):
        shouldStop = True
        if len(self.generations) >= self.maxGenerations:
            print('Done: reached max generations')
        elif self.__hasExceededPotentialFitness():
            print(f'Done: exceeded potential fitness <{self.potentialFitness}>')
        elif self.__hasExceededNoImprovementLimit():
            print(f'Done: exceeded no improvement in <{self.noImprovementGenerationsCount}>')
        else:
            shouldStop = False

        return shouldStop

    @staticmethod
    def __calcKnapsackFitness(knapsackConfig, knapsackItems: List[Item], maxWeight):
        totalWeight = 0
        totalValue = 0
        for idx, isInKnapsack in enumerate(knapsackConfig):
            if (isInKnapsack):
                totalWeight += knapsackItems[idx].weight
                totalValue += knapsackItems[idx].value

        return totalValue if maxWeight == 0 or totalWeight < maxWeight else 0

    def __crossoverGeneration(self, generation):
        crossedGenerations = list(map(lambda knapsackA, knapsackB:
                                      knapsackA[len(knapsackA) // 2:] +
                                      knapsackB[:len(knapsackB) // 2],
                                      generation,
                                      generation[1:]))

        # joining in this manner will always produce n-1 lists
        crossedGenerations.append(generation[-1])

        return crossedGenerations

    @staticmethod
    def __flipNCells(knapsack, count):
        flipIdxs = sample(range(0, len(knapsack)), count)

        newKnapsack = knapsack.copy()
        for idx in flipIdxs:
            newKnapsack[idx] = not newKnapsack[idx]

        return newKnapsack

    def __mutateGeneration(self, generation, mutationsCount):
        return list(map(partial(self.__flipNCells, count=mutationsCount), generation))

    def __getGenerationFitnesses(self, generation):
        return list(
            map(lambda knapsackConfig: self.__calcKnapsackFitness(knapsackConfig, self.items, self.weightLimit),
                generation)
        )

    def __updateNoImprovementCount(self, prevGenerationFitnesses, nextGenerationFitnesses):
        prevGenerationHighest = sorted(prevGenerationFitnesses, reverse=True)[0]
        nextGenerationHighest = sorted(nextGenerationFitnesses, reverse=True)[0]

        if (prevGenerationHighest >= nextGenerationHighest):
            self.noImprovementGenerationsCount += 1
        else:
            self.noImprovementGenerationsCount = 0

    def advance(self):
        while (not self.__shouldStopGenerationsCondition()):
            currentGeneration = self.generations[-1]
            currentGenerationFitnesses = self.__getGenerationFitnesses(currentGeneration)
            generationByFitness = [knapsack for _, knapsack in
                                   sorted(zip(currentGenerationFitnesses, currentGeneration))]

            # crossover first half
            crossedLowerFitness = self.__crossoverGeneration(generationByFitness[:len(generationByFitness) // 2])
            # mutate second half
            mutatedHigherFitness = self.__mutateGeneration(generationByFitness[len(generationByFitness) // 2:],
                                                           self.mutationsCount)
            self.generations.append(crossedLowerFitness + mutatedHigherFitness)

            newGeneration = self.generations[-1]
            newGenerationFitnesses = self.__getGenerationFitnesses(newGeneration)
            self.__updateNoImprovementCount(currentGenerationFitnesses, newGenerationFitnesses)

            yield self.generation(-1)
