import random
from random import randint, sample
from functools import partial
from typing import List
from pandas import Series

from knapsackParser import Item


class GeneticAlgorithm:
    ENDLESS_GENERATIONS = -1

    def __init__(self, parentsCount, weightLimit, items, maxGenerations, potentialFitness,
                 noImprovementGenerationsCount, mutationsCount, crossOverProbability, mutationProbability):
        self._parentsCount = parentsCount
        self._weightLimit = weightLimit
        self._mutationsCount = mutationsCount
        self._crossOverProbability = crossOverProbability
        self._mutationProbability = mutationProbability

        self._maxGenerations = maxGenerations
        self._potentialFitness = potentialFitness
        self._noImprovementGenerations = noImprovementGenerationsCount
        self._noImprovementGenerationsCount = 0

        self._items = items
        self._itemsCount = len(items)

        self._currentGeneration = []
        self._generationsStatistics = []
        self.__initFirstGeneration()

        # good for understanding what is a good value for this
        self._maxNoImprovements = 0

    def __toKnapsack(self, knapsackConfig):
        fitness = GeneticAlgorithm.__calcKnapsackFitness(knapsackConfig, self._items, self._weightLimit)

        return {
            'fitness': fitness,
            'knapsack': list(
                map(lambda isIncluded, item: str(item) if isIncluded else None, knapsackConfig, self._items))
        }

    def __fromKnapsack(self, knapsackDetails):
        return [item if item == None else True for item in knapsackDetails['knapsack']]

    @property
    def currentGeneration(self):
        try:
            return list(
                map(self.__toKnapsack, self._currentGeneration)
            )
        except IndexError:
            return None

    @currentGeneration.setter
    def currentGeneration(self, generation):
        self._currentGeneration = generation
        generationFitnesses = list(
            map(lambda knapsack: GeneticAlgorithm.__calcKnapsackFitness(knapsack, self._items, self._weightLimit),
                generation)
        )

        self._generationsStatistics.append(Series(generationFitnesses).describe())

    def __createRandomSack(self):
        return [True if randint(0, 1) == 1 else False for _ in range(self._itemsCount)]

    def __initFirstGeneration(self):
        self.currentGeneration = [self.__createRandomSack() for _ in range(self._parentsCount)]

    def __hasExceededPotentialFitness(self):
        latestGenerationHighestFitness = self._generationsStatistics[-1].get('max')

        # user can stop before "best possible result"
        return latestGenerationHighestFitness >= self._potentialFitness

    def __hasExceededNoImprovementLimit(self):
        return len(self._generationsStatistics) > self._noImprovementGenerations == self._noImprovementGenerationsCount

    def __shouldStopGenerationsCondition(self):
        shouldStop = True
        if self._maxGenerations != self.ENDLESS_GENERATIONS and len(self._generationsStatistics) >= self._maxGenerations:
            print('Done: reached max generations')
        elif self.__hasExceededPotentialFitness():
            print(f'Done: exceeded potential fitness <{self._potentialFitness}>')
        elif self.__hasExceededNoImprovementLimit():
            print(f'Done: exceeded no improvement in <{self._noImprovementGenerationsCount}>')
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

    def __updateNoImprovementCount(self, prevGenerationMaxFitness, currentGenerationMaxFitness):
        self._maxNoImprovements = max(self._noImprovementGenerationsCount, self._maxNoImprovements)

        if (prevGenerationMaxFitness >= currentGenerationMaxFitness):
            self._noImprovementGenerationsCount += 1
        else:
            self._noImprovementGenerationsCount = 0

    def __selectParents(self, currentGeneration):
        currentGenerationFitnesses = [generation['fitness'] for generation in currentGeneration]
        fitnessSum = sum(currentGenerationFitnesses)
        if fitnessSum == 0:
            currentGenerationFitnesses = None
        return random.choices(currentGeneration, weights=currentGenerationFitnesses, k=len(currentGeneration))

    def __crossPair(self, parentA, parentB):
        [startingCrossIdx, endingCrossIdx] = sorted(random.choices(range(0, len(parentA)), k=2))
        parentABlock = parentA[startingCrossIdx:endingCrossIdx]
        parentBBlock = parentB[startingCrossIdx:endingCrossIdx]

        return [parentA[:startingCrossIdx] + parentBBlock + parentA[endingCrossIdx:],
                parentB[:startingCrossIdx] + parentABlock + parentB[endingCrossIdx:], ]

    def __crossParents(self, parents):
        crossedChildren = [
            self.__crossPair(parents[0], parents[1]) if random.random() > self._crossOverProbability else [parents[0],
                                                                                                           parents[1]]
            for parents in zip(parents[::2], parents[1::2])
        ]

        # flatten pairs
        crossedChildren = [parent for parentPairs in crossedChildren for parent in parentPairs]

        # last parent in an odd list can't be crossed
        if ((len(parents) % 2) != 0):
            crossedChildren.append(parents[-1])

        return crossedChildren

    def __flipNCells(self, parent, flipCount):
        flipIdxs = sample(range(0, len(parent)), flipCount)

        child = parent.copy()
        for idx in flipIdxs:
            child[idx] = not parent[idx]

        return child

    def __mutateParents(self, parents):
        mutatedChildren = [
            self.__flipNCells(parent, self._mutationsCount) if random.random() < self._mutationProbability else parent
            for parent in parents
        ]

        return mutatedChildren

    def __mateAndMutate(self, parents):
        parentsConf = [self.__fromKnapsack(knapsack) for knapsack in parents]

        crossedChildren = self.__crossParents(parentsConf)
        mutatedChildren = self.__mutateParents(crossedChildren)

        return mutatedChildren

    def advance(self):
        while (not self.__shouldStopGenerationsCondition()):
            parents = self.__selectParents(self.currentGeneration)
            newGeneration = self.__mateAndMutate(parents)
            self.currentGeneration = newGeneration
            self.__updateNoImprovementCount(self._generationsStatistics[-2].get('max'),
                                            self._generationsStatistics[-1].get('max'))

            yield self._generationsStatistics[-1]
