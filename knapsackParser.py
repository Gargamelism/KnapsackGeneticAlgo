import json


class Item:
    def __init__(self, name, weight, value):
        self.name = name
        self.weight = weight
        self.value = value

    def __str__(self):
        return f'{{"name": "{self.name}", "weight": {self.weight}, value: {self.value}}}'


def parseFile(path):
    with open(path) as knapsackFile:
        knapsackConf = json.loads(knapsackFile.read())

        return [(lambda item: Item(item['name'], item['weight'], item['value']))(item) for item in
                knapsackConf['items']]
