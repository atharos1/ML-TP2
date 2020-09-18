import csv
from random import shuffle

from classifier.ClassificationClass import ClassificationClass


class DataSet:
    classes: dict
    classes_relative_frequencies: dict
    examples: list
    properties: dict
    class_param_index: int

    def __init__(self, examples: list, properties: dict, classes: dict, class_param_index: int):
        self.examples = examples.copy()
        self.properties = properties.copy()
        self.classes = classes.copy()
        self.classes_relative_frequencies = {}
        self.class_param_index = class_param_index

        for example in self.examples:
            classes[example[class_param_index]].examples.append(example)

        for class_name, classification_class in self.classes.items():
            assert isinstance(classification_class, ClassificationClass)  # Para autocompletado del IDE

            self.classes_relative_frequencies[class_name] = len(classification_class.examples) / len(
                self.examples)  # TODO hay que aplicar Laplace?
            classification_class.calculate_props_relative_frequencies()

    @classmethod
    def build_train_test_set_from_csv(cls, csv_path: str, separator: str, class_param_index: int,
                                      train_set_percentage: int):
        examples = []
        properties = {}
        classes = {}

        with open(csv_path, newline='') as f:
            csv_reader = csv.reader(f, delimiter=separator)
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if line_count == 1:
                    for prop_index in range(len(row)):
                        properties[row[prop_index]] = prop_index
                    continue

                examples.append(row)
                classes.setdefault(row[class_param_index], ClassificationClass(row[class_param_index], properties))

        shuffle(examples)

        train_set_size = int(train_set_percentage * len(examples))

        return (
            cls(examples[:train_set_size], properties, classes, class_param_index),
            cls(examples[train_set_size:], properties, classes, class_param_index),
        )
