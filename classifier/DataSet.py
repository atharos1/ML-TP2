import copy
import csv
import math
from random import shuffle

class DataSet:
    classes: dict
    classes_relative_frequencies: dict
    examples: list
    properties: dict
    class_param_index: int
    props_possible_values: list  # lista de propiedades, contiene diccionarios con frecuencias relativas de cada posible valor

    def __init__(self, examples: list, properties: dict, class_param_index: int = None):
        self.examples = examples
        self.properties = properties
        self.class_param_index = class_param_index

        self.classes = {}
        self.classes_relative_frequencies = {}

        if class_param_index is not None:
            class_list_tmp = {}

            for example in examples:
                class_list_tmp.setdefault(example[class_param_index], []).append(example)

            for class_name in class_list_tmp.keys():
                self.classes[class_name] = DataSet(class_list_tmp[class_name], properties)

            for class_name, class_dataset in self.classes.items():
                assert isinstance(class_dataset, DataSet)  # Para autocompletado del IDE

                # TODO hay que aplicar Laplace?
                self.classes_relative_frequencies[class_name] = len(class_dataset.examples) / len(
                    self.examples) if len(self.examples) > 0 else 0

        self.__calculate_props_relative_frequencies()

    def __calculate_props_relative_frequencies(self):
        self.props_possible_values = []
        for i in range(len(self.properties)):
            self.props_possible_values.append({})

        for prop_name, prop_index in self.properties.items():
            values_dict = self.props_possible_values[prop_index]
            assert isinstance(values_dict, dict)

            values_dict.clear()

            for example in self.examples:
                values_dict[example[prop_index]] = values_dict.setdefault(example[prop_index], 0) + 1

            for prop_value, count in values_dict.items():
                values_dict[prop_value] = (count) / (len(self.examples) )

    def subset(self, attr: str, attr_val: str):
        example_subset = [ex for ex in self.examples if ex[self.properties[attr]] == attr_val]
        return DataSet(example_subset, self.properties, self.class_param_index)

    def gain(self, attr: str):
        acum = self.entropy()

        for attr_val in self.props_possible_values[self.properties[attr]].keys():
            subset = self.subset(attr, attr_val)
            acum -= (len(subset.examples) / len(self.examples)) * subset.entropy()

        return acum

    def entropy(self):
        acum = 0
        for class_name, classification_class in self.classes.items():
            acum += self.classes_relative_frequencies[class_name] * math.log2(
                self.classes_relative_frequencies[class_name])

        return -acum

    @classmethod
    def build_train_test_set_from_csv(cls, csv_path: str, separator: str, class_param_index: int,
                                      train_set_percentage: int):
        examples = []
        properties = {}

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

        shuffle(examples)

        train_set_size = int(train_set_percentage * len(examples))

        return (
            cls(examples[:train_set_size], properties, class_param_index),
            cls(examples[train_set_size:], properties, class_param_index),
        )
