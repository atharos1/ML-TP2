from classifier.DataSet import DataSet
import numpy as np


class ID3Tree:
    decision_class: str
    attr: str
    children: dict  # Valor del atributo - nodo hijo / nombre de clase (decisión)

    def __init__(self, train_data: DataSet, parent_train_data: DataSet = None, ignored_props: set = set()):
        for class_name, class_rel_freq in train_data.classes_relative_frequencies.items():
            if class_rel_freq == 1:
                self.decision_class = class_name
                return

        if len(train_data.examples) == 0:
            self.decision_class = parent_train_data.most_frequent_class
            return

        best_gain_prop = None
        best_gain_prop_val = 0
        for prop, prop_index in train_data.properties.items():
            if prop_index == train_data.class_param_index or prop in ignored_props:
                continue

            prop_gain = train_data.gain(prop)
            if prop_gain >= best_gain_prop_val:
                best_gain_prop = prop
                best_gain_prop_val = prop_gain

        if best_gain_prop is None:
            self.decision_class = train_data.most_frequent_class()
            return

        self.decision_class = None
        self.attr = best_gain_prop
        self.children = {}

        ignored_props_updated = ignored_props.copy()
        ignored_props_updated.add(self.attr)
        for attr_val in train_data.props_possible_values[train_data.properties[self.attr]].keys():
            self.children[attr_val] = ID3Tree(train_data.subset(self.attr, attr_val), train_data, ignored_props_updated)

    def classify_example(self, example: list, structure_dataset: DataSet):
        if self.decision_class is not None:
            return self.decision_class

        return self.children[example[structure_dataset.properties[self.attr]]].classify_example(example,
                                                                                                structure_dataset)

    def classify_set(self, test_set: DataSet):
        results = []
        for i in range(len(test_set.examples)):
            results.append(self.classify_example(test_set.examples[i], test_set))

        return results

    def confusion_matrix(self, test_set: DataSet, full_class_list: list):
        confusion_matrix = {c: {c: 0 for c in full_class_list} for c in full_class_list}

        for example in test_set.examples:
            processed_result = self.classify_example(example, test_set)
            confusion_matrix[example[test_set.class_param_index]][processed_result] += 1

        return confusion_matrix
