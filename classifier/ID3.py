from classifier.DataSet import DataSet
import matplotlib.pyplot as plt

# TODO que pasa si entreno con un conjunto que tiene dias soleados y lluviosos, pero no nublados, y en el test set viene un día nublado?

class ID3Tree:
    decision_class: str
    attr: str
    children: dict  # Valor del atributo - nodo hijo / nombre de clase (decisión)
    train_data: DataSet
    height: int

    def __init__(self, train_data: DataSet, parent_train_data: DataSet = None, ignored_props: set = set()):
        self.height = 1
        for class_name, class_rel_freq in train_data.classes_relative_frequencies.items():
            if class_rel_freq == 1:
                self.decision_class = class_name
                return

        if len(train_data.examples) == 0:
            self.decision_class = parent_train_data.most_frequent_class
            return

        self.train_data = train_data

        best_gain_prop = None
        best_gain_prop_val = 0
        for prop, prop_index in train_data.properties.items():
            if prop == train_data.classification_prop or prop in ignored_props:
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
        ignored_props_updated.remove(self.attr)

        max_child_depth = 0
        for _, child in self.children.items():
            if child.height > max_child_depth:
                max_child_depth = child.height
        self.height += max_child_depth

    def print(self, level: int = 0):
        print('\t' * level, end='')

        if self.decision_class is not None:
            print("{[(" + self.decision_class + ")]}")
            return

        print(self.attr.upper())
        level += 1
        for attr_val, child_tree in self.children.items():
            print('\t' * level, end='')
            print(attr_val)
            child_tree.print(level + 1)

    def classify_example(self, example: list, max_depth: int = None):
        if self.decision_class is not None:
            return self.decision_class

        # TODO chequear que se hace así
        if example[self.train_data.properties[self.attr]] not in self.children.keys() or max_depth == 0:
            return self.train_data.most_frequent_class()

        if max_depth is not None:
            max_depth -= 1

        return self.children[example[self.train_data.properties[self.attr]]].classify_example(example, max_depth)

    def classify_set(self, test_set: DataSet, max_depth: int = None):
        results = []
        errors = 0
        successes = 0
        for i in range(len(test_set.examples)):
            example = test_set.examples[i]
            result = self.classify_example(example, max_depth)
            results.append(result)
            if result == example[self.train_data.properties[self.train_data.classification_prop]]:
                successes += 1
            else:
                errors += 1

        return results, successes, errors, (float(successes) / len(test_set.examples))

    def confusion_matrix(self, test_set: DataSet):
        # confusion_matrix = {c: {c: 0 for c in full_class_list} for c in full_class_list}
        confusion_matrix = {c: {c: 0 for c in self.train_data.classes.keys()} for c in self.train_data.classes.keys()}

        for example in test_set.examples:
            processed_result = self.classify_example(example)
            confusion_matrix[example[test_set.properties[test_set.classification_prop]]][processed_result] += 1

        return confusion_matrix

    def draw_precision_curve(self, datasets: {}, min_tree_depth: int = 0, max_tree_depth: int = None):
        if max_tree_depth is None:
            max_tree_depth = self.height

        colors = [
            "tab:red",
            "tab:blue",
            "tab:green",
            "tab:orange",
            "tab:purple"
        ]

        curr_color_index = 0
        for dataset_name, dataset in datasets.items():
            x_axis = []
            y_axis = []

            for depth in range(min_tree_depth, max_tree_depth + 1):
                _, _, _, success_rate = self.classify_set(dataset, depth)
                x_axis.append(depth)
                y_axis.append(success_rate)

            plt.plot(x_axis, y_axis, label=dataset_name, color=colors[curr_color_index])
            curr_color_index += 1

        plt.xlabel('Maximum tree depth')
        plt.ylabel('Success rate')
        plt.legend(loc='upper left')
        plt.show()
