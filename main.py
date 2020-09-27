from classifier.DataSet import DataSet
from classifier.ID3 import ID3Tree


def print_table(table):
    rows = list(table.keys())
    columns = list(table[rows[0]].keys())

    matrix = ''.ljust(14)[:14] + ' '
    for col in columns:
        matrix += col.ljust(14)[:14] + ' '

    for row in rows:
        matrix += '\n' + row.ljust(14)[:14] + ' '
        for col in columns:
            matrix += str(table[row][col]).ljust(14)[:14] + ' '

    print(matrix)


if __name__ == '__main__':
    train_set: DataSet
    test_set: DataSet

    # train_set, test_set = DataSet.build_train_test_set_from_csv("data/titanic.csv", "\t", "survived", 0.7)
    train_set, test_set = DataSet.build_train_test_set_from_csv("data/tennis.csv", ";", "juega", 1)

    full_class_list = {**train_set.classes, **test_set.classes}.keys()

    ignored_props_set = set()
    ignored_props_set.add("dia")

    tree = ID3Tree(train_set, ignored_props=ignored_props_set)
    # print_table(tree.confusion_matrix(test_set, full_class_list))

    tree.print()
