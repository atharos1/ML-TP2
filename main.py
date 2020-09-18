from classifier.DataSet import DataSet

if __name__ == '__main__':
    train_set: DataSet
    test_set: DataSet

    # train_set, test_set = DataSet.build_train_test_set_from_csv("data/titanic.csv", "\t", 1, 0.7)
    train_set, test_set = DataSet.build_train_test_set_from_csv("data/tennis.csv", ";", 5, 1)

    print(train_set)
    print(test_set)
