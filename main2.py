import math
from classifier.DataSet import DataSet
from classifier.KNN import KNN

def distance(elem1: list, elem2: list):
    d = 0
    for i in range(len(elem1)):
        d += (elem1[i] - elem2[i]) ** 2
    return math.sqrt(d)

if __name__ == '__main__':
    train_set: DataSet
    test_set: DataSet

    discretizations_dict = {}
    discretizations_dict["wordcount".upper()] = lambda val: float(val)
    discretizations_dict["titleSentiment".upper()] = lambda val: float(val)
    discretizations_dict["textSentiment".upper()] = lambda val: float(val)
    discretizations_dict["sentimentValue".upper()] = lambda val: float(val)

    train_set, test_set = DataSet.build_train_test_set_from_csv("data/reviews_sentiment_fixed.csv", ";", "Star Rating".upper(), 0.7, discretizations_dict, True)
    knn = KNN(train_set, distance)
    knn.classify(test_set, 5, False)
