import math
from classifier.DataSet import DataSet
from classifier.KNN import KNN
import matplotlib.pyplot as plt
import numpy as np

def distance(elem1: list, elem2: list):
    d = 0
    for i in range(len(elem1)):
        d += (elem1[i] - elem2[i]) ** 2
    return math.sqrt(d)

def plot_knn(knn: KNN, k: int, weighted: bool, varRanges: list):
    [[x_min, x_max], [y_min, y_max]] = varRanges
    width = 200
    height = 200
    data = []
    for i in range(height):
        row = []
        y = (y_max - y_min) * i / height + y_min
        for j in range(width):
            x = (x_max - x_min) * j / width + x_min
            class_value = knn.classify_example([x, y], k, weighted)
            row.append(int(class_value) if class_value != None else 0)
        data.append(row)

    fig, ax = plt.subplots()
    plt.set_cmap('Spectral')
    #plt.set_cmap('gist_rainbow')
    #plt.set_cmap('jet')
    im = ax.imshow(data, interpolation='nearest')
    X = [(x[0] - x_min) * width / (x_max - x_min) for x in knn.X_raw]
    Y = [(x[1] - y_min) * height / (y_max - y_min) for x in knn.X_raw]
    ax.scatter(X+[-1], Y+[-1], c=[int(c) for c in knn.Y]+[0], s=60, linewidths=1.5, edgecolors='white')
    ticks = 5
    x_range = (x_max - x_min) / ticks
    y_range = (y_max - y_min) / ticks
    plt.xticks(np.arange(-0.5, width, width / ticks), np.arange(x_min, x_max+x_range, x_range))
    plt.yticks(np.arange(-0.5, width, width / ticks), np.arange(y_min, y_max+y_range, y_range))
    plt.xlim(-0.5, width-0.5)
    plt.ylim(-0.5, height-0.5)
    plt.colorbar(im)
    plt.show()

if __name__ == '__main__':
    train_set: DataSet
    test_set: DataSet

    discretizations_dict = {}
    discretizations_dict["wordcount".upper()] = lambda val: float(val)
    discretizations_dict["titleSentiment".upper()] = lambda val: float(val)
    discretizations_dict["textSentiment".upper()] = lambda val: float(val)
    discretizations_dict["sentimentValue".upper()] = lambda val: float(val)

    train_set, test_set = DataSet.build_train_test_set_from_csv("data/reviews_sentiment_fixed.csv", ";", "Star Rating".upper(), 1.0, discretizations_dict, True)
    ignored_props = {"textSentiment".upper(), "titleSentiment".upper()}
    knn = KNN(train_set, distance, ignored_props)
    plot_knn(knn, 5, False, [[0, 35], [-1.5, 3.5]])
    
