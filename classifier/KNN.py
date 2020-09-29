from classifier.DataSet import DataSet
from sklearn.preprocessing import StandardScaler

class KNN:
    distance_function: callable
    train_data: DataSet

    def __init__(self, train_data: DataSet, distance_function: callable, ignored_props: set = set()):
        self.train_data = train_data
        self.distance_function = distance_function
        self.ignored_props = ignored_props
        self.X_raw = []
        self.Y = []
        for example in train_data.examples:
            prop_values, class_value = self.__split(example)
            self.X_raw.append(prop_values)
            self.Y.append(class_value)
        self.scaler = StandardScaler().fit(self.X_raw)
        self.X = self.scaler.transform(self.X_raw).tolist()

    def __split(self, example: list):
        prop_values = []
        class_value = None
        for prop in self.train_data.properties:
            if prop != self.train_data.classification_prop:
                if prop not in self.ignored_props:
                    prop_values.append(example[self.train_data.properties[prop]])
            else:
                class_value = example[self.train_data.properties[prop]]
        return prop_values, class_value

    def __find_nearest(self, distances: list, weighted: bool):
        d_dict = {}
        if weighted and distances[0][0] < 0.0001:
            distances = [x for x in distances if x[0] < 0.0001]
            weighted = False
        for d_tuple in distances:
            w = 1
            if weighted:
                w /= (d_tuple[0] ** 2)
            d_dict[d_tuple[1]] = d_dict.setdefault(d_tuple[1], 0) + w

        max_weight = 0
        max_props = []
        for prop in d_dict:
            if d_dict[prop] > max_weight:
                max_props = [prop]
                max_weight = d_dict[prop]
            elif d_dict[prop] == max_weight:
                max_props.append(prop)

        if len(max_props) > 1:
            return None
        else:
            return max_props[0]

    def classify_example(self, example: list, k: int, weighted: bool):
        example = self.scaler.transform([example]).tolist()[0]
        distances = []
        for i in range(len(self.X)):
            d = self.distance_function(example, self.X[i])
            distances.append((d, self.Y[i]))
        distances.sort(key=lambda x: x[0])
        distances = distances[:k]
        return self.__find_nearest(distances, weighted)

    def classify(self, test_set: DataSet, k: int, weighted: bool):
        for element in test_set.examples:
            prop_values, class_value = self.__split(element)
            result = self.classify_example(prop_values, k, weighted)
            print(class_value + " : " + (result if result != None else "None"))
