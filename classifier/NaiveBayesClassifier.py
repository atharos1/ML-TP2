import csv

from classifier.ClassificationClass import ClassificationClass


class NaiveBayesClassifier:
    classes: dict
    classes_relative_frequencies: dict
    examples: list
    properties: dict

    # Calculo todas las probabilidades necesarias (entrenamiento)
    def __init__(self, training_file_path: str):
        self.examples = []
        self.classes = {}
        self.classes_relative_frequencies = {}
        self.properties = {}

        with open(training_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if line_count == 1:
                    for prop_index in range(len(row) - 1):
                        self.properties[row[prop_index]] = prop_index
                    continue

                self.examples.append(row)
                self.classes.setdefault(row[len(row)-1], ClassificationClass(row[len(row)-1], self.properties)).examples.append(row)

        for class_name, classification_class in self.classes.items():
            assert isinstance(classification_class, ClassificationClass)  # Para autocompletado del IDE

            self.classes_relative_frequencies[class_name] = len(classification_class.examples) / len(self.examples) # TODO hay que aplicar Laplace?
            classification_class.calculate_props_relative_frequencies()

    def classify_example(self, example: list) -> str:
        best_class = None
        best_score = 0
        all_scores = {}
        accum_score = 0
        classification_class: ClassificationClass
        for classification_class_name in self.classes:
            classification_class = self.classes[classification_class_name]
            score = classification_class.classify_example(example) * self.classes_relative_frequencies[classification_class_name]
            all_scores[classification_class_name] = score
            accum_score += score

            if score > best_score or best_class == None:
                best_class = classification_class
                best_score = score

        return best_class.name, {c: all_scores[c] / accum_score for c in all_scores}

    def test(self, test_file_path: str):
        successes = 0
        errors = 0
        confusion_matrix = {c: {c: 0 for c in self.classes} for c in self.classes}
        raw_results = {c: [] for c in self.classes}
        expected_results = []
        with open(test_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if line_count == 1:
                    continue

                processed_result, probabilities = self.classify_example(row)
                for c in self.classes:
                    raw_results[c].append(probabilities[c])
                expected_results.append(row[len(self.properties)])
                confusion_matrix[row[len(self.properties)]][processed_result] += 1

                if row[len(self.properties)] != processed_result:
                    errors += 1
                    #print(f'{row[len(self.properties)]} != {processed_result}')
                else:
                    successes += 1
            
        return successes, errors, confusion_matrix, raw_results, expected_results

