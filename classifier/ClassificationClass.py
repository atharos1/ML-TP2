class ClassificationClass:
    properties_count: int
    properties: dict
    name: str
    examples: list
    props_possible_values: list  # lista de propiedades, contiene diccionarios con frecuencias relativas de cada posible valor

    def __init__(self, name: str, properties: dict):
        self.properties = properties
        self.name = name
        self.examples = []

        self.props_possible_values = []
        for i in range(len(self.properties)):
            self.props_possible_values.append({})

    def calculate_props_relative_frequencies(self):
        for prop_name, prop_index in self.properties.items():
            values_dict = self.props_possible_values[prop_index]
            assert isinstance(values_dict, dict)

            values_dict.clear()

            for example in self.examples:
                values_dict[example[prop_index]] = values_dict.setdefault(example[prop_index], 0) + 1

            for prop_value, count in values_dict.items():
                values_dict[prop_value] = (count + 1) / (len(self.examples) + 2)  # Corrección de Laplace

    def classify_example(self, example: list):
        val = 1.0
        for prop_name, prop_index in self.properties.items():
            values_dict = self.props_possible_values[prop_index]
            assert isinstance(values_dict, dict)

            val *= values_dict.setdefault(str(example[prop_index]), 1 / (len(self.examples) + 2)) # Corrección de Laplace

        return val
