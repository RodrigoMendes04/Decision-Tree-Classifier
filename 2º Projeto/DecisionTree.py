from math import log2
from collections import Counter
import numpy as np

class DecisionTreeClassifier:

    def __init__(self, data_frame):
        """
        Initialize the decision tree classifier.
        :param data_frame: Complete dataset.
        """
        data_frame = self._transform_boolean_columns(data_frame)
        available_attributes = data_frame.columns.tolist()[1:-1]  # Exclude the first and last columns
        self.decision_tree = self._construct_tree(data_frame, data_frame, available_attributes)

    def classify(self, data_frame):
        """
        Classify the target value for each row in the provided DataFrame.
        :param data_frame: DataFrame containing the input data.
        :return: List of classified target values.
        """
        results = []
        for _, row in data_frame.iterrows():
            result = self._navigate_tree(self.decision_tree, row)
            if result is None:
                results.append(result)
                continue
            results.append(result[0])
        return results

    def _navigate_tree(self, tree, row):
        """
        Navigate the decision tree recursively to classify the target value for the given input row.
        :param tree: Decision tree represented as a nested dictionary.
        :param row: A row of the dataset.
        :return: The classified target value based on the decision tree.
                Returns None if the value is not present in the tree or an error occurs.
        """
        for attribute, subtree in tree.items():
            value = row[attribute]
            if isinstance(value, bool):
                value = str(value)
            if isinstance(subtree, dict):
                if isinstance(value, str) and value not in subtree:
                    return None  # Value not present in the tree, return None
                elif not isinstance(value, str):
                    # Handling numerical attributes
                    split_key = list(subtree.keys())[0]
                    split_operator, split_value = split_key.split(' ')

                    if split_operator == '<=':
                        try:
                            if float(value) <= float(split_value):
                                subtree = subtree['<= ' + split_value]
                            else:
                                subtree = subtree['> ' + split_value]
                        except ValueError:
                            return None  # Non-numeric value, skip the comparison
                    elif split_operator == '>':
                        try:
                            if float(value) > float(split_value):
                                subtree = subtree['> ' + split_value]
                            else:
                                subtree = subtree['<= ' + split_value]
                        except ValueError:
                            return None  # Non-numeric value, skip the comparison
                    else:
                        return None  # Invalid split operator
                else:
                    subtree = subtree[value]

                if isinstance(subtree, dict):
                    return self._navigate_tree(subtree, row)
                else:
                    return subtree
            else:
                return subtree

    def _construct_tree(self, df, data, attributes):
        """
        Recursively constructs a decision tree based on the provided training data and attributes.
        :param df: DataFrame containing the training data.
        :param data: Subset of the training data for the current node.
        :param attributes: A list of attribute names available for splitting the data.
        :return: A nested dictionary representing the decision tree.
        """
        labels = data[data.columns[-1]].tolist()
        class_counts = Counter(labels)

        # Base cases
        if len(set(labels)) == 1:
            return [labels[0], class_counts[labels[0]]]  # Return [class_label, class_count]
        if len(attributes) == 0:
            return [class_counts.most_common(1)[0][0], len(labels)]  # Return [class_label, total_count]

        best_attribute = self._select_best_attribute(data, attributes)
        node = {best_attribute: {}}

        attribute_values = df[best_attribute].unique()
        if data[best_attribute].dtype == 'int64' or data[best_attribute].dtype == 'float64':
            best_split_value = self._find_best_split_value(data, best_attribute)
            subset1 = data[data[best_attribute] <= best_split_value]
            subset2 = data[data[best_attribute] > best_split_value]
            remaining_attributes = attributes.copy()
            remaining_attributes.remove(best_attribute)
            node[best_attribute]['<= ' + str(best_split_value)] = self._construct_tree(df, subset1, remaining_attributes)
            node[best_attribute]['> ' + str(best_split_value)] = self._construct_tree(df, subset2, remaining_attributes)
        else:
            for value in attribute_values:
                subset = data[data[best_attribute] == value]
                if len(subset) == 0:
                    node[best_attribute][value] = [class_counts.most_common(1)[0][0], 0]
                else:
                    remaining_attributes = attributes.copy()
                    remaining_attributes.remove(best_attribute)
                    node[best_attribute][value] = self._construct_tree(df, subset, remaining_attributes)

        return node

    @staticmethod
    def _compute_entropy(labels):
        """
        Computes the entropy of a list of labels.
        :param labels: List of labels.
        :return: Entropy value.
        """
        label_counts = Counter(labels)
        total_samples = len(labels)
        entropy = 0

        for count in label_counts.values():
            probability = count / total_samples
            entropy -= probability * log2(probability)

        return entropy

    def _select_best_attribute(self, data, attributes):
        """
        Selects the best attribute to split the data based on information gain.
        :param data: DataFrame containing data.
        :param attributes: List of attribute names available for splitting.
        :return: Name of the best attribute.
        """
        entropy_s = self._compute_entropy(data[data.columns[-1]].tolist())
        information_gains = []

        for attribute in attributes:
            entropy_attribute = self._compute_attribute_entropy(data, attribute)
            information_gain = entropy_s - entropy_attribute
            information_gains.append(information_gain)

        best_attribute_index = information_gains.index(max(information_gains))
        return attributes[best_attribute_index]

    def _compute_attribute_entropy(self, data, attribute):
        """
        Computes the entropy of an attribute in the data.
        :param data: DataFrame containing data.
        :param attribute: Name of the attribute.
        :return: Entropy value of the attribute.
        """
        attribute_values = data[attribute].unique()
        entropy_attribute = 0

        for value in attribute_values:
            subset = data[data[attribute] == value]
            subset_labels = subset[data.columns[-1]].tolist()
            subset_entropy = self._compute_entropy(subset_labels)
            subset_probability = len(subset_labels) / len(data)
            entropy_attribute += subset_probability * subset_entropy

        return entropy_attribute

    def _find_best_split_value(self, data, attribute):
        """
        Finds the best split value for a numerical attribute based on the information gain.
        :param data: DataFrame containing data.
        :param attribute: Name of the numerical attribute.
        :return: The best split value for the attribute.
        """
        attribute_values = data[attribute].unique()
        best_split_value = None
        best_information_gain = float('-inf')

        if len(attribute_values) == 1:
            # All instances have the same value for the attribute
            return attribute_values[0]

        for value in attribute_values:
            subset1 = data[data[attribute] <= value]
            subset2 = data[data[attribute] > value]

            labels1 = subset1[data.columns[-1]].tolist()
            labels2 = subset2[data.columns[-1]].tolist()

            information_gain = self._compute_information_gain(data[data.columns[-1]].tolist(), labels1, labels2)

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_split_value = value

        return best_split_value

    def _compute_information_gain(self, parent_labels, labels1, labels2):
        """
        Computes the information gain by splitting the parent labels into two subsets.
        :param parent_labels: List of labels from the parent node.
        :param labels1: List of labels from one subset.
        :param labels2: List of labels from the other subset.
        :return: The information gain.
        """
        parent_entropy = self._compute_entropy(parent_labels)
        weight1 = len(labels1) / len(parent_labels)
        weight2 = len(labels2) / len(parent_labels)
        entropy1 = self._compute_entropy(labels1)
        entropy2 = self._compute_entropy(labels2)
        information_gain = parent_entropy - (weight1 * entropy1) - (weight2 * entropy2)
        return information_gain

    def __str__(self):
        """
        Return a string representation of the decision tree.
        :return: A string representation of the decision tree.
        """
        return self._tree_to_string(self.decision_tree)

    def _tree_to_string(self, tree=None, indent=''):
        """
        Converts the decision tree to a string representation.
        :param tree: The tree to convert (default is the instance's tree).
        :param indent: The indentation string.
        :return: The string representation of the tree.
        """
        if tree is None:
            tree = self.decision_tree

        result = ""
        if isinstance(tree, dict):
            for key, value in tree.items():
                if isinstance(value, dict):
                    result += f'{indent}{key}:\n'
                    result += self._tree_to_string(value, indent + '  ')
                else:
                    result += f'{indent}{key}: {value[0]}  ({value[1]})\n'
        return result

    @staticmethod
    def _transform_boolean_columns(data_frame):
        """
        Transforms boolean columns in a DataFrame to string type.
        :param data_frame: The DataFrame to transform.
        :return: The transformed DataFrame.
        """
        boolean_columns = data_frame.select_dtypes(include=bool).columns

        for column in boolean_columns:
            data_frame[column] = data_frame[column].map({False: 'False', True: 'True'})

        return data_frame

    @staticmethod
    def accuracy_score(y_true, y_pred):
        """
        Computes the accuracy score between true labels and predicted labels.
        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The accuracy score.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        correct_predictions = 0
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:
                correct_predictions += 1
        return correct_predictions / len(y_true)
