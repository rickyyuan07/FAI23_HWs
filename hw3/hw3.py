from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston


# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def z_score_norm(X: np.ndarray) -> np.ndarray:
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def min_max_norm(X: np.ndarray) -> np.ndarray:
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))


def normalize(X: np.ndarray) -> np.ndarray:
    # TODO: 1%
    # Normalize the input features using z-score normalization. (or min-max normalization)
    return z_score_norm(X)


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    class_to_int = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    return np.array([class_to_int[label] for label in y])


# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.n_classes = None
        self.n_features = None
        self.model_type = model_type
        # Note: weights.shape == logistic: (n_features + 1, n_classes), linear: (n_features + 1,)
        self.weights = None

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # TODO: 2%
        X = np.insert(X, 0, 1, axis=1) # add bias
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]

        if self.model_type == "logistic":
            self.weights = np.random.randn(self.n_features, self.n_classes)
            # One-hot encode the target labels of shape (n_samples, n_classes)
            y = np.eye(self.n_classes)[y]
            # gradient descent, grad.shape == (n_features + 1, n_classes)
            for _ in range(self.iterations):
                grad = self._compute_gradients(X, y)
                self.weights -= self.learning_rate * grad

        else:
            # Directly solve closed-form solution for linear regression
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        if self.model_type == "linear":
            # TODO: 2%
            # Perform a matrix multiplication between the input features (X) and the learned weights.
            y_pred = X @ self.weights

        elif self.model_type == "logistic":
            # TODO: 2%
            # Perform a matrix multiplication between the input features (X) and the learned weights, then apply the softmax function to the result.
            y_pred = self._softmax(X @ self.weights)
            y_pred = np.argmax(y_pred, axis=1)

        return y_pred


    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.model_type == "linear":
            # TODO: 3%
            # Not implemented since we directly solve the closed-form solution for linear regression.
            raise NotImplementedError
        elif self.model_type == "logistic":
            # TODO: 3%
            # Calculate the gradients for logistic regression by computing the dot product of X transposed and the difference between the one-hot
            # encoded true values and the softmax of the predicted values, then normalize by the number of samples.
            gradient = X.T @ (self._softmax(X @ self.weights) - y)
            return gradient / len(X)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        # z.shape == (n_samples, n_classes)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)


class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        # Create a mask based on the best feature and threshold that separates the samples into two groups. Then, recursively build
        # the left and right child nodes of the current node.
        mask = X[:, feature] <= threshold
        left_child = self._build_tree(X[mask], y[mask], depth + 1)
        right_child = self._build_tree(X[~mask], y[~mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            # Hint: For classification, return the most common class in the given samples.
            return np.bincount(y).argmax()
        else:
            # TODO: 1%
            # Hint: For regression, return the mean of the given samples.
            return np.mean(y)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Calculate the Gini index for the left and right samples, then compute the weighted average based on the number of samples in each group.
        # ref: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity, https://ithelp.ithome.com.tw/articles/10276079
        left_count = len(left_y)
        right_count = len(right_y)
        total_count = left_count + right_count

        left_probabilities = np.bincount(left_y) / left_count
        right_probabilities = np.bincount(right_y) / right_count

        left_gini = 1 - np.sum(left_probabilities**2)
        right_gini = 1 - np.sum(right_probabilities**2)
        weighted_gini = (left_count / total_count) * left_gini + (right_count / total_count) * right_gini
        return weighted_gini

    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Calculate the mean squared error for the left and right samples
        left_count = len(left_y)
        right_count = len(right_y)
        total_count = left_count + right_count

        left_mean = np.mean(left_y)
        right_mean = np.mean(right_y)

        left_mse = np.mean((left_y - left_mean)**2)
        right_mse = np.mean((right_y - right_mean)**2)

        # Compute the weighted average based on the number of samples in each group.
        weighted_mse = (left_count / total_count) * left_mse + (right_count / total_count) * right_mse
        return weighted_mse

    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model_type = model_type
        self.trees = []
        for _ in range(n_estimators):
            tree = DecisionTree(max_depth=max_depth, model_type=model_type)
            self.trees.append(tree)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for tree in self.trees:
            # TODO: 2%
            # Generate bootstrap indices by random sampling with replacement
            bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
            bootstrap_X = X[bootstrap_indices]
            bootstrap_y = y[bootstrap_indices]

            # Fit each tree with the corresponding samples from X and y
            tree.fit(bootstrap_X, bootstrap_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        predictions = []
        for tree in self.trees:
            # Predict the output for each tree
            tree_predictions = tree.predict(X)
            predictions.append(tree_predictions)

        # Combine the predictions based on the model type
        if self.model_type == "classifier":
            # Majority voting for classification
            return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        else:
            # Averaging for regression
            return np.mean(predictions, axis=0)


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    # Calculate the percentage of correct predictions by comparing the true and predicted labels.
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    # Calculate the mean squared error (MSE) between the true and predicted values.
    return np.mean((y_true - y_pred) ** 2)


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    logistic_regression = LinearModel(model_type="logistic")
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
