from typing import Any, List

import numpy as np
from sklearn.svm import SVC


def expspace(span: list) -> np.ndarray:
    return np.exp(np.linspace(span[0], span[1], num=int(span[1]) - int(span[0]) + 1))


class SVMModel:

    def __init__(
        self,
        C: float = 0.1,
        kernel: str = "rbf",
        max_iter: int = 10000,
    ) -> None:
        """
        Initialize SVM model with either LinearSVR or SVC(depends on the kernel)

        :param C: Regularization parameter
        :type C: float(default=1.0)
        :param kernel: Either linear, rbf, poly or sigmoid
        :type kernel: str(default="rbf")
        :param max_iter: Max number of iterations for fitting
        :type max_iter: int(default=10000)
        """
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter

        self.model = (
            SVC(
                max_iter=max_iter,
                kernel="linear",
                gamma="auto",
                decision_function_shape="ovr",
                probability=True,
            )
            if self.kernel == "linear"
            else SVC(
                max_iter=max_iter,
                kernel=self.kernel,
                gamma="auto",
                decision_function_shape="ovr",
                probability=True,
            )
        )

    def fit(self, X: list, y: list) -> None:
        """
        Fit the SVM model on training data

        :param X: training features
        :type X: list
        :param y: training labels
        :type y: list
        """
        self.model.fit(X, y)

    def predict(self, X: list) -> List[Any]:
        """
        Perform inference on passed list

        :param X: The passed features to perform inference
        :type X: list

        :return: The predictions
        """
        return self.model.predict(X)  # type: ignore

    def predict_proba(self, X: list) -> List[Any]:
        """
        Perform inference on passed list, but return probabilities

        :param X: The passed features to perform inference
        :type X: list

        :return: Prediction probabilities
        """
        return self.model.predict_proba(X)  # type: ignore

    def get_model(self) -> Any:
        """
        Returns the SVC model
        """
        return self.model
