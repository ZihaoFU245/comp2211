import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB



class OurImplementedNaiveBayesCategorical:

    def __init__(self, alpha=1.0e-10):
        self.alpha = alpha
        self.priors = []
        self.likelihoods = []

    def fit(self, X, y):
        """
        Fit the Naive Bayes model.

        Parameters
        ----------
        X: array, shape (num_samples, num_features)
            2D array of training samples
        y: array, shape (num_samples,)
            1D array of training labels

        Returns
        -------
        None
        """
        # Compute priors
        self.priors = self.compute_priors(y)

        # Compute likelihoods for each feature (stored in a list)
        self.likelihoods = self.compute_likelihoods(X, y)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X: array, shape (num_samples, num_features)
            2D array of test samples

        Returns
        -------
        predicted_class: array, shape (num_samples,)
            1D array of predicted class labels for each test sample
        """
        # Compute posteriors for each test sample
        posteriors = self.compute_posteriors(self.priors, self.likelihoods, X)

        # Pick class with the highest probability
        predicted_class = np.argmax(posteriors, axis=1)
        return predicted_class

    def compute_priors(self, y):
        """
        Compute prior probabilities for each class.

        Parameters
        ----------
        y: array, shape (num_samples,)
            1D array of class labels

        Returns
        -------
        priors: array, shape (num_classes,)
            1D array of prior probabilities for each class
        """

        values = np.unique(y)  # [0 , 1] Eg [p1 ,p2]
        length = len(y)
        priors = np.empty(len(values))
        for i, value in enumerate(values):
            count = np.count_nonzero(y == value)
            p = count / length
            priors[i] = p

        return priors

    def compute_likelihoods(self, X, y):
        """
        Compute P(feature_value | class) using Laplace smoothing.

        Parameters
        ----------
        X: array, shape (num_samples, num_features)
            2D array of feature values
        y: array, shape (num_samples,)
            1D array of class labels

        Returns
        -------
        likelihoods: list, length (num_features)
            list of 2D arrays of likelihoods for each feature value and class
            (2D array shape: number of feature categories, number of classes)
        """
        # YOUR CODE HERE
        # NOTE USE VECTORIEZED NUMPY FUNCTIONS. YOU MAY NOT USE NESTED FOR LOOPS

        cols = [X[: , i] for i in range(X.shape[1])]    #1D good

        row_by_feature = [np.eye(np.max(col).astype(int) + 1)[col.astype(int), :] for col in cols ]
        feature_by_row = [np.transpose(row_feature_matrix) for row_feature_matrix in row_by_feature]

        row_by_class = np.eye(np.max(y).astype(int) + 1)[y.astype(int) , :]
        class_counts = np.array([np.sum(y == i) for i in range(np.max(y).astype(int) + 1)])

        counts = [np.dot(f_by_row , row_by_class) for f_by_row in feature_by_row]

        def calc_prob(arr : np.ndarray , class_count : np.ndarray) -> np.ndarray:
            prob = (arr + self.alpha) / (class_count + arr.shape[0] * self.alpha)
            return prob

        likelihoods = [calc_prob(count , class_counts) for count in counts]

        return likelihoods

    # TODO Task 3: Compute class probabilities
    def compute_posteriors(self, priors, likelihoods, X):
        """
        Compute posteriors for each test sample, returning shape (num_samples, num_classes).

        Parameters
        ----------
        priors: array, shape (num_classes,)
            1D array of prior probabilities for each class
        likelihoods: list, length (num_features)
            list of 2D arrays of likelihoods for each feature value and class
            (2D array shape: number of feature categories, number of classes)
        X: array, shape (num_samples, num_features)
            2D array of test samples

        Returns
        -------
        posteriors: array, shape (num_samples, num_classes)
            2D array of posteriors for each test sample and class
        """

        log_prior = np.log(priors)

        row_by_classes_log = [np.log(likelihood[X[: , i].astype(int)]) for i , likelihood in enumerate(likelihoods)]

        log_prob = np.sum(row_by_classes_log , axis=0) + log_prior
        prob = np.exp(log_prob)
        posteriors = prob / np.sum(prob , axis=1)[: , None]

        return posteriors
    
    # TODO Task 4: Compute accuracy, F1 score, precision, and recall
def compute_metrics( y_test, y_pred):
        """
        Compute accuracy, F1 score, precision, and recall.

        parameters
        ----------
        y_test : np.ndarray, shape (n_samples,)
            True class labels.
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels.

        Returns
        -------
        accuracy : float
            Accuracy score.
        f1 : float
            F1 score.
        precision : float
            Precision score.
        recall : float
            Recall score.
        """
        # YOUR CODE HERE
        true_positive = np.sum((y_pred == 1) & (y_test == 1))
        true_negative = np.sum((y_pred == 0) & (y_test == 0))
        false_positive = np.sum((y_pred == 0) & (y_test == 1))
        false_negative = np.sum((y_pred == 1) & (y_test == 0))

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)

        return accuracy, f1, recall , precision  #swap preciion with recall
    


if __name__ == '__main__':
    from sklearn.naive_bayes import CategoricalNB
    # Load data
    X_train = pd.read_csv('./Lab2/X_train.csv')
    X_test = pd.read_csv('./Lab2/X_test.csv')
    y_train = pd.read_csv('./Lab2/y_train.csv')
    y_test = pd.read_csv('./Lab2/y_test.csv')

    print(f"X_train columns: {[_ for _ in X_train.columns]}")
    print(f"y_train columns: {[_ for _ in y_train.columns]}")
    print(f"X_train Rows: {X_train.shape[0]} Columns: {X_train.shape[1]}")
    print(f"X_test Rows: {X_test.shape[0]} Columns: {X_test.shape[1]}")
    print(f"y_train Rows: {y_train.shape[0]} Columns: {y_train.shape[1]}")
    print(f"y_test Rows: {y_test.shape[0]} Columns: {y_test.shape[1]}")

    X_train = np.array(X_train, dtype=np.int16)
    X_test = np.array(X_test, dtype=np.int16)
    y_train = np.array(y_train, dtype=np.int16).flatten()
    y_test = np.array(y_test, dtype=np.int16).flatten()

    #sklearn
    classifier = CategoricalNB(alpha=1.0e-10)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    result = "Pass" if y_pred[0] == 1 else "Fail"
    print(f"Student with Hours Studied {X_test[0][0]}, \n"
         f"Sleep Hours {X_test[0][1]}, \n"
         f"Interest Level {X_test[0][2]}, \n"
         f"Attendance {X_test[0][3]}, \n"
         f"Social Media Usage {X_test[0][4]}, \n"
         f"and Extracurricular Activities {X_test[0][5]} is predicted to have a grade of {result}")



    #DEBUG
    our_classifier = OurImplementedNaiveBayesCategorical()
    our_classifier.fit(X_train, y_train)
    our_y_pred = our_classifier.predict(X_test)

    # Task 1: Compute prior probabilities
    print("========Task 1: Compute class probabilites========")
    priors = our_classifier.compute_priors(y_train)
    print(f"Priors: {priors}\n")
    """Priors: [0.20355219 0.79644781]"""

    # Task 2: Compute likelihoods
    print("========Task 2: Compute class probabilites========")
    likelihoods = our_classifier.compute_likelihoods(X_train, y_train)
    print(f"Likelihoods: {likelihoods[0][0]}\n")
    """Likelihoods: [9.80392157e-14 2.50563768e-14]"""


    # Task 3: Compute class probabilities
    print("========Task 3: Compute class probabilites========")
    posteriors = our_classifier.compute_posteriors(priors, likelihoods, X_test)
    print(f"Posterior Probabilities: {posteriors[:5]}\n")
    """Posterior Probabilities: [[0.00502236 0.99497764]
    [0.00798569 0.99201431]
    [0.58423732 0.41576268]
    [0.50463072 0.49536928]
    [0.42336149 0.57663851]]"""

    # Task 4: Compute metrics
    print("========Task 4: Compute metrics========")
    accuracy, f1, precision, recall = compute_metrics(y_test, our_y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}\n")
    """ Accuracy: 0.84
        F1 Score: 0.90
        Precision: 0.88
        Recall: 0.92 """

    # Compare with sklearn's implementation
    print(f"Our Implementation and Sklearn's Implementation are equivalent: {np.all(y_pred == our_y_pred)}")
    """Our Implementation and Sklearn's Implementation are equivalent: True"""




