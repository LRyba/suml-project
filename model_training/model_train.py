from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ModelTraining:
    """
    A class used to train and evaluate a Random Forest model.
    """
    
    def __init__(self, train_set, test_set, target_column, random_state=42):
        """
        Initializes the ModelTraining class with training and test datasets and target column.

        train_set : The training dataset.
        test_set : The test dataset.
        target_column : The name of the target column.
        random_state : The random state for the Random Forest classifier (default is 42).
        """
        self.clf = RandomForestClassifier(random_state=random_state)
        self.train_set = train_set
        self.test_set = test_set
        self.target_column = target_column
        self.X_train = self.train_set.drop(self.target_column, axis=1)
        self.Y_train = self.train_set[self.target_column]
        self.X_test = self.test_set.drop(self.target_column, axis=1)
        self.Y_test = self.test_set[self.target_column]
        self.Y_pred = None

    def train_model(self):
        """
        Trains the Random Forest model using the training data.
        """
        self.clf.fit(self.X_train, self.Y_train)

    def predict(self):
        """
        Makes predictions using the trained model on the test data.
        """
        self.Y_pred = self.clf.predict(self.X_test)

    def evaluate_model(self):
        """
        Evaluates the model performance.

        Returns
        accuracy : The accuracy of the model.
        conf_matrix : The confusion matrix of the model predictions.
        class_report : The classification report of the model predictions.
        """
        accuracy = accuracy_score(self.Y_test, self.Y_pred)
        conf_matrix = confusion_matrix(self.Y_test, self.Y_pred)
        class_report = classification_report(self.Y_test, self.Y_pred)
        return accuracy, conf_matrix, class_report