from data_preparation.data_prep import Dataset
from model_training.model_train import ModelTraining


def main():
    """
    Main function to prepare the dataset, train the model, and evaluate the results.
    """
    dataset = Dataset('ObesityDataSet.csv')
    cols_to_remove = []
    cols_to_normalize = []
    cols_to_transform = {
        'Gender': {'Female': 0, 'Male': 1},
        'family_history_with_overweight': {'no': 0, 'yes': 1},
        'FAVC': {'no': 0, 'yes': 1},
        'SMOKE': {'no': 0, 'yes': 1},
        'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'SCC': {'no': 0, 'yes': 1},
        'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    }
    dataset.transform_text_values(cols_to_transform)
    dataset.one_hot_encode('MTRANS')
    dataset.remove_columns(cols_to_remove)
    dataset.normalize(cols_to_normalize)
    dataset.split_data()

    model_training = ModelTraining(train_set=dataset.train_set,
                                   test_set=dataset.test_set,
                                   target_column='NObeyesdad')
    model_training.train_model()
    model_training.predict()
    accuracy, conf_matrix, class_report = model_training.evaluate_model()

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

if __name__ == "__main__":
    main()