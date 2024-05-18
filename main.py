from data_prep import Dataset


# prepare dataset
dataset = Dataset('ObesityDataSet.csv')

cols_to_remove = []
cols_to_normalize = []
cols_to_transform = {
    'Gender': {'Female': 0,'Male': 1},
    'family_history_with_overweight': {'no': 0, 'yes': 1},
    'FAVC': {'no': 0, 'yes': 1},
    'SMOKE': {'no': 0, 'yes': 1},
    'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3 },
    'SCC': {'no': 0, 'yes': 1},
    'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3 },
}

# apply the transformation
dataset.transform_text_values(cols_to_transform)

# one-hot encode MTRANS column
dataset.one_hot_encode('MTRANS')

dataset.remove_columns(cols_to_remove)
dataset.normalize(cols_to_normalize)
dataset.split_data()

from model_train import ModelTraining

# train model
model_training = ModelTraining(train_set=dataset.train_set,
                               test_set=dataset.test_set,
                               target_column='NObeyesdad')
model_training.train_model()
model_training.predict()
accuracy, conf_matrix, class_report = model_training.evaluate_model()

# Print the evaluation results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
