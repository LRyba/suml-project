from pickle import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def train_logistic_regression(X_train, y_train, max_iter=50000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=10, random_state=101):
  forest = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
  forest.fit(X_train, y_train)
  print(f"Las: {forest.score(X_train, y_train)}")
  return forest

def train_decision_tree(X_train, y_train):
  tree = DecisionTreeClassifier()
  tree.fit(X_train, y_train)
  print(f"Drzewa decyzyjne: {tree.score(X_train, y_train)}")
  return tree

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

def export_model(model, file_path):
    dump(model, open(file_path, "wb"))