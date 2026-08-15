import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    @staticmethod
    def train(X, y, model_type="random_forest", test_size=0.2, random_state=42):
        """
        Train a classifier and return model, accuracy, classification report.
        model_type: 'random_forest' or 'logistic'
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        elif model_type == "logistic":
            model = LogisticRegression(random_state=random_state, max_iter=1000)
        else:
            raise ValueError("model_type must be 'random_forest' or 'logistic'")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return model, acc, report
