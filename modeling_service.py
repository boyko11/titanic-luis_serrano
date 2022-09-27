import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
import time
from tabulate import tabulate


class ModelingService:

    def __init__(self):
        pass

    @classmethod
    def model(cls, dataset_file_preprocessed):

        data = pd.read_csv(dataset_file_preprocessed)
        features = data.drop(['Survived'], axis=1)
        labels = data['Survived']

        features_train, features_validation_test, labels_train, labels_validation_test = \
            train_test_split(features, labels, test_size=.4, random_state=7777)

        features_validate, features_test, labels_validate, labels_test = \
            train_test_split(features_validation_test, labels_validation_test, test_size=.5, random_state=7777)

        print(len(features_train), len(labels_train))
        print(len(features_validate), len(labels_validate))
        print(len(features_test), len(labels_test))

        log_reg_model = cls.train_model("Logistic Regression", LogisticRegression(max_iter=700), features, labels)

        dt_model = cls.train_model("Decision Tree", DecisionTreeClassifier(), features, labels)

        nb_model = cls.train_model("Naive Bayes", GaussianNB(), features, labels)

        svm_model = cls.train_model("SVM", SVC(), features, labels)

        rf_model = cls.train_model("Random Forest", RandomForestClassifier(), features, labels)

        gb_model = cls.train_model("Gradient Booster", GradientBoostingClassifier(), features, labels)

        ada_model = cls.train_model("AdaBoost", AdaBoostClassifier(), features, labels)

        print("Done Training.")

        cls.score_models([log_reg_model, dt_model, nb_model, svm_model, rf_model, gb_model, ada_model],
                         features_validate, labels_validate, features_test, labels_test)

        print("Done Scoring.")

    @staticmethod
    def train_model(this_model_name, this_model_instance, features, data):

        print(f'Training {this_model_name}...')

        start = time.time_ns()

        trained_model = this_model_instance.fit(features, data)

        end = time.time_ns()

        print(f'It took {(end - start) // 1_000_000} milliseconds.')

        return trained_model

    @classmethod
    def score_models(cls, models_list, features_validate, labels_validate, features_test, labels_test):

        tabulate_data = []
        for this_model in models_list:
            model_name = type(this_model).__name__

            validation_accuracy, validation_f1_score = cls.accuracy_f1_score(this_model, features_validate,
                                                                             labels_validate)

            test_accuracy, test_f1_score = cls.accuracy_f1_score(this_model, features_test, labels_test)

            tabulate_data.append([model_name, validation_accuracy, validation_f1_score, test_accuracy, test_f1_score])

        headers = ['Model', 'Validate Accuracy', 'Validate F1-Score', 'Test Accuracy', 'Test F1-Score' ]
        print(tabulate(tabulate_data, headers=headers))
        print('-------------------------------------')

    @staticmethod
    def accuracy_f1_score(this_model, features, labels):

        accuracy = round(this_model.score(features, labels), 3)
        predicted = this_model.predict(features)
        model_f1_score = round(f1_score(labels, predicted), 3)

        return accuracy, model_f1_score


