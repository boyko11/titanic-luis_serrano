import pandas as pd


class DataPreprocessingService:

    def __init__(self):
        pass

    @classmethod
    def preprocess_data(cls, dataset_file_cleaned):
        preprocessed_dataset = pd.read_csv(dataset_file_cleaned)

        gender_columns = pd.get_dummies(preprocessed_dataset['Sex'], prefix='Sex')
        print(gender_columns)

        embarked_columns = pd.get_dummies(preprocessed_dataset["Embarked"], prefix="Embarked")
        # embarked_columns = pd.get_dummies(preprocessed_dataset["Pclass"], prefix="Pclass")
        print(embarked_columns)

        preprocessed_dataset = pd.concat([preprocessed_dataset, gender_columns], axis=1)
        preprocessed_dataset = pd.concat([preprocessed_dataset, embarked_columns], axis=1)

        preprocessed_dataset = preprocessed_dataset.drop(['Sex', 'Embarked'], axis=1)

        cls.percentage_Pclass_survived(1, preprocessed_dataset)
        cls.percentage_Pclass_survived(2, preprocessed_dataset)
        cls.percentage_Pclass_survived(3, preprocessed_dataset)

        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        categorized_age = pd.cut(preprocessed_dataset['Age'], bins)
        preprocessed_dataset['Categorized_age'] = categorized_age
        # preprocessed_dataset = preprocessed_dataset.drop(["Age"], axis=1)

        cagegorized_age_columns = pd.get_dummies(preprocessed_dataset['Categorized_age'], prefix='Categorized_age')
        preprocessed_dataset = pd.concat([preprocessed_dataset, cagegorized_age_columns], axis=1)
        preprocessed_dataset = preprocessed_dataset.drop(['Categorized_age'], axis=1)

        # remove no-information columns
        preprocessed_dataset = preprocessed_dataset.drop(['Name', 'Ticket'], axis=1)

        return preprocessed_dataset

    @staticmethod
    def percentage_Pclass_survived(which_Pclass, preprocessed_dataset):

        # return \
        #     len(preprocessed_dataset.query('Pclass == @which_Pclass and Survived == 1')) / \
        #     len(preprocessed_dataset.query('Pclass == @which_Pclass'))

        # OR
        this_class_only = preprocessed_dataset[preprocessed_dataset['Pclass'] == which_Pclass]
        survival_rate = sum(this_class_only['Survived']) / len(this_class_only)
        print(f'Pclass {which_Pclass} survival rate: {survival_rate}')
        return survival_rate