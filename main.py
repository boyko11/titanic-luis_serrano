import pandas as pd
from data_cleaning_service import DataCleaningService
from data_preprocessing_service import DataPreprocessingService
from modeling_service import ModelingService

if __name__ == '__main__':

    dataset_file = 'data/titanic.csv'
    dataset_file_cleaned = 'data/clean_titanic_data.csv'
    dataset_file_preprocessed = 'data/preprocessed_titanic_data.csv'

    raw_data = pd.read_csv(dataset_file, index_col='PassengerId')

    cleaned_data = DataCleaningService.clean_data(raw_data)
    cleaned_data.to_csv(dataset_file_cleaned, index=None)

    preprocessed_data = DataPreprocessingService.preprocess_data(dataset_file_cleaned)
    preprocessed_data.to_csv(dataset_file_preprocessed, index=None)

    ModelingService.model(dataset_file_preprocessed)

    print('Done')