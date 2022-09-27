class DataCleaningService:

    def __init__(self):
        pass

    @staticmethod
    def clean_data(raw_data):

        print(raw_data.isna().sum())

        clean_data = raw_data.drop('Cabin', axis=1)

        print(clean_data.columns)

        print(clean_data['Age'].median())

        print(clean_data[clean_data['Age'].isna()]['Age'])

        print(clean_data.iloc[[5, 17, 19]]['Age'])
        # clean_data.loc[clean_data['Age'].isna(), 'Age'] = clean_data['Age'].median()
        # same as
        clean_data['Age'] = clean_data['Age'].fillna(clean_data['Age'].median())
        print(clean_data.iloc[[5, 17, 19]]['Age'])

        print(clean_data['Embarked'].value_counts())
        clean_data['Embarked'] = clean_data['Embarked'].fillna('U')
        print(clean_data['Embarked'].value_counts())

        return clean_data
