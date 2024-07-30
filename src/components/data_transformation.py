import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation!
        '''
        try:
            numercial_colums=['Application mode','Course',
                'Previous qualification',
                'Previous qualification (grade)',
                'Nacionality',
                "Mother's qualification",
                "Father's qualification",
                "Mother's occupation",
                "Father's occupation",
                'Admission grade',
                'Age at enrollment',
                'Curricular units 1st sem (credited)',
                'Curricular units 1st sem (enrolled)',
                'Curricular units 1st sem (evaluations)',
                'Curricular units 1st sem (approved)',
                'Curricular units 1st sem (grade)',
                'Curricular units 1st sem (without evaluations)',
                'Curricular units 2nd sem (credited)',
                'Curricular units 2nd sem (enrolled)',
                'Curricular units 2nd sem (evaluations)',
                'Curricular units 2nd sem (approved)',
                'Curricular units 2nd sem (grade)',
                'Curricular units 2nd sem (without evaluations)',
                'Unemployment rate',
                'Inflation rate',
                'GDP']
            
            categorical_columns=['Marital status',
                'Application order',
                'Daytime/evening attendance',
                'Displaced',
                'Educational special needs',
                'Debtor',
                'Tuition fees up to date',
                'Gender',
                'Scholarship holder',
                'International',
                ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Categorical Columns encoding completed.")

            logging.info("Categorical columns encoding completed.")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numercial_colums),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train test data read completed!")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj=self.get_data_transformer_object()
            
            
            target_column_name='Target'
        
            
            numercial_colums=['Application mode','Course',
                'Previous qualification',
                'Previous qualification (grade)',
                'Nacionality',
                "Mother's qualification",
                "Father's qualification",
                "Mother's occupation",
                "Father's occupation",
                'Admission grade',
                'Age at enrollment',
                'Curricular units 1st sem (credited)',
                'Curricular units 1st sem (enrolled)',
                'Curricular units 1st sem (evaluations)',
                'Curricular units 1st sem (approved)',
                'Curricular units 1st sem (grade)',
                'Curricular units 1st sem (without evaluations)',
                'Curricular units 2nd sem (credited)',
                'Curricular units 2nd sem (enrolled)',
                'Curricular units 2nd sem (evaluations)',
                'Curricular units 2nd sem (approved)',
                'Curricular units 2nd sem (grade)',
                'Curricular units 2nd sem (without evaluations)',
                'Unemployment rate',
                'Inflation rate',
                'GDP']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_obj(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)
        