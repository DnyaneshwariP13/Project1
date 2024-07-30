import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
#from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data.")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest Classifier" : RandomForestClassifier(),
                "AdaBoost Classifier" : AdaBoostClassifier(),
                "Gradient Boosting Classifier" : GradientBoostingClassifier(),
                "XGBoost Classifier" : XGBClassifier(),
                "LightGBM Classifier" : LGBMClassifier(verbose=0)

            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
                
            #to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            #to get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found!")
            
            logging.info("Best model found on both training and testing dataset.")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)

            return r2_square
        


        except Exception as e:
            raise CustomException(e,sys)
