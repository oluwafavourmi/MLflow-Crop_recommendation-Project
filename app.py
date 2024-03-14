import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Crop_recommendation.csv')

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(['rice', 'maize', 'jute', 'cotton', 'coconut', 'papaya', 'orange', 'apple', 'muskmelon', 'watermelon', 'grapes', 'mango', 'banana', 'pomegranate', 'lentil', 'blackgram', 'mungbean', 'mothbeans', 'pigeonpeas', 'kidneybeans', 'chickpea', 'coffee'])
df['label'] = label_encoder.transform(df['label'])

import os
import warnings
import sys
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    return accuracy

if __name__ == '__main__':
    x = df.drop('label', axis='columns')
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    
    C = 4.5
    degree = 2
    max_iter = -1
    
    with mlflow.start_run():
        svm = SVC(C=C, degree=degree, max_iter=max_iter)
        svm.fit(x_train, y_train)
    
        predicted_values = svm.predict(x_test)
        (accuracy) = eval_metrics(y_test, predicted_values)
    
        print('Support Vector Machine(C={:f}, degree={:f}, max_iter={:f}):'.format(C, degree, max_iter))
        print('accuracy:',accuracy )
    
        mlflow.log_param('C', C)
        mlflow.log_param('degree',degree)
        mlflow.log_param('max_iter', max_iter)
        mlflow.log_metric('accuracy', accuracy)
        
        predictions = svm.predict(x_test)
        signature = infer_signature(x_train, svm.predict(x_test))
        
        # For remote server 
        remote_server_uri = "https://dagshub.com/oluwafavourmi/MLflow-Crop_recommendation-Project.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != 'file':
            
            mlflow.sklearn.log_model(
                svm, 'model', registered_model_name='CropRecommmendationSVM', signature=signature
            )
        else:
            mlflow.sklearn.log_model(
                svm, 'model', signature=signature
            )