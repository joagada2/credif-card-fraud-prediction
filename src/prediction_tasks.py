import os
import pathlib
import shutil
import mlflow
import pandas as pd
import numpy as np

@hydra.main(version_base=None, config_path="../config", config_name="prediction_config")
def get_input(config: DictConfig):
    input_path = abspath(config.prediction_config.input_file.path)
    data = pd.read_csv(input_path)
    return data

def predict_fraud(config: DictConfig):
    input_path = abspath(config.prediction_config.input_file.path)
    data = get_input(input_path)
    print(f"Input data: {data}")
    model = joblib.load(abspath(config.model.path))
    prediction = model.predict(data)
    print(f"Predict result type: {type(prediction)}")
    print(f"Predict result: {prediction}")
    data['predictions'] = predictions.tolist()
    return data

def save_prediction(config: DictConfig):
    input_path = abspath(config.prediction_config.input_file.path)
    prediction_df = predict_fraud(input_path)
    prediction_df.to_csv(abspath(config.prediction_config.output_file.path))