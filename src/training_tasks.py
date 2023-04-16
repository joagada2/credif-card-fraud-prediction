import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from patsy import dmatrices
from sklearn.model_selection import train_test_split
import joblib
import mlflow
from helper import BaseLogger
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score, make_scorer, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

@hydra.main(version_base=None, config_path="../config", config_name="training_config")
def extract_data(config: DictConfig):
    raw_path = abspath(config.training_config.raw.path)
    data = pd.read_csv(raw_path)
    return data

def validate_data(config: DictConfig):
    raw_path = abspath(config.training_config.raw.path)
    data = extract_data(raw_path)
    print(data.head())
    print(data.info())

def process_data(config: DictConfig):
    raw_path = abspath(config.training_config.raw.path)
    processed_path = abspath(config.trainin_data.processed.path)
    data = extract_data(raw_path)
    # there are no processing required so data is saved to processed folder
    data.to_csv(processed_path,index=False)
    # read data from processed folder
    data = pd.read_csv(processed_path)
    # Separate the dataset as response/target variable and feature variables
    X = data.drop("fraud", axis=1)
    y = data["fraud"]
    # Train and test splitting of data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    print(f"x train shape: {X_train.shape}")
    print(f"x train head: {X_train[:10]}")
    print(f"x test shape: {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y train head: {y_train[:10]}")
    print(f"y test shape: {y_test.shape}")
    #save final data
    X_train.to_csv(abspath(config.final.X_train.path), index=False)
    X_test.to_csv(abspath(config.final.X_test.path), index=False)
    y_train.to_csv(abspath(config.final.y_train.path), index=False)
    y_test.to_csv(abspath(config.final.y_test.path), index=False)

def train_model(config: DictConfig):
    """Function to train the model"""
    def load_data(path: DictConfig):
        X_train = pd.read_csv(abspath(path.X_train.path))
        X_test = pd.read_csv(abspath(path.X_test.path))
        y_train = pd.read_csv(abspath(path.y_train.path))
        y_test = pd.read_csv(abspath(path.y_test.path))
        return X_train, X_test, y_train, y_test

    def get_objective(
    		X_train: pd.DataFrame,
    		y_train: pd.DataFrame,
    		X_test: pd.DataFrame,
    		y_test: pd.DataFrame,
    		config: DictConfig,
    		space: dict,
		):

        model = XGBClassifier(
            use_label_encoder=config.model.use_label_encoder,
            objective=config.model.objective,
        	n_estimators=space["n_estimators"],
        	max_depth=int(space["max_depth"]),
        	gamma=space["gamma"],
        	reg_alpha=int(space["reg_alpha"]),
        	min_child_weight=int(space["min_child_weight"]),
        	colsample_bytree=int(space["colsample_bytree"]),
    		)

        evaluation = [(X_train, y_train), (X_test, y_test)]

        model.fit(
        	X_train,
        	y_train,
        	eval_set=evaluation,
        	eval_metric=config.model.eval_metric,
        	early_stopping_rounds=config.model.early_stopping_rounds,
    		)
        prediction = model.predict(X_test.values)
        accuracy = accuracy_score(y_test, prediction)
        print("SCORE:", accuracy)
        return {"loss": -accuracy, "status": STATUS_OK, "model": model}

    def optimize(objective: Callable, space: dict):
        trials = Trials()
        best_hyperparams = fmin(
        	fn=objective,
        	space=space,
        	algo=tpe.suggest,
        	max_evals=100,
        	trials=trials,
    		)
        print("The best hyperparameters are : ", "\n")
        print(best_hyperparams)
        best_model = trials.results[
        		np.argmin([r["loss"] for r in trials.results])
    		]["model"]
        return best_model

    X_train, X_test, y_train, y_test = load_data(config.processed)

    # Define space
    space = {
        "max_depth": hp.quniform("max_depth", **config.model.max_depth),
        "gamma": hp.uniform("gamma", **config.model.gamma),
        "reg_alpha": hp.quniform("reg_alpha", **config.model.reg_alpha),
        "reg_lambda": hp.uniform("reg_lambda", **config.model.reg_lambda),
        "colsample_bytree": hp.uniform(
        "colsample_bytree", **config.model.colsample_bytree
        ),
        	"min_child_weight": hp.quniform(
            	"min_child_weight", **config.model.min_child_weight
        	),
        	"n_estimators": config.model.n_estimators,
        	"seed": config.model.seed,
    	}
    objective = partial(
        	get_objective, X_train, y_train, X_test, y_test, config
    	)

    # Find best model
    best_model = optimize(objective, space)

    # Save model
    joblib.dump(best_model, abspath(config.model.path))

def evaluate_model(config: DictConfig):
    def load_data(path: DictConfig):
        X_test = pd.read_csv(abspath(path.X_test.path))
        y_test = pd.read_csv(abspath(path.y_test.path))
        return X_test, y_test

    def load_model(model_path: str):
        return joblib.load(model_path)

    def predict(model: XGBClassifier, X_test: pd.DataFrame):
        return model.predict(X_test)

    # function to log parameters to dagshub and mlflow
    def log_params(model: XGBClassifier):
        logger.log_params({"model_class": type(model).__name__})
        model_params = model.get_params()

        for arg, value in model_params.items():
            logger.log_params({arg: value})

    logger.log_params({"features": features})

    def log_metrics(**metrics: dict):
        logger.log_metrics(metrics)

    os.environ['MLFLOW_TRACKING_URI'] = config.mlflow_tracking_ui
    os.environ['MLFLOW_TRACKING_USERNAME'] = config.mlflow_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = config.mlflow_PASSWORD

    with mlflow.start_run():
        # Load data and model
        X_test, y_test = load_data(config.final)
        model = load_model(abspath(config.model.path))
        # Get predictions
        prediction = predict(model, X_test)
        # Get metrics
        f1 = f1_score(y_test, prediction)
        print(f"F1 Score of this model is {f1}.")
        # get metrics
        accuracy = balanced_accuracy_score(y_test, prediction)
        print(f"Accuracy Score of this model is {accuracy}.")
        area_under_roc = roc_auc_score(y_test, prediction)
        print(f"Area Under ROC is {area_under_roc}.")
        precision = precision_score(y_test, prediction)
        print(f"Precision of this model is {precision}.")
        recall = recall_score(y_test, prediction)
        print(f"Recall for this model is {recall}.")
        # log metrics to remote server (dagshub)
        log_params(model)
        log_metrics(f1_score=f1, accuracy_score=accuracy, area_Under_ROC=area_under_roc, precision=precision,
                    recall=recall)
