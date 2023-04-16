from datetime import timedelta
import os
from textwrap import dedent

import yaml

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

import prediction_tasks

with open(
    os.path.join(os.getenv("CONFIG_FOLDER"), "prediction_config.yml")
) as f:
    try:
        prediction_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["joe88data@gmail.com.com"],
    "email_on_failure": True,
    "email_on_retry": Ture,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "prediction",
    default_args=default_args,
    description="Prediction DAG",
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["prediction"],
) as dag:

    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    for task in prediction_config["pred_set_names"]:
        prediction_config["input"] = prediction_config["input_file"].format(
            task
        )
        prediction_config["output"] = prediction_config["output_file"].format(
            task
        )

        get_input = PythonOperator(
            task_id="get_input_" + task,
            python_callable=prediction_tasks.get_input,
            op_kwargs=prediction_config,
        )
        predict_fraud = PythonOperator(
            task_id="predict_fraud_" + task,
            python_callable=prediction_tasks.predict_fraud,
            op_kwargs=prediction_config,
        )
        save_prediction = PythonOperator(
            task_id="save_prediction_" + task,
            python_callable=prediction_tasks.save_prediction,
            op_kwargs=prediction_config,
        )
        get_input.doc_md = dedent(
            """\
        #### Task Documentation
        This task fetches input data for prediction
        """
        )

        predict_fraud.doc_md = dedent(
            """
        #### Task Documentation
        This task loads model and predicts on the given input data
        """
        )

        save_prediction.doc_md = dedent(
            """
        #### Task Documentation
        Save prediction to output folder
        """
        )

        dag.doc_md = __doc__
        dag.doc_md = """
            Prediction DAG
            """

        start >> get_input >> predict_fraud >> save_prediction >> end
