from datetime import timedelta
import os
from textwrap import dedent
import yaml
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import training_tasks

with open(
    os.path.join(os.getenv("CONFIG_FOLDER"), "training_config.yml")
) as f:
    try:
        training_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["joe88data@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}
with DAG(
    "training",
    default_args=default_args,
    description="Training DAG",
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["training"],
) as dag:

    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=training_tasks.data_extraction,
        op_kwargs=training_config,
    )
    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=training_tasks.data_validation,
        op_kwargs=training_config,
    )
    process_data = PythonOperator(
        task_id="process_data",
        python_callable=training_tasks.data_preparation,
        op_kwargs=training_config,
    )
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=training_tasks.model_training,
        op_kwargs=training_config,
    )
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=training_tasks.model_evaluation,
        op_kwargs=training_config,
    )

    extract_data.doc_md = dedent(
        """\
    #### Task Documentation
    This task copies data from source folder to intermedia folder
    """
    )

    validate_data.doc_md = dedent(
        """\
    #### Task Documentation
    This task prints some rows from the input data
    """
    )

    process_data.doc_md = dedent(
        """\
    #### Task Documentation
    This task splits the data into train and test and save them as parquet
    files
    """
    )

    train_model.doc_md = dedent(
        """\
    #### Task Documentation
    This task contains model training and hyperparameter tuning using hyperopt
    """
    )

    evaluate_model.doc_md = dedent(
        """
    #### Task Documentation
    This task validates the model and logs into mlflow
    """
    )

    dag.doc_md = __doc__
    dag.doc_md = """
    Training DAG
    """  # otherwise, type it like this

    (
        extract_data
        >> validate_data
        >> process_data
        >> train_model
        >> evaluate_model
    )
