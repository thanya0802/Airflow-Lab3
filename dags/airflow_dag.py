# airflow_dag.py — Farmer Segmentation with K-Means
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow import configuration as conf
import sys, os
sys.path.append('/opt/airflow/src')
from lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# Enable pickling so Airflow can pass complex objects (like serialized DataFrames)
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments for the DAG
default_args = {
    'owner': 'thanya',
    'depends_on_past': False,
    'start_date': datetime(2025, 10, 20),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'farmer_segmentation_dag',
    default_args=default_args,
    description='Farmer Segmentation using K-Means (Crop Yield Optimization)',
    schedule_interval=None,
    catchup=False,
)

# Define tasks
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, '/opt/airflow/working_data/clustering_model.pkl'],
    dag=dag,
)

load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=['/opt/airflow/working_data/clustering_model.pkl', build_save_model_task.output],
    dag=dag,
)

# Define task dependencies (pipeline order)
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.cli()

  
