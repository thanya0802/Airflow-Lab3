# Farmer Segmentation using Apache Airflow

## Overview

This project demonstrates how to automate a **Machine Learning workflow** using **Apache Airflow** within a **Dockerized environment**.  
The workflow segments farmers into meaningful groups based on purchasing and credit data using **K-Means Clustering**.  
The lab showcases **MLOps practices** such as workflow orchestration, task dependency management, and automated model execution.

---

## Objective

To build, orchestrate, and execute an **end-to-end ML pipeline** using Apache Airflow that performs:

1. **Data Loading**
2. **Data Preprocessing**
3. **Model Training and Saving**
4. **Model Loading and Testing**

---

## Architecture

**Tools & Technologies Used:**

-  **Docker** — containerized Airflow environment (Postgres + Redis + Scheduler + Webserver)
-  **Apache Airflow 2.5.1** — pipeline orchestration and task management
-  **Python 3.13 + scikit-learn + pandas + kneed** — ML modeling and preprocessing
-  **PostgreSQL** — metadata database for Airflow
-  **Redis** — Celery message broker for distributed task execution

---

##  Workflow

The DAG (`farmer_segmentation_dag`) automates the following sequence:

| Task | Description |
|------|-------------|
| **`load_data_task`** | Reads the raw dataset (`file.csv`) from `/data/`. |
| **`data_preprocessing_task`** | Cleans and scales numerical features. |
| **`build_save_model_task`** | Trains a K-Means model and saves it as `/working_data/clustering_model.pkl`. |
| **`load_model_task`** | Loads the saved model and applies it to unseen farmer data from `test.csv`. |

All tasks are connected sequentially and executed inside Airflow's scheduler.

---

##  Folder Structure

```
Airflow-Lab3/
├── dags/
│   └── airflow_dag.py
├── src/
│   └── lab.py
├── data/
│   ├── file.csv
│   └── test.csv
├── working_data/
│   └── clustering_model.pkl
├── docker-compose.yaml
└── README.md
```

---

##  How to Run

### Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM allocated to Docker

### Steps

1. **Start Docker Desktop**

2. **Build and start Airflow:**
   ```bash
   docker compose up --build
   ```

3. **Open the Airflow UI:**  
   Navigate to [http://localhost:8080](http://localhost:8080)  
   - **Username:** `airflow`  
   - **Password:** `airflow`

4. **Enable and trigger the DAG:**  
   - Locate `farmer_segmentation_dag` in the DAGs list
   - Toggle it ON
   - Click the "Play" button to trigger manually

5. **Monitor task execution:**  
   - View progress in **Graph View** or **Grid View**
   - Check logs for each task if needed

6. **Verify outputs:**  
   The trained model will be saved at:
   ```
   Airflow-Lab3/working_data/clustering_model.pkl
   ```

---

## Results

-  All DAG tasks completed successfully
-  The trained K-Means clustering model was saved as `clustering_model.pkl`
-  The pipeline is fully reproducible and automated using Airflow
-  Demonstrates proper task dependency management and orchestration

---

This project demonstrates:

- **Workflow Orchestration:** Using Airflow to manage complex ML pipelines
- **Containerization:** Running reproducible ML workflows in Docker
- **MLOps Best Practices:** Separating data, code, and models
- **Task Dependencies:** Creating sequential and parallel task execution
- **Model Persistence:** Saving and loading trained models for inference

---


