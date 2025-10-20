import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os

# 1️⃣ Load the dataset
def load_data():
    """
    Loads training data (file.csv) and serializes it for Airflow XCom.
    """
    df = pd.read_csv(os.path.join("data", "file.csv"))
    print(f"✅ Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
    data = pickle.dumps(df)
    return data


# 2️⃣ Preprocess the data
def data_preprocessing(data):
    """
    Scales numeric features for fair clustering.
    """
    df = pickle.loads(data)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    print(f"✅ Preprocessed {len(numeric_cols)} numeric features")
    return pickle.dumps(scaled_data)


# 3️⃣ Build and save the K-Means model
def build_save_model(data, filename):
    """
    Trains K-Means for k=1..10, saves the model, returns serialized SSE list.
    """
    X = pickle.loads(data)
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    with open(filename, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"✅ Model saved to {filename}")
    return pickle.dumps(sse)


# 4️⃣ Load model and find optimal cluster count
def load_model_elbow(filename, sse):
    """
    Finds the optimal number of clusters using the Elbow Method.
    """
    with open(filename, "rb") as f:
        model = pickle.load(f)
    sse_vals = pickle.loads(sse)
    kl = KneeLocator(range(1, 11), sse_vals, curve="convex", direction="decreasing")
    result = f"Optimal number of clusters (farmer segments): {kl.elbow}"
    print("✅", result)
    return result


# 5️⃣ Optional — Predict clusters for new farms (test.csv)
def predict_clusters(model_path="working_data/clustering_model.pkl", test_path="data/test.csv"):
    """
    Loads model and assigns clusters to new farms in test.csv.
    """
    df = pd.read_csv(test_path)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    clusters = model.predict(X_scaled)
    df["Cluster"] = clusters
    print("✅ Cluster assignments for test data:")
    print(df)
    df.to_csv("working_data/test_clusters.csv", index=False)
    return "✅ Saved cluster-labeled test data."

