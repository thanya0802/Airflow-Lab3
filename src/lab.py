import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os

# Load data
def load_data():
    df = pd.read_csv(os.path.join("data", "file.csv"))
    print(f"Loaded {df.shape[0]} rows.")
    return pickle.dumps(df)

# Preprocess
def data_preprocessing(data):
    df = pickle.loads(data)
    num_cols = df.select_dtypes(include=['float64','int64']).columns
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[num_cols])
    print(f" Scaled {len(num_cols)} features.")
    return pickle.dumps(scaled)

# Train + save model
def build_save_model(data, filename):
    X = pickle.loads(data)
    sse = []
    for k in range(1,11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        sse.append(km.inertia_)
    with open(filename, "wb") as f:
        pickle.dump(km, f)
    print(f" Model saved to {filename}")
    return pickle.dumps(sse)

# Find elbow
def load_model_elbow(filename, sse):
    with open(filename,"rb") as f:
        model = pickle.load(f)
    sse_vals = pickle.loads(sse)
    kl = KneeLocator(range(1,11), sse_vals, curve="convex", direction="decreasing")
    print(f" Optimal k = {kl.elbow}")
    return f"Optimal clusters: {kl.elbow}"
