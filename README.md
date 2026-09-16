# Anti-Money Laundering (AML) Detection via Graph Neural Networks

## Overview
This project implements an end-to-end Graph Machine Learning pipeline to detect illicit transactions and money laundering activities within cryptocurrency (Bitcoin) networks. 

Unlike traditional tabular machine learning models that evaluate transactions in isolation, this system leverages Graph Representation Learning to capture the complex, relational topology of financial networks. The core predictive model uses GraphSAGE (Sample and Aggregate) to generate node embeddings based on transaction features and neighborhood structures, which is then served as a real-time REST API microservice.

## Architecture & Technical Stack
The system is built with a focus on modularity and production-readiness:

* **Data Processing & Graph Construction:** `pandas`, `NumPy`, `NetworkX`
* **Deep Learning Framework:** `PyTorch`, `PyTorch Geometric (PyG)`
* **Model Topology:** Inductive Representation Learning (GraphSAGE)
* **Serving & API:** `FastAPI`, `Uvicorn`, `Pydantic`
* **Containerization:** `Docker`

## Installation & Deployment
The application is fully containerized to eliminate environment dependencies. To deploy the service locally:

```bash
# 1. Clone the repository
git clone https://github.com/RaufEksi/gnn-aml-detection.git
cd gnn-aml-detection

# 2. Build the Docker image
docker build -t gnn-fraud-api .

# 3. Run the container
docker run -d -p 8000:8000 --name aml-api gnn-fraud-api
```

## API Usage
Once the container is running, the interactive API documentation (Swagger UI) is available at http://localhost:8000/docs.

The API expects a subgraph of a transaction environment to make inductive predictions.

**Endpoint:** `POST /predict`
**Payload Example:**

```json
{
  "x": [[...], [...], [...]],
  "edge_index": [[0, 1], [0, 2]],
  "target_node_idx": 0
}
```

## Model Performance & Methodology
The dataset presents a highly imbalanced class distribution (illicit transactions constitute a very small minority). To mitigate this, the loss function (CrossEntropyLoss) is strictly weighted.

Validation Strategy: Temporal split (preventing data leakage from future transactions).

Current Baseline F1-Score: ~0.55

Target Precision: Threshold optimization is applied to maintain Precision > 0.70, minimizing false-positive alerts to reduce operational review costs.