# GNN Implementation & MLOps Pipeline

This repository implements a Graph Convolutional Network (GCN) from scratch using PyTorch tensor operations and extends it into an end-to-end MLOps pipeline for experiment tracking, model validation, deployment, and monitoring.

## Task 1: GCN From Scratch

Implements and verifies a two-layer GCN without relying on external GNN libraries. The experiment uses a synthetic graph of 1,000 nodes, each with three features, with ring connections and shortcut edges. The dataset is split into 60% training, 20% validation, and 20% testing.

The implementation covers graph generation, GCN and MLP models, full-batch training, validation-based checkpointing, and multi-seed evaluation with edge-removal experiments.

**Main files:** `generate_data.py`, `model.py`, `train.py`, `checks.py`, and `run.py` (all under `task1/`).

Run the complete experiment:

```bash
python task1/run.py
```

The verification suite checks manual layer calculations, permutation equivariance, and the identity between the GCN and MLP when the graph contains no edges. All three numerical checks target a maximum difference below \(10^{-6}\).

The experiments also demonstrate the impact of graph structure: the GCN achieves **0.743 test accuracy** on the original graph, compared with **0.589** on a random graph and **0.686** for the MLP baseline. Progressive edge removal evaluates how performance changes as graph connectivity decreases.

## Task 4: MLOps Pipeline for GCN

Extends the GCN into a reproducible and deployable pipeline with experiment tracking, model versioning, validation gates, drift monitoring, rollback, and a FastAPI inference service.

The pipeline uses an inductive split with 800 training nodes and 200 held-out nodes. It records three model versions: a baseline (`v1.0.0`), a tuned model (`v1.1.0`), and a deliberately degraded model (`v1.2.0-bad`). The validation gate promotes approved models and blocks regressions.

It also tests common serving failures, including isolated nodes, duplicate target IDs, invalid edge endpoints, incorrect feature dimensions, unknown target IDs, and model-loading failures. Additional checks compare offline predictions with API responses, demonstrate inference on existing and unseen nodes, monitor graph and embedding drift, and verify rollback to a previously approved model.

**Main files:** `data.py`, `registry.py`, `drift.py`, `checks.py`, `app.py`, `run.py`, and `Dockerfile` (all under `task4/`).

Run the full demonstration:

```bash
python task4/run.py
```

## Running the API Locally

Start the FastAPI service with:

```bash
uvicorn task4.app:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "0": [0.1, -0.5, 0.4],
      "1": [1.2, 0.8, -0.2]
    },
    "edges": [[0, 1]],
    "target_ids": [0, 1]
  }'
```

## Running with Docker

Build and start the inference service:

```bash
docker build -f task4/Dockerfile -t gcn-serving .
docker run -p 8000:8000 gcn-serving
```

The API will then be accessible at [http://localhost:8000/docs](http://localhost:8000/docs).