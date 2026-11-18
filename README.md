# Conveyor Belt Jamming Prediction Using MLOps

## Objective
Predict whether a conveyor belt is likely to jam during its next operating shift using sensor readings and maintenance records.

## Dataset
The supplied CSV contains 500 records across versions: V1, V2_Festival. V2 represents the higher-throughput/stress scenario as defined in the assignment dataset.

## Pipeline
1. Validate and preprocess sensor records.
2. Engineer `motor_load_ratio` and `belt_wear_pressure_index`.
3. Save reusable features to `data/feature_store.csv`.
4. Train Logistic Regression and Random Forest.
5. Compare accuracy, recall, F1 and confusion matrices.
6. Save the model selected by recall then F1.
7. Serve predictions through FastAPI.
8. Run automated API tests through GitHub Actions.

## Run locally
```bash
python -m pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000/docs` for the API interface.

## Test
```bash
pytest -v
```

## API
- `GET /health`
- `POST /predict`

## Model result
Selected model in this packaged run: **Logistic Regression**. See `reports/model_metrics.json` for evaluation metrics.

## MLflow
MLflow is listed as a project dependency. To track a run, add MLflow logging/registry execution after model training in your environment.

## Important limitation
This is an educational prediction prototype. Synthetic/assignment data and model scores are not evidence of real-world industrial safety performance. Validate with real operational data and engineering review before operational use.
