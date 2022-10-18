## Dynamic risk assessment system

This project sets up an automated ML training, monitoring and re-deployment process. The goal is to predict which customers are in the risk of exiting their contracts from a toy dataset.

Run individual scripts with, for instance:

```python
python dynamic_risk_assessment/ingestion.py
```

for data ingestion, 

```python
python dynamic_risk_assessment/training.py
```

To start serving the main app, use:
```python
python app.py
```

Once the app is running, evaluate the endpoints with:
```python
python dynamic_risk_assessment/api_calls.py
```

You can also run the full pipeline with:
```python
python main.py
```

All model details are contained in the `conf/config.json` file.