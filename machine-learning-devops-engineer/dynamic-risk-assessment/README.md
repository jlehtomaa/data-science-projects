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
python dynamic_risk_assessment/apicalls.py
```

You can also run the full pipeline with:
```python
python fullprocess.py
```

All model details are contained in the `conf/config.json` file. To toggle between the training and production datasets, simply switch the `path` parameters as follows:

`input_data`: either `practice_data` or `source_data`

`trained_model`: either `models/trained_model.pkl` or `practice_models/trained_model.pkl`

`model_score`: either `models/latest_score.txt` or `practice_models/latest_score.txt`

`reports`: either `models/confmatrix2.png` or `practice_models/confmatrix.png`

`api_returns`: either `models/api_returns2.txt` or `practice_models/api_returns.txt`