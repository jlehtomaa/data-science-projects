# build-ml-pipeline

This subfolder corresponds to the third project in the Udacity nanodegree `Machine learning devops engineer`.

## Using the repository

Add dependencies with `poetry`:

```python
poetry install
```

Train the ML model:
```python
python train.py
```

Serve the app by running:
```python
python main.py
```

Once the server is running, query it by running:
```python
python live_query.py
```

Train the model with

```python
python train.py
```

A push to this repo subfolder automatically triggers the unit tests in the `tests` folder. To run them manually, simply use:

```python
pytest -vv
```