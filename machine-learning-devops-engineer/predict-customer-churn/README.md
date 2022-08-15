# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree by Udacity.

## Project Description

Train two models to predict which credit card customers are likely to churn.
The two models considered are a scikit learn random forest classifier and a logistic regression model.

## Running the analysis (with poetry and pyenv)

Create a new virtual environment using the pyenv virtualenv plugin in python 3.8.8:

```bash
pyenv virtualenv 3.8.8 predict-customer-churn-env
```

Activate the virtual environment:

```bash
pyenv activate predict-customer-churn-env
```

Switch to the project folder. Then, install dependencies:

```bash
poetry add $(cat requirements.txt)
```

If errors occur, updating the pip for poetry often helps: ```poetry run pip install --upgrade pip```.

Run all tests via:

```bash
pytest churn_script_logging_and_tests.py
```

Run the main training script with:

```bash
python churn_library.py
```


