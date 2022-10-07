# Build an ML Pipeline for Short-Term Rental Prices in NYC

You are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. You need to estimate the typical price for a given property based 
on the price of similar properties. Your company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

In this project you will build such a pipeline.

## Running the code

To run the full pipeline, including training and testing, use:
```python
mlflow run .
```

For individual steps in `download, basic_cleaning, data_check, data_split, train_random_forest`, you can also use:

```python
mlflow run . -P steps=download,basic_cleaning
```
To modify the hyperparameters through the hydra config, use:

```python
mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10"
```
