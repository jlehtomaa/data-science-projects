# Model Card

For additional information see the [Model Card paper.](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

The model used in this project is a scikit-learn random forest classifier.
The model parameters are those defined in `./conf/config.yaml` For any parameter
values not specified in this file, the scikit-learn defaults are used.

## Intended Use

The model is used to predict the salary class (whether or not salary is below USD 50K) based on the US census dataset.

## Training Data

The training dataset is the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income). The base year is 1994. It has 14 features and a binary income label.

## Evaluation Data
The evaluation dataset is extracted from the full data with a random 20-80 train-test split.

## Metrics
The model evaluation considers three different metrics, with the following performance:

- precision: 0.73
- recall: 0.64
- fbeta: 0.68

## Ethical Considerations

The dataset contains several features (native country, race, sex) that, if not properly taken into account, might inject severe biases to the model. Consequently, some use cases, such as providing bank loan decision, might be particularly fraught.

## Caveats and Recommendations

The dataset is alredy 30 years old, and should be heavily revised before any serious use.