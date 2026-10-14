# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created by LaShelleP in October 2026 as part of the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project for WGU D501. It is a binary classifier built with scikit-learn's RandomForestClassifier, using default hyperparameters and a fixed random_state of 42 so that results are reproducible. Categorical features are one-hot encoded and the label is binarized during preprocessing. The trained model and encoder are saved as pickle files in the model/ folder and served through a FastAPI application.

## Intended Use
The model predicts whether a person's annual income is above or below $50,000 based on demographic and employment information from the U.S. Census. It is intended for educational purposes, specifically to demonstrate building, testing, monitoring, and deploying a machine learning pipeline. It is not intended to be used for real decisions about individuals, such as hiring, lending, or eligibility for services.

## Training Data
The model was trained on the Census Income dataset (also known as the "Adult" dataset) from the UCI Machine Learning Repository, which was extracted from the 1994 U.S. Census database. The dataset provided in data/census.csv contains 32,561 rows and 15 columns, including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain and loss, hours per week, native country, and the salary label. The data was split using train_test_split, with 80% of the rows (26,048) used for training.

## Evaluation Data
The remaining 20% of the dataset (6,513 rows) was held out as a test set and was not used during training. The test set was processed with the same encoder and label binarizer that were fit on the training data, so that categories are represented consistently between training and evaluation.

## Metrics
The model was evaluated using precision, recall, and F1 score. Precision measures how many of the people predicted to earn over $50K actually do. Recall measures how many of the people who actually earn over $50K the model correctly identified. The F1 score is the harmonic mean of precision and recall. On the held-out test set, the model achieved a precision of 0.7419, a recall of 0.6384, and an F1 score of 0.6863.

Performance was also measured on slices of the test data, holding each value of every categorical feature fixed. These results are saved in slice_output.txt. Performance varies across slices, and slices with very few rows can show extreme scores, such as 1.0000 or 0.0000, that are not reliable because they are based on only a handful of examples.

## Ethical Considerations
The dataset includes sensitive attributes such as race, sex, and native country, and the model uses these features to make predictions. Because the data reflects income patterns from 1994, the model can learn and reproduce historical inequalities related to these attributes. The slice-based metrics should be reviewed to identify groups where the model performs noticeably worse. The model should not be used to make or influence decisions about real people.

## Caveats and Recommendations
The training data is more than 30 years old, so income levels, occupations, and demographic patterns have changed and the model's predictions may not reflect present-day conditions. The dataset is imbalanced, with roughly a quarter of people earning over $50K, which contributes to recall being lower than precision, meaning the model misses some higher earners. No hyperparameter tuning or cross-validation was performed. Recommended improvements include tuning the model's hyperparameters, using cross-validation, addressing class imbalance, evaluating fairness metrics across sensitive groups, and retraining on more recent data.