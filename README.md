# Phase 3 Review - Predictive Classification Workflow

## Students Will Be Able To
- Understand the overall process to solve a predictive classification problem
- Understand and implement multiple classification algorithms
- Implement cross-validation techniques
- Handle class imbalance using SMOTE
- Perform GridSearch to determine optimal hyperparameter combinations
- Create Pipelines to streamline the modeling process

## Business and Data Understanding

This dataset was downloaded from [Kaggle](https://www.kaggle.com/uciml/adult-census-income) and contains information on adult incomes. We are trying to predict whether or not an individual's yearly salary was greater than or equal to \$50,000 (binary classification). The column `salary` will be either a 0 (less than \\$50,000) or a 1 (greater than or equal to \$50,000). The metric we will be using is accuracy.

## Tasks

### Data Preparation

#### Train-Test Split

We will be using cross-validation for the duration of this notebook. Please perform two train-test splits. First splitting the entire dataframe into train and test sets and then splitting the *train* data into *training and validation sets. Use `random_state=2021` and `test_size=.15` in both splits for reproducibility. We will be using the train and validation sets for the majority of this notebook. **The test set should be left alone until the very end**.

#### Preprocessing

Please perform the standard data preprocessing steps on the training and validation data:
- Check for missing data and impute if necessary (or drop)
- Scale numerical data
- OneHotEncode categorical data

### Modeling

#### Baseline Logistic Regression

Create a `LogisticRegression` model and fit it on the preprocessed training data. Check the performance of the model on the training and validation data. 

Please plot a confusion matrix of the model's predictions and compare it to the previous performance metric. What might be causing the accuracy score to be misleading? (*HINT*: Check the value counts of your target variable)

#### Second Logistic Regression

Please use SMOTE ([documentation here](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)) to adjust the imbalance of target classes. You can use your preprocessed training data at this step. Once you have resampled your training data, please fit another Logistic Regression model and check its performance using the training and validation data. Plot another confusion matrix and explain whether or not resampling helped improve the performance of your model.

Inspect the coefficients of this model and report the 5 features with the largest coefficients and the 5 features with the lowest coefficients. ([documentaion here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))

#### Third LogisticRegression

Please create a third and final LogisticRegression model and adjust at least one hyperparameter related to the regularization of the model. Fit the model on the preprocessed, resampled training data. Check the performance on the training and validation data.

Once again, inspect the coefficients of this model and report the 5 features with the largest coefficients and the 5 features with the lowest coefficients. ([documentaion here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)). How have the coefficients changed from regularization?

#### DecisionTreeClassifier

Create a `DecisionTreeClassifier` using hyperparameters of your choosing. Please fit the model on the training data and check its performance on the train and validation sets.

#### GridSearch RandomForest

For your final model, please use `GridSearchCV` ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)) to determine the optimal hyperparameter combination for a `RandomForestClassifier`. Assign the GridSearch's `best_estimator_` to a variable and check its performance on the train and validation sets.

#### Model Evaluation

Of your five models created, which performs best? Please assign this model to the variable `best_model`. 

Transform your `X_test` using the same preprocessing tools fitted on your `X_train`. Calculate performance metrics on this final test set. How did the model perform on the *real test set*?

### Pipelines

Using your best performing model, please create a Pipeline ([documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)) to perform the entire modeling process. The Pipeline should impute missing data using `SimpleImputer`, scale numerical data using `StandardScaler` and one hot encode categorical data with `OneHotEncoder`.

For this pipeline, you will need to make use of `make_column_selector` ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector) and `make_column_transformer` ([documentation](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer)).

You will need to perform another train test split. Just perform a single split (you will not need a validation set) using `random_state=2021` and `test_size=.25`.

Once your pipeline has been created, pass it into `cross_val_score` along with your training data to calculate the 5 Fold cross-validation accuracy score. How does the average accuracy score of these 5 splits compare to your best performing model's accruacy score from the previous section?

Use your pipeline to make predictions on your test set. How does the accuracy score for this test set compare to the the score from the previous section?