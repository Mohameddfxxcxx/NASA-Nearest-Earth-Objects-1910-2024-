# NASA-Nearest-Earth-Objects-1910-2024-

## Project Title

Predicting Hazardous Near-Earth Objects (NEOs)

## Overview

This project aims to analyze and predict hazardous near-Earth objects (NEOs) based on various features such as absolute magnitude, estimated diameter, and relative velocity. We utilize machine learning techniques to build a classification model that identifies potentially hazardous NEOs.


## Data

 The dataset used in this project is the "Near-Earth Objects (1910-2024)" dataset from NASA. It contains information about NEOs including their size, orbit, and whether they are hazardous.

## Dataset Details

Source: https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024/data

## Columns:

neo_id: Unique identifier for the NEO
name: Name of the NEO
orbiting_body: The body that the NEO is orbiting
absolute_magnitude: Absolute magnitude of the NEO
estimated_diameter_min: Minimum estimated diameter of the NEO
estimated_diameter_max: Maximum estimated diameter of the NEO
relative_velocity: Relative velocity of the NEO
is_hazardous: Whether the NEO is hazardous (target variable)


## Process

## Data Preprocessing

Loading Data: Read the dataset into a Pandas DataFrame.

import pandas as pd
df = pd.read_csv("/kaggle/input/nasa-nearest-earth-objects-1910-2024/nearest-earth-objects(1910-2024).csv")


Handling Missing Values: Removed rows with missing values.

df.dropna(inplace=True)

Removing Outliers: Used Interquartile Range (IQR) to remove outliers from selected features.

def remove_outliers(feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.65)
    iqr = q3 - q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    df = df.loc[(df[feature] < upper_limit) & (df[feature] > lower_limit)]


Feature Selection: Dropped non-numeric and target columns for model training.

X = df.drop(["neo_id", "name", "orbiting_body", "is_hazardous"], axis=1)
y = df["is_hazardous"]


Balancing Classes: Applied SMOTE to balance the classes.

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


Train-Test Split: Split the dataset into training and testing sets.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)


## Model Training and Evaluation

Model Training: Used RandomForestClassifier for classification.

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)


Model Evaluation: Evaluated the model using accuracy, confusion matrix, and feature importance.

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)


Accuracy: accuracy%

Confusion Matrix:

[[TN, FP],
 [FN, TP]]


 Feature Importance: Visualized feature importance to understand model behavior.

 import matplotlib.pyplot as plt
feature_importances = model.feature_importances_
feature_names = x_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', height=0.4)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()



Learning Curves: Plotted learning curves to visualize model performance over training examples.

from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    model, x_train, y_train, cv=5, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="lightblue", label="Training score std")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="lightgreen", label="Validation score std")
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="darkgreen", label="Validation score")
plt.title("Learning Curves")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()




## Results

Accuracy: accuracy%

Confusion Matrix:

[[TN, FP],
 [FN, TP]]


Feature Importance: Displayed in the feature importance chart.

Learning Curves: Visualized to show model performance and learning trends.



## Insights

The model performed well with an accuracy of accuracy%.
Feature importance analysis revealed that [important_features] are the most influential features in predicting hazardous NEOs.
Learning curves indicate that the model is [overfitting/underfitting] and may benefit from further tuning or additional data.


## Conclusion

This analysis demonstrates the ability to predict hazardous NEOs using machine learning techniques. Future work could involve exploring additional features, tuning model parameters, or using more advanced algorithms to improve prediction accuracy.


## References

https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024/data

Scikit-learn Documentation

Imbalanced-learn Documentation
