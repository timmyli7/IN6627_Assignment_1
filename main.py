# Importing the required libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
def convert_categorical_value(df) :
    # encode categorical variables using Label Encoder
    df_categorical = df.select_dtypes(include=['object'])
    encoder = preprocessing.LabelEncoder()
    df_categorical = df_categorical.apply(encoder.fit_transform)
    print(df_categorical)

    df = df.drop(df_categorical.columns, axis=1)
    df = pd.concat([df, df_categorical], axis=1)

    # convert target variable income to categorical
    df['income'] = df['income'].astype('category')
    return df

def data_preprocessing_remove_unwant_values(df, unwanted_val):
    # Check what other features has data like ?
    print(df_categorical.apply(lambda x: x == unwanted_val, axis=0).sum())
    df = df[~(df == unwanted_val).any(axis=1)]
    return df


#Read the updated dataset into df
df = pd.read_csv('adult_data_updated.csv')

# select all categorical variables
df_categorical = df.select_dtypes(include=['object'])

# dropping the data with "?"s
data_preprocessing_remove_unwant_values(df,'?')

print(df) # 30162 data count, 32561 - 30162 = 2399 data was dropped due to ? values

df = convert_categorical_value(df)



# Importing train-test-split
from sklearn.model_selection import train_test_split
# Putting feature variable to X
x_train = df.drop('income',axis=1)

# Putting response variable to y
y_train = df['income']

df_test = pd.read_csv('adult_test_updated.csv')
df_test = data_preprocessing_remove_unwant_values(df_test, '?')
df_test = convert_categorical_value(df_test)
x_test = df_test.drop('income',axis=1)
y_test = df_test['income']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def decision_tree_classifier_train(dtree, parameters, decision_tree, x_train, y_train, x_test, y_test):
    decision_tree.fit(x_train, y_train)
    # Making predictions
    y_pred_default = dtree.predict(x_test)
    print(classification_report(y_test, y_pred_default))
    return y_pred_default

def print_confusion_matrix(y_test, y_pred_default):
    # Printing classification report
    print(classification_report(y_test, y_pred_default))
    # Printing confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred_default)
    classes = ['<=50K', '>50K']
    cm_df = pd.DataFrame(cm.T, index=classes, columns=classes)
    cm_df.index.name = 'Predicted'
    cm_df.columns.name = 'True'
    print(cm_df)
    print(accuracy_score(y_test, y_pred_default))

#################### Training dataset #######################
# Importing decision tree classifier from sklearn library
parameters = {}

# ### First modification, max_depth = 5
# dt_default = DecisionTreeClassifier(max_depth=5)
# y_pred_result = decision_tree_classifier_train(parameters, dt_default, x_train, y_train, x_test, y_test)
#

### Second modification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
# Define the parameter grid

###Training grid, comment out since the best model has been identified.

# parameters = {
#     'criterion': ['gini', 'entropy'],
#     'splitter': ['best', 'random'],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     # 'min_samples_split': range(5, 200, 20),
#     'min_samples_leaf': range(5, 200, 50),
#     'min_samples_split': range(5, 200, 50),
#     'max_features': [None, 'auto', 'sqrt', 'log2'],
#     'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
#     'class_weight': [None, 'balanced'],
#     'min_impurity_decrease': [0.0, 0.1, 0.2]
# }

dtree = DecisionTreeClassifier(class_weight=None,
                               criterion='gini',
                               max_depth=None,
                               max_leaf_nodes=50,
                               min_impurity_decrease=0.0,
                               min_samples_leaf=5,
                               max_features=None,
                               min_samples_split=155,
                               splitter='best'
)
# dtree = DecisionTreeClassifier()
# fit tree on training data
tree = GridSearchCV(dtree, parameters,
                    cv=n_folds,
                    scoring="accuracy",return_train_score=True)

import time

# Measure training time
start_train_time = time.time()
tree.fit(x_train, y_train)
end_train_time = time.time()

print("Training Time: {:.4f} seconds".format(end_train_time - start_train_time))

# Measure prediction time
start_pred_time = time.time()
y_pred = tree.predict(x_test)
end_pred_time = time.time()

print("Prediction Time: {:.4f} seconds".format(end_pred_time - start_pred_time))

# scores of GridSearch CV
scores = tree.cv_results_
print(pd.DataFrame(scores).head())

# Print the best parameters and corresponding score
print("Best Parameters:", tree.best_params_)
print("Best Cross-validation Score:", tree.best_score_)

# Evaluate on test data
test_score = tree.score(x_test, y_test)
print("Test Score:", test_score)

def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10,8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' - Feature Importance')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

# Assuming tree is the GridSearchCV object and best estimator has been fit
best_tree = tree.best_estimator_
plot_feature_importance(best_tree.feature_importances_, x_train.columns, 'Decision Tree')

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(best_tree, filled=True, feature_names=x_train.columns, class_names=['<=50K', '>50K'], rounded=True)
plt.show()

print("################################# Result #####################################")
print_confusion_matrix(y_test, y_pred)

import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()