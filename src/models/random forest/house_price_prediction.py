# **This notebook is an exercise in the [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) course.  
# You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/introduction).**

# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")  
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

X_train.head()

from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))

# Model 1 MAE: 24015
# Model 2 MAE: 23740
# Model 3 MAE: 23528
# Model 4 MAE: 23996
# Model 5 MAE: 23706

# Step 1: Evaluate several models
# Fill in the best model
best_model = model_3

# Step 2: Generate test predictions
# Define a model
my_model = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0) # Your code here


# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)



##################################
# Reference: https://www.kaggle.com/code/alexisbcook/missing-values

# Missing Values
# Three Approaches
# 1. Drop Columns with Missing Values
# 2. Imputation
# 3. Extension to Imputation - adding a column to indicate which values were imputed


# Score from Approach 1 (Drop Columns with Missing Values)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# MAE from Approach 1 (Drop columns with missing values):
# 183550.22137772635

# Score from Approach 2 (Imputation)
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# MAE from Approach 2 (Imputation):
# 178166.46269899711


# Score from Approach 3 (An Extension to Imputation)
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# MAE from Approach 3 (An Extension to Imputation):
# 178927.503183954

# As we can see, Approach 3 performed slightly worse than Approach 2.

# So, why did imputation perform better than dropping the columns?
# The training data has 10864 rows and 12 columns, where three columns contain missing data. For each column, less than half of the entries are missing. 
# Thus, dropping the columns removes a lot of useful information, and so it makes sense that imputation would perform better.

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# (10864, 12)
# Car               49
# BuildingArea    5156
# YearBuilt       4307
# dtype: int64

# Conclusion
# As is common, imputing missing values (in Approach 2 and Approach 3) yielded better results, 
# relative to when we simply dropped columns with missing values (in Approach 1).

# **This notebook is an exercise in the [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) course.  
# You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/missing-values).**


# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex2 import *
print("Setup Complete")

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

X_train.head()

# Step 1: Preliminary investigation

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Fill in the line below: How many rows are in the training data?
num_rows = 1168

# Fill in the line below: How many columns in the training data
# have missing values?
num_cols_with_missing = 3

# Fill in the line below: How many missing entries are contained in 
# all of the training data?
tot_missing = missing_val_count_by_column.values.sum()

# Since there are relatively few missing entries in the data (the column with the greatest percentage of missing values is missing less than 20% of its entries), 
# we can expect that dropping columns is unlikely to yield good results. This is because we'd be throwing away a lot of valuable data, and so imputation will likely perform better.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Step 2: Drop columns with missing values

# Fill in the line below: get names of columns with missing values
columns_names_with_missing_data = list(missing_val_count_by_column[missing_val_count_by_column > 0].keys()) # Your code here

# Fill in the lines below: drop columns in training and validation data
reduced_X_train = X_train.drop(columns_names_with_missing_data, axis=1)
reduced_X_valid = X_valid.drop(columns_names_with_missing_data, axis=1)

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# MAE (Drop columns with missing values):
# 17837.82570776256

# Step 3: Imputation
from sklearn.impute import SimpleImputer

# Fill in the lines below: imputation
my_imputer = SimpleImputer() # Your code here
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Fill in the lines below: imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# MAE (Imputation):
# 18062.894611872147


# Given that thre are so few missing values in the dataset, we'd expect imputation to perform better than dropping columns entirely. 
# However, we see that dropping columns performs slightly better! While this can probably partially be attributed to noise in the dataset, 
# another potential explanation is that the imputation method is not a great match to this dataset. That is, maybe instead of filling in the mean value, 
# it makes more sense to set every missing value to a value of 0, to fill in the most frequently encountered value, or to use some other method. 
# For instance, consider the GarageYrBlt column (which indicates the year that the garage was built). 
# It's likely that in some cases, a missing value could indicate a house that does not have a garage. 
# Does it make more sense to fill in the median value along each column in this case? 
# Or could we get better results by filling in the minimum value along each column? 
# It's not quite clear what's best in this case, but perhaps we can rule out some options immediately - 
# for instance, setting missing values in this column to 0 is likely to yield horrible results!


# Step 4: Generate test predictions
# Preprocessed training and validation features
# Since GarageYrBlt represents the year of the garage is built, and when its equal to zero, 
# means that there is no garage built. In this case, we can use the Approach 3, Extention Imputation
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

X_train_plus['GarageYrBlt_was_missing'] = X_train['GarageYrBlt'].isnull()
X_valid_plus['GarageYrBlt_was_missing'] = X_valid['GarageYrBlt'].isnull()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

final_X_train = imputed_X_train_plus
final_X_valid = imputed_X_valid_plus

# Define and fit model
model = RandomForestRegressor(n_estimators=1000, random_state=0)
model.fit(final_X_train, y_train)

# Get validation predictions and MAE
preds_valid = model.predict(final_X_valid)
print("MAE (Your approach):")
print(mean_absolute_error(y_valid, preds_valid))
# MAE (Your approach):
# 17543.774589856494
# Fill in the line below: preprocess test data
X_test_plus = X_test.copy()
X_test_plus['GarageYrBlt_was_missing'] = X_test_plus['GarageYrBlt'].isnull()
final_X_test = pd.DataFrame(my_imputer.transform(X_test_plus))

# Fill in the line below: get test predictions
preds_test = model.predict(final_X_test)

# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)


# Next, follow the instructions below:
# 1. Begin by clicking on the **Save Version** button in the top right corner of the window.  This will generate a pop-up window.  
# 2. Ensure that the **Save and Run All** option is selected, and then click on the **Save** button.
# 3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 4. Click on the **Data** tab near the top of the screen.  Then, click on the file you would like to submit, and click on the **Submit** button to submit your results to the leaderboard.

# You have now successfully submitted to the competition!

# If you want to keep working to improve your performance, select the **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.