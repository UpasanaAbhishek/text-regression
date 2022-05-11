
### Submission - 04

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import autokeras as ak
import tensorflow as tf

# Reading the dataset
train = pd.read_csv('data/train_file.csv')
test = pd.read_csv('data/test_file.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

# Viewing the dataframes - 
train.head()
train.columns
test.head()
test.columns

sample_submission.head()
sample_submission.columns

# Checking for missing values - 
train.isnull().sum()
# Source column has 175 missing values

# Checking for data types -
train.dtypes

# Checking the data summary - 
train.describe().T

# Checking the value_counts of topic

train['Topic'].value_counts()

#economy      20486
#obama        16917
#microsoft    12911
#palestine     5618

test['Topic'].value_counts()

#economy      13436
#obama        11687
#microsoft     8946
#palestine     3219

# TITLE model 02 - 

x_title = train['Title']
y_title = train['SentimentTitle']

x_train_title, x_valid_title, \
y_train_title, y_valid_title = train_test_split(\
                               x_title, y_title, 
                               shuffle = True, test_size = 0.25)

# Initialize the text regressor.
reg = ak.TextRegressor(overwrite=True, max_trials=5, metrics=[tf.keras.metrics.MeanAbsoluteError()]) 

# Feed the text regressor with training data.
reg.fit(np.array(x_train_title), y_train_title, epochs=2)

predicted_y = reg.predict(np.array(x_valid_title))
# Evaluate the best model with testing data.
print(reg.evaluate(np.array(x_valid_title), y_valid_title))

y_train_title_pred = reg.predict(np.array(x_train_title))
mae_title_train = mean_absolute_error(y_train_title, y_train_title_pred)
print("mae_title_train: ", mae_title_train)

y_valid_title_pred = reg.predict(np.array(x_valid_title))
mae_title_valid = mean_absolute_error(y_valid_title, y_valid_title_pred)
print("mae_title_valid: ", mae_title_valid)


# HEADLINE model 02 - 

x_headline = train['Headline']
y_headline = train['SentimentHeadline']

x_train_headline, x_valid_headline, \
y_train_headline, y_valid_headline = train_test_split(\
                               x_headline, y_headline, 
                               shuffle = True, test_size = 0.25)
# Initialize the text regressor.
regh = ak.TextRegressor(overwrite=True, max_trials=1, metrics=[tf.keras.metrics.MeanAbsoluteError()]) 

regh.fit(np.array(x_train_title), y_train_title, epochs=2)


y_train_headline_pred = regh.predict(np.array(x_train_headline))
mae_headline_train = mean_absolute_error(y_train_headline, y_train_headline_pred)
print("mae_headline_train: ", mae_headline_train)

y_valid_headline_pred = regh.predict(np.array(x_valid_headline))
mae_headline_valid = mean_absolute_error(y_valid_headline, y_valid_headline_pred)
print("mae_headline_valid: ", mae_headline_valid)

print("Score (training):", (1-((0.4*mae_title_train+(0.6*mae_headline_train)))))
print("Score (validation):", (1-((0.4*mae_title_valid+(0.6*mae_headline_valid)))))

# Overall models for final predictions on external testing file - 

x_title_train = train['Title']
y_title_train = train['SentimentTitle']
#x_title_train_transformed = text_transformation(x_title_train)

reg.fit(np.array(x_title_train), y_title_train)

x_title_test = test['Title']
#x_title_test_transformed = text_transformation(x_title_test)
x_title_test = reg.predict(np.array(x_title_test))

test['SentimentTitle'] = x_title_test

x_headline_train = train['Headline']
y_headline_train = train['SentimentHeadline']
#x_headline_train_transformed = text_transformation(x_headline_train)

regh.fit(np.array(x_headline_train), y_headline_train)

x_headline_test = test['Headline']
#x_headline_test_transformed = text_transformation(x_headline_test)
x_headline_test = regh.predict(np.array(x_headline_test))


test['SentimentHeadline'] = x_headline_test

submission = test[['IDLink', 'SentimentTitle', 'SentimentHeadline']]

submission.to_csv('submission_04.csv',index=False)

print("Final submission saved!!")