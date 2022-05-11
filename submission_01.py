### Submission - 01

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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


def text_preprocessing(text_series):
    preprocessed_text = []
    for i in list(text_series):
        preprocessed_text.extend([i.strip().lower()])
    return preprocessed_text

cv = CountVectorizer(max_features = 100, ngram_range=(1,2))

def text_transformation(text_series, cv = cv):
    text_list = text_preprocessing(text_series)
    cv.fit(text_list)
    cv_transformed_x = cv.transform(text_list)
    return TfidfTransformer().fit_transform(cv_transformed_x)


# TITLE model 01 - 

x_title = train['Title']
y_title = train['SentimentTitle']

x_train_title, x_valid_title, \
y_train_title, y_valid_title = train_test_split(\
                               x_title, y_title, 
                               shuffle = True, test_size = 0.25)

x_train_title_transformed = text_transformation(x_train_title)
x_valid_title_transformed = text_transformation(x_valid_title)

model_1 = RandomForestRegressor(n_estimators=50,
                      max_depth=30,
                      random_state=0, n_jobs = -1)

model_1.fit(x_train_title_transformed, y_train_title)

y_train_title_pred = model_1.predict(x_train_title_transformed)
mae_title_train = mean_absolute_error(y_train_title, y_train_title_pred)
print("mae_title_train: ", mae_title_train)

y_valid_title_pred = model_1.predict(x_valid_title_transformed)
mae_title_valid = mean_absolute_error(y_valid_title, y_valid_title_pred)
print("mae_title_valid: ", mae_title_valid)


# HEADLINE model 01 - 

x_headline = train['Headline']
y_headline = train['SentimentHeadline']

x_train_headline, x_valid_headline, \
y_train_headline, y_valid_headline = train_test_split(\
                               x_headline, y_headline, 
                               shuffle = True, test_size = 0.25)

x_train_headline_transformed = text_transformation(x_train_headline)
x_valid_headline_transformed = text_transformation(x_valid_headline)

model_1 = RandomForestRegressor(n_estimators=50,
                      max_depth=30,
                      random_state=0, n_jobs = -1)

model_1.fit(x_train_headline_transformed, y_train_headline)

y_train_headline_pred = model_1.predict(x_train_headline_transformed)
mae_headline_train = mean_absolute_error(y_train_headline, y_train_headline_pred)
print("mae_headline_train: ", mae_headline_train)

y_valid_headline_pred = model_1.predict(x_valid_headline_transformed)
mae_headline_valid = mean_absolute_error(y_valid_headline, y_valid_headline_pred)
print("mae_headline_valid: ", mae_headline_valid)

print("Score (training):", (1-((0.4*mae_title_train+(0.6*mae_headline_train)))))
print("Score (validation):", (1-((0.4*mae_title_valid+(0.6*mae_headline_valid)))))

# Overall models for final predictions on external testing file - 

x_title_train = train['Title']
y_title_train = train['SentimentTitle']
x_title_train_transformed = text_transformation(x_title_train)

model_1.fit(x_title_train_transformed, y_title_train)

x_title_test = test['Title']
x_title_test_transformed = text_transformation(x_title_test)
x_title_test = model_1.predict(x_title_test_transformed)

test['SentimentTitle'] = x_title_test

x_headline_train = train['Headline']
y_headline_train = train['SentimentHeadline']
x_headline_train_transformed = text_transformation(x_headline_train)

model_1.fit(x_headline_train_transformed, y_headline_train)

x_headline_test = test['Headline']
x_headline_test_transformed = text_transformation(x_headline_test)
x_headline_test = model_1.predict(x_headline_test_transformed)

test['SentimentHeadline'] = x_headline_test

submission = test[['IDLink', 'SentimentTitle', 'SentimentHeadline']]

submission.to_csv('submission_01.csv', index = False)

print("Final submission saved!!")