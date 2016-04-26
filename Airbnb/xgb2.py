import numpy as np
import pandas as pd
import csv
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(r"C:\Anaconda\Lib\xgboost-master\wrapper")
import xgboost as xgb


def dcg_score(y_true, y_score, k=5, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
    
def ndcg_score(y_true, y_score, k=5, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best



def dcg_at_k(r, k=5, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def score_predictions(preds, truth, n_modes=5):
    """
    preds: pd.DataFrame
      one row for each observation, one column for each prediction.
      Columns are sorted from left to right descending in order of likelihood.
    truth: pd.Series
      one row for each obeservation.
    """
    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0

    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return score.mean(axis=0)


def ndcg5(preds, truth):
    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0
    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return score
    
    
    
def ndcg5_xgb(preds, dtrain):
    labels = dtrain.get_label()
    labels = labels.astype(int)
    labels = le.inverse_transform(labels)
    preds = np.argsort(preds)
    preds = np.fliplr(preds)
    preds = le.inverse_transform(preds)
    preds = preds[:,:5]

    sub_train_val = pd.DataFrame(preds, columns=['1','2','3','4','5'])
    sub_test_val = pd.Series(labels)

    error = score_predictions(sub_train_val, sub_test_val)
    return "ndcg5_xgb", 1-error


#np.random.seed(0)


#Loading data
df_train = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/train_users_2.csv')
df_test = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/test_users.csv')
df_countries = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/countries.csv')
#df_sessions = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/sessions.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#print df_sessions.head(50)

#Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)  
 
"""
#create countries dict
df_countries['country_destination'] = le.fit_transform(df_countries['country_destination'])  
df_train['country_destination'] = le.fit_transform(df_train['country_destination'])  
print df_train['country_destination']
countries_1 = dict(zip(df_countries['country_destination'], df_countries['distance_km']))
countries_2 = dict(zip(df_countries['country_destination'], df_countries['language_levenshtein_distance']))

df_train['distance_km'] = df_train['country_destination'].map(countries_1)
df_train['language_levenshtein_distance'] = df_train['country_destination'].map(countries_2)

df_train = df_train.drop(['country_destination'], axis=1)
"""
#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)



#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
#df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
#df_all['tfa_year'] = tfa[:,0]
#df_all['tfa_month'] = tfa[:,1]
#df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>75), -1, av)
#df_all['age'] = df_all['age'].replace(-1, np.nan)
#df_all['age'] = df_all['age'].fillna(df_all['age'].mean())

#Gender
df_all['gender'] = df_all['gender'].replace('-unknown-', 'OTHER')
gender_other = df_all['gender'] == "OTHER"
#gender_arr = ['MALE', 'FEMALE']
#df_all.gender[gender_other] = np.random.choice(gender_arr, df_all.gender[gender_other].shape[0], p=[0.5, 0.5])

"""
#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                                'first_affiliate_tracked', 'signup_app', 'first_device_type']
df_all = df_all.drop(['first_browser'], axis=1)
#, 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
"""

#Label-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
                                'first_affiliate_tracked', 'signup_app', 'first_device_type']
df_all = df_all.drop(['first_browser'], axis=1)
#, 'first_browser']
le2 = LabelEncoder()
for f in ohe_feats:
    df_all[f] = le2.fit_transform(df_all[f])

#Splitting train and test
vals = df_all.values
X = vals[:piv_train]
test = vals[piv_train:]



#Parameters for XGB
params = {"objective": "multi:softprob",
          "num_class": 12,
          "eta": 0.3,
#          "max_depth": 10,
          "subsample": 0.5,
          "colsample_bytree": 0.5,
          "seed": np.random.randint(10,1000),
          "silent": 1
          }
num_trees = 25

y_pred_test = np.zeros((62096,12))
err_list = []
num_cv = 5

for i in range (num_cv):
    #cross-validation split

    #Gender
    for i in range(X.shape[0]):
        if X[i,0] == 2:
            if np.random.randint(1,3) == 1:
                X[i,0] = 1
            else:
                X[i,0] = 0

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=np.random.randint(10,1000))
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_test, y_test)
    dtest = xgb.DMatrix(test)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, num_trees, evals=watchlist, early_stopping_rounds=5)

    print("Validating...")
    y_pred = gbm.predict(xgb.DMatrix(X_test))
    y_pred = np.argsort(y_pred)
    y_pred = np.fliplr(y_pred)
    y_pred = le.inverse_transform(y_pred)
    y_pred = y_pred[:,:5]
    y_test = le.inverse_transform(y_test)

    sub_train_val = pd.DataFrame(y_pred, columns=['1','2','3','4','5'])
    sub_test_val = pd.Series(y_test)

    error = score_predictions(sub_train_val, sub_test_val)
    err_list.append(error)
    print('error', error)
    print("Predicting...")
    y_pred_test = y_pred_test + gbm.predict(dtest)

y_pred_test = y_pred_test/num_cv
#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for j in range(len(id_test)):
    idx = id_test[j]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_test[j])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('xgb3_logloss_eval.csv',index=False)
print err_list
print ("avg error", sum(err_list)/len(err_list))