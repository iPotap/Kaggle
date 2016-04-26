import numpy as np
import pandas as pd
import csv
import datetime
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append(r"C:\Anaconda\Lib\xgboost-master\wrapper")
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, feature_extraction, preprocessing

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

    

def StackModels(train, test, y, clfs, n_folds): # train data (pd data frame), test data (pd date frame), Target data,
                                                # list of models to stack, number of folders

# StackModels() performs Stacked Aggregation on data: it uses n different classifiers to get out-of-fold 
# predicted probabilities of signal for train data. It uses the whole training dataset to obtain predictions for test.
# This procedure adds n meta-features to both train and test data (where n is number of models to stack).

    print("Generating Meta-features")
    skf = list(StratifiedKFold(y, n_folds))
    training = train.as_matrix()
    testing = test.as_matrix()
    scaler = StandardScaler().fit(training)
    train_all = scaler.transform(training)
    test_all = scaler.transform(testing)
    blend_train = np.zeros((training.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((testing.shape[0], len(clfs)))   # Number of testing data x Number of classifiers
    
    for j, clf in enumerate(clfs):
        
        print ('Training classifier [%s]' % (j))
        for i, (tr_index, cv_index) in enumerate(skf):
            
            print ('stacking Fold [%s] of train data' % (i))
            
            # This is the training and validation set (train on 2 folders, predict on a 3d folder)
            X_train = training[tr_index]
            Y_train = y[tr_index]
            X_cv = training[cv_index]
            scaler=StandardScaler().fit(X_train)
            X_train=scaler.transform(X_train)
            X_cv=scaler.transform(X_cv)
                                  
            clf.fit(X_train, Y_train)
            pred = clf.predict_proba(X_cv)
            
            if pred.ndim==1:  # XGBoost produces ONLY probabilities of success as opposed to sklearn models
                 
                 blend_train[cv_index, j] = pred
                 
            else:
                
                blend_train[cv_index, j] = pred[:, 1]
        
        print('stacking test data')        
        clf.fit(train_all, y)
        pred = clf.predict_proba(test_all)
        
        if pred.ndim==1 :      # XGBoost produces ONLY probabilities of success as opposed to sklearn models
        
           blend_test[:, j] = pred
           
        else:
            
           blend_test[:, j] = pred[:, 1]

    X_train_blend=np.concatenate((training, blend_train), axis=1)
    X_test_blend=np.concatenate((testing, blend_test), axis=1)
    return X_train_blend, X_test_blend, blend_train, blend_test

#np.random.seed(0)


#Loading data
#df_train = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/train_users_2.csv')
#df_test = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/test_users.csv')
df_train = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/train_users_sessions7.csv')
df_test = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/test_users_sessions7.csv')
#df_countries = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/countries.csv')
labels = df_train['country_destination'].values
#df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]

#Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)  
 

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
#df_all = df_all.drop(['id', 'user_id', 'date_first_booking'], axis=1)
df_all = df_all.drop(['user_id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)


#####Feature engineering#######
#date_account_created

dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]

df_all['dac_year'].fillna(df_all['dac_year'].median(),inplace = True);
df_all['dac_month'].fillna(df_all['dac_month'].median(),inplace = True);
df_all['dac_day'].fillna(df_all['dac_day'].median(),inplace = True);


df_all['dac_year']=df_all['dac_year'].apply(str)
df_all['dac_month']=df_all['dac_month'].apply(str)
df_all['dac_day']=df_all['dac_day'].apply(str)

df_all['dac_temp'] = df_all['dac_year'] + '-' + df_all['dac_month'] + '-' + df_all['dac_day']

df_all['dac_weekday'] = pd.DatetimeIndex(df_all['dac_temp']).weekday

av = df_all.dac_weekday.values
df_all['dac_weekday']  = np.where(np.logical_or(av == 5, av == 6), 1, 0)

df_all = df_all.drop(['dac_temp'], axis=1)
#df_all = df_all.drop(['dac_day'], axis=1)
df_all = df_all.drop(['date_account_created'], axis=1)

df_all['dac_year']=df_all['dac_year'].apply(str)
df_all['dac_month']=df_all['dac_month'].apply(str)
df_all['dac_day']=df_all['dac_day'].apply(str)


#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)

df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]

df_all['tfa_year']=df_all['tfa_year'].apply(str)
df_all['tfa_month']=df_all['tfa_month'].apply(str)
df_all['tfa_day']=df_all['tfa_day'].apply(str)

df_all['tfa_hour'] = tfa[:,3]
df_all['tfa_hour'].fillna(df_all['tfa_hour'].median(),inplace = True);
df_all['tfa_hour']=df_all['tfa_hour'].apply(int)
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Create time feature
av = df_all.tfa_hour.values
df_all['tfa_hour_group']  = np.where(np.logical_and(av >= 0, av <= 2), 1, 0)
bv = df_all.tfa_hour_group.values
df_all['tfa_hour_group']  = np.where(np.logical_and(av > 2, av <= 6), 2, bv)
bv = df_all.tfa_hour_group.values
df_all['tfa_hour_group']  = np.where(np.logical_and(av > 6, av <= 9), 3, bv)
bv = df_all.tfa_hour_group.values
df_all['tfa_hour_group']  = np.where(np.logical_and(av > 9, av <= 18), 4, bv)
bv = df_all.tfa_hour_group.values
df_all['tfa_hour_group']  = np.where(np.logical_and(av > 18, av <= 24), 5, bv)

#Create summer/winter feature
av = df_all.dac_month.values
df_all['dac_summer_winter']  = np.where(np.logical_and(av >= 6, av <= 8), 1, 0)
bv = df_all.dac_summer_winter.values
df_all['dac_summer_winter']  = np.where(np.logical_or(av == 12, av <= 2), 2, bv)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>70), -1, av)

#Age_grouping
av = df_all.age.values
df_all['age_group']  = np.where(np.logical_and(av >= 14, av <= 20), 1, 0)
bv = df_all.age_group.values
df_all['age_group']  = np.where(np.logical_and(av > 20, av <= 24), 2, bv)
bv = df_all.age_group.values
df_all['age_group']  = np.where(np.logical_and(av > 24, av <= 30), 3, bv)
bv = df_all.age_group.values
df_all['age_group']  = np.where(np.logical_and(av > 30, av <= 35), 4, bv)
bv = df_all.age_group.values
df_all['age_group']  = np.where(np.logical_and(av > 35, av <= 45), 5, bv)
bv = df_all.age_group.values
df_all['age_group']  = np.where(np.logical_and(av > 45, av <= 60), 6, bv)
bv = df_all.age_group.values
df_all['age_group']  = np.where(np.logical_and(av > 60, av <= 70), 7, bv)
bv = df_all.age_group.values
df_all['age_group']  = np.where(av==-1, 8, bv)

#Gender
df_all['gender'] = df_all['gender'].replace('-unknown-', 'OTHER')
gender_other = df_all['gender'] == "OTHER"
gender_arr = ['MALE', 'FEMALE']

#Age-gender
df_all['age_gender'] = df_all['age_group'].apply(str) + df_all['gender']
df_all['age_group'] = df_all['age_group'].apply(str) 
df_all = df_all.drop(['age_group'], axis=1)

#drop 
#df_all = df_all.drop(['first_browser'], axis=1)
#df_all = df_all.drop(['first_device_type'], axis=1)
df_all = df_all.drop(['modify'], axis=1)

#couple extra features
df_all['sum_zeros'] = (df_all == 0).astype(int).sum(axis=1)

"""
'Unnamed: 0', 'age' 'tfa_hour', 'dac_day', 'first_browser', 'tfa_day',
 'signup_flow', 'affiliate_channel',  'secs_elapsed', 'dac_month', 'first_affiliate_tracked', '-unknown-_x', 
 'Mac Desktop', 'first_device_type', 'view_search_results' 'view' 'affiliate_provider' 'change_trip_characteristics'
 'click' 'similar_listings'
"""
del df_train
del df_test

df_all.to_csv('C:/Users/1/Desktop/kaggle/airbnb/temp/df_all_5.csv',index=False)
del df_all

strong_features=['Unnamed: 0',
 'dac_day',
 'tfa_hour',
 'age',
 'age_gender',
 'dac_month',
 'first_browser',
 'affiliate_channel',
 'secs_elapsed',
 'signup_flow',
 'first_affiliate_tracked',
 'view_search_results',
 'Mac Desktop',
 'first_device_type',
 'affiliate_provider',
 'view',
 '-unknown-_x',
 'click',
 'wishlist_content_update',
 'Windows Desktop',
 'p3',
 'change_trip_characteristics',
 'signup_method',
 'similar_listings',
 'submit',
 'data',
 'p5',
 'message_post_x',
 'gender',
 '-unknown-',
 'tfa_hour_group',
 'dac_year',
 'update_listing',
 'dashboard',
 'create_phone_numbers',
 'user_profile',
 'contact_host',
 'pending',
 'iPad Tablet',
 'language',
 'signup_app',
 'message_thread',
 'dac_weekday',
 'iPhone',
 'header_userpic',
 'booking_request',
 'user_wishlists',
 'cancellation_policies',
 'confirm_email_link',
 'your_trips',
 'edit_profile',
 'listing_reviews_page',
 'post_checkout_action',
 'secs_elapsed_translate',
 '-unknown-_y',
 'create_user',
 'user_profile_content_update',
 'update_user_profile',
 'change_contact_host_dates',
 'user_social_connections',
 'message_post_y',
 'login',
 'profile_verifications',
 'listing_reviews',
 'p1',
 'user_listings',
 'create_listing',
 'your_listings',
 'manage_listing',
 'reservations',
 'unavailable_dates',
 'account_notification_settings',
 'message_to_host_change',
 'payment_instruments',
 'wishlist',
 'p4',
 'message_to_host_focus',
 'user_reviews',
 'partner_callback',
 'notifications',
 'instant_book',
 'apply_coupon',
 'signup_login_page',
 'translate_listing_reviews',
 'list_your_space',
 'request_new_confirm_email',
 'Android App Unknown Phone/Tablet',
 'update_user',
 'message_inbox',
 'send_message',
 'oauth_login',
 'profile_references']


y_pred_test = np.zeros((62096,12))
y_pred_test_groups1 = np.zeros((62096,12))  
err_list = []

df_all_new = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/temp/df_all_5.csv')

#Assign others to male or female randomly
df_all_new.gender[gender_other] = np.random.choice(gender_arr, df_all_new.gender[gender_other].shape[0], p=[0.45, 0.55])
print df_all_new.shape
#Splitting train and test
df_train, df_test = df_all_new[:piv_train], df_all_new[piv_train:]
print df_train.shape
df_train = df_train[df_train['tfa_year']==2014]
print df_train.shape

    
df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)
df_train = df_train[df_train['sum_zeros'] != 0]
df_test = df_test[df_test['sum_zeros'] != 0]
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
del df_all_new
#Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)  

#Label-encoding features
ohe_feats = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider',
                                 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'age_gender', 'dac_day', 'dac_month', 'dac_year', 'tfa_day', 'tfa_month', 'tfa_year']

le2 = LabelEncoder()
for f in ohe_feats:
    df_train[f] = le2.fit_transform(df_train[f])
    df_test[f] = le2.fit_transform(df_test[f])

vals_train = df_train.values
vals_test = df_test.values
X = vals_train
test = vals_test


#Parameters for XGB
params = {"objective": "multi:softprob",
                "eval_metric": 'merror',
                "min_child_weight": 5,
                "num_class": 12,
                "eta": 0.3,
                "subsample": 0.5,
                "colsample_bytree": 0.5,
                "seed": np.random.randint(10,1000),
                "silent": 1
                }
num_trees = 5

#cross-validation split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.015, random_state=np.random.randint(10,1000))
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
print y_pred_test.shape

y_pred_test_groups1 = y_pred_test


df_train = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/train_users_sessions7.csv')
labels = df_train['country_destination'].values
#Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)  

del df_train

y_pred_test_all = np.zeros((62096,12))
err_list = []
num_cv = 20
for i in range(num_cv):
    print i

    
    df_all_new = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/temp/df_all_5.csv')
    df_all_new = df_all_new.drop(['id'], axis=1)
    df_all_new = df_all_new.drop(['country_destination'], axis=1)
    
    #Assign others to male or female randomly
    df_all_new.gender[gender_other] = np.random.choice(gender_arr, df_all_new.gender[gender_other].shape[0], p=[0.45, 0.55])

    #Label-encoding features
    ohe_feats = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider',
                                     'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'age_gender', 'dac_day', 'dac_month', 'dac_year', 'tfa_day', 'tfa_month', 'tfa_year']
    
    le2 = LabelEncoder()
    for f in ohe_feats:
        df_all_new[f] = le2.fit_transform(df_all_new[f])

    #Splitting train and test
    vals = df_all_new.values
    X = vals[:piv_train]
    test = vals[piv_train:]

    #Parameters for XGB
    params = {"objective": "multi:softprob",
              "eval_metric": 'merror',
              "min_child_weight": 5,
              "num_class": 12,
              "eta": 0.3,
              "subsample": 0.5,
              "colsample_bytree": 0.5,
              "seed": np.random.randint(10,1000),
              "silent": 1
              }
    num_trees = 25

    #cross-validation split
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.015, random_state=np.random.randint(10,1000))
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
    y_pred_test_all = y_pred_test_all + gbm.predict(dtest)
y_pred_test_all = y_pred_test_all/num_cv

y_pred_test2_all = np.zeros((62096,12))    
err_list = []
num_cv = 20
for i in range(num_cv):
    print i
    df_all_new = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/temp/df_all_5.csv')
    df_all_new = df_all_new.drop(['id'], axis=1)
    df_all_new = df_all_new.drop(['country_destination'], axis=1)
    
    #Assign others to male or female randomly
    df_all_new.gender[gender_other] = np.random.choice(gender_arr, df_all_new.gender[gender_other].shape[0], p=[0.45, 0.55])

    #Label-encoding features
    ohe_feats = ['gender', 'signup_method', 'language', 'affiliate_channel', 'affiliate_provider',
                                     'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'age_gender', 'dac_day', 'dac_month', 'dac_year', 'tfa_day', 'tfa_month', 'tfa_year']
    
    le2 = LabelEncoder()
    for f in ohe_feats:
        df_all_new[f] = le2.fit_transform(df_all_new[f])

    #feacture selection
    df_all_new = df_all_new[strong_features]
    
    #Splitting train and test
    vals = df_all_new.values
    X = vals[:piv_train]
    test = vals[piv_train:]
    

    
    #Parameters for XGB
    params = {"objective": "multi:softprob",
              "eval_metric": 'merror',
              "min_child_weight": 5,
              "num_class": 12,
              "eta": 0.3,
              "subsample": 0.5,
              "colsample_bytree": 0.5,
              "seed": np.random.randint(10,1000),
              "silent": 1
              }
    num_trees = 25

    #cross-validation split
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.015, random_state=np.random.randint(10,1000))
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
    y_pred_test2_all = y_pred_test2_all + gbm.predict(dtest)
y_pred_test2_all = y_pred_test2_all/num_cv

y_pred_test_all = (0.8*y_pred_test_all+0.2*y_pred_test2_all)




y_pred_test_groups =  (0.9*y_pred_test_all+0.1*y_pred_test_groups1)
#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for j in range(len(id_test)):
    idx = id_test[j]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_test_groups[j])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('C:/Users/1/Desktop/kaggle/airbnb/temp/6_90_10_lb.csv',index=False)

y_pred_test_groups =  (0.8*y_pred_test_all+0.2*y_pred_test_groups1)
#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for j in range(len(id_test)):
    idx = id_test[j]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_test_groups[j])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('C:/Users/1/Desktop/kaggle/airbnb/temp/6_80_20_lb.csv',index=False)

y_pred_test_groups =  (0.7*y_pred_test_all+0.3*y_pred_test_groups1)
#Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for j in range(len(id_test)):
    idx = id_test[j]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred_test_groups[j])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('C:/Users/1/Desktop/kaggle/airbnb/temp/6_70_30_lb.csv',index=False)
