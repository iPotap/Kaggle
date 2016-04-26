import numpy as np
import pandas as pd
import csv
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
import sys


def sessions_stats(group):
    group.fillna(0, inplace=True)

    if group.count() == 0:
        return {'user_id': group.name,
                'sessions_total_duration': group.max() - group.min(),
                'average_action_duration': 0,
                'actions_total_count': 0}
    else:
        return {'user_id': group.name,
                'sessions_total_duration': group.max() - group.min(),
                'average_action_duration': (group.max() - group.min()) / group.count(),
                'actions_total_count': group.count()}
                
                
                
#Loading data
df_train = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/train_users_2.csv')
df_test = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/test_users.csv')
df_countries = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/countries.csv')
sessions = pd.read_csv('C:/Users/1/Desktop/kaggle/airbnb/data/sessions.csv')

print (sessions.head(10))

#sessions['action'] = sessions['action'].fillna('999')
#data roll-up
#secs_elapsed
grpby = sessions.groupby(['user_id'])['secs_elapsed'].sum().reset_index()
grpby.columns = ['user_id','secs_elapsed']

df_sstats = sessions['secs_elapsed'].groupby(sessions['user_id']).apply(sessions_stats).unstack()

print (df_sstats.head())
        
#translate count
sessions['secs_elapsed_translate1'] = np.where(sessions['action']=='ajax_google_translate', sessions['secs_elapsed'], 0)
sessions['secs_elapsed_translate2'] = np.where(sessions['action']=='ajax_google_translate_reviews', sessions['secs_elapsed'], 0)
sessions['secs_elapsed_translate3'] = np.where(sessions['action']=='ajax_google_translate_description', sessions['secs_elapsed'], 0)
sessions['secs_elapsed_translate4'] = np.where(sessions['action']=='spoken_languages', sessions['secs_elapsed'], 0)
sessions['secs_elapsed_translate'] = sessions['secs_elapsed_translate1'] + sessions['secs_elapsed_translate2'] + sessions['secs_elapsed_translate3'] + sessions['secs_elapsed_translate4']

sessions = sessions.drop(['secs_elapsed_translate1', 'secs_elapsed_translate2', 'secs_elapsed_translate3', 'secs_elapsed_translate4'], axis=1)
translate = sessions.groupby(['user_id'])['secs_elapsed_translate'].sum().reset_index()
translate.columns = ['user_id','secs_elapsed_translate']

"""
#unique
uq = sessions['action_detail'].unique()

with open("C:/Users/1/Desktop/kaggle/airbnb/data/uq3.csv", "w") as toWrite:
        writer = csv.writer(toWrite, delimiter=",")
        for i in range(sessions.shape[0]): #assuming you have a 2D numpy array
            writer.writerow(sessioins[i,:])
"""
# agg = grpby['secs_elapsed'].agg({'time_spent' : np.sum})

#action
#print(sessions.action_type.value_counts())
#print(sessions.groupby(['action_type'])['user_id'].nunique().reset_index())
action_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
action_type = action_type.drop(['booking_response'],axis=1)


#print(sessions.groupby(['device_type'])['user_id'].nunique().reset_index())
#print(sessions.groupby(['user_id'])['device_type'].nunique().reset_index())
device_type = pd.pivot_table(sessions, index = ['user_id'],columns = ['device_type'],values = 'action',aggfunc=len,fill_value=0).reset_index()
device_type = device_type.drop(['Blackberry','Opera Phone','iPodtouch','Windows Phone'],axis=1)
#device_type = device_type.replace(device_type.iloc[:,1:]>0,1)


action_detail = pd.pivot_table(sessions, index = ['user_id'],columns = ['action_detail'],values = 'action',aggfunc=len,fill_value=0).reset_index()


sessions_data = pd.merge(action_type,device_type,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,grpby,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,translate,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,action_detail,on='user_id',how='inner')
sessions_data = pd.merge(sessions_data,df_sstats,on='user_id',how='inner')

sessions_data['secs_elapsed_log'] = np.log(sessions_data['secs_elapsed']+1)
sessions_data['secs_elapsed_translate_log'] = np.log(sessions_data['secs_elapsed_translate']+1)
sessions_data['average_action_duration'] = sessions_data['average_action_duration'].astype(float)
sessions_data['sessions_total_duration'] = sessions_data['sessions_total_duration'].astype(float)
sessions_data['average_action_duration_log'] = np.log(sessions_data['average_action_duration']+1)
sessions_data['sessions_total_duration_log'] = np.log(sessions_data['sessions_total_duration']+1)

#sessions_data['secs_elapsed_cancel'] = np.log(sessions_data['secs_elapsed_cancel']+1)
#sessions_data['secs_elapsed_cancel'] = np.where(sessions_data['secs_elapsed_cancel']>0, 1, 0)
#sessions_data['secs_elapsed_translate'] = np.where(sessions_data['secs_elapsed_translate']>0, 1, 0)

df_train_sessions = pd.merge(df_train, sessions_data,left_on='id',right_on='user_id',how='left')
df_test_sessions = pd.merge(df_test, sessions_data,left_on='id',right_on='user_id',how='left')

df_train_sessions.to_csv('C:/Users/1/Desktop/kaggle/airbnb/data/train_users_sessions7.csv')
df_test_sessions.to_csv('C:/Users/1/Desktop/kaggle/airbnb/data/test_users_sessions7.csv')