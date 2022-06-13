#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lightgbm

print(lightgbm.__version__)


# In[2]:


# LightGBM의 파이썬 패키지인 lightgbm에서 LGBMRegressor import
from lightgbm import LGBMRegressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[3]:


from IPython.display import display

pd.options.display.precision = 15
pd.options.display.max_rows = 10000
pd.options.display.max_columns = 10000
pd.options.display.max_colwidth = 10000

pd.set_option('display.max_columns', 100)


# In[4]:


EXCLUDE_COLS = ['Id', 'isnormal','matchType', 'matchId', 'groupId']
# CATEGORICAL_COLS = ['matchId', 'groupId']
TARGET = 'winPlacePerc'
TRAIN_SIZE = 0.9


# In[5]:


# data load
base_path = "/Users/krc/Documents/dev/pubgML12/"
# base_path = "/content/drive/MyDrive/Colab Notebooks/미니프로젝트_pubg"
data = pd.read_csv(base_path +"train_V2.csv")
df = data.copy()

df


# In[6]:


# 메모리 줄이기
def reduce_mem_usage(memory_df,verbose=True):
    numerics = ['int16','int32','int64','float16','float32','float64']
    start_mem = memory_df.memory_usage().sum()/1024**2
    for col in memory_df.columns:
        col_type = memory_df[col].dtypes
        if col_type in numerics:
            c_min = memory_df[col].min()
            c_max = memory_df[col].max()
            if str(col_type)[:3]=='int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    memory_df[col] = memory_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    memory_df[col] = memory_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    memory_df[col] = memory_df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    memory_df[col] = memory_df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    memory_df[col] = memory_df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    memory_df[col] = memory_df[col].astype(np.float32)
                else: 
                    memory_df[col] = memory_df[col].astype(np.float64)
    end_mem = memory_df.memory_usage().sum()/1024**2
    if verbose : print('Meme usage decrease to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100*(start_mem - end_mem)/start_mem))
    return memory_df

reduce_mem_usage(df)


# In[7]:


df = df.dropna()
## start - 박성원
# feature merge
df["total_item"] = df.boosts + df.heals
df["total_distance"] = df.rideDistance + df.walkDistance + df.swimDistance
# boosts
df = df.drop(df[df.boosts>22][df.winPlacePerc<0.2].index)
# total_item
df = df.drop(df[df.winPlacePerc==0][df.total_item>10].index)
# total_distance
df = df.drop(df[df.winPlacePerc==1][df.total_distance==0].index)
df = df.drop(df[df.kills >= 2][df.total_distance == 0].index)
# weaponsAcquired
df = df.drop(df[df.weaponsAcquired>30].index)
df = df.drop(df[df.weaponsAcquired==0][df.winPlacePerc==1].index)
## end - 박성원

## start - 김한길
# 로드디스턴스0 & 로드킬 >0
df=df.drop(index=df[ (df['rideDistance']==0) & (df['roadKills']>0)  ].index)

# 딜이없는데 킬이있는경우
df=df.drop(index=df[ (df['damageDealt']==0) & (df['kills']>0)  ].index)
## end - 김한길

## start - 이채영
# kills, headshotKills, killStreaks, longestKill 이상치 제거 (각 column 별 약 1000개 정도 없어지도록 값을 잡음)
df=df.drop(index=df[ (df['kills']>15) | (df['headshotKills']>7) | (df['killStreaks']>5) | (df['longestKill']>500)  ].index)

## end - 이채영

## start - 김지혜
# DBNOs remove outliers
df.drop(df[df['DBNOs'] > 6].index, inplace= True)

# revives remove outliers
df.drop(df[df['revives'] > 3].index, inplace= True)

# teamkills drop
df=df.drop(['teamKills'],axis=1)
## end - 김지혜

## start - 안희수
df.loc[df.matchType.str.contains('solo'),'solo'] = 1
df.loc[~df.matchType.str.contains('solo'),'solo'] = 0
df.loc[df.matchType.str.contains('duo'),'duo'] = 1
df.loc[~df.matchType.str.contains('duo'),'duo'] = 0
df.loc[df.matchType.str.contains('squad'),'squad'] = 1
df.loc[~df.matchType.str.contains('squad'),'squad'] = 0
df.loc[(df.matchType.str.contains('normal'))|
       (df.matchType.str.contains('crash'))|
       (df.matchType.str.contains('flare')),'event'] = 1
df.loc[(~df.matchType.str.contains('normal'))&
       (~df.matchType.str.contains('crash'))&
       (~df.matchType.str.contains('flare')),'event'] = 0

# row 제거
# remove missing value
# 같은 게임에 참여한 사람 수 컬럼 수 만듦
df['num']=df.groupby('matchId')['Id'].transform('count')
# 한 게임에서 최대 킬수 컬럼
df['max']=df.groupby('matchId')['kills'].transform('max')
# df.loc[df['num']<=df['max'],['num','max']] # 2124 rows

# 최대 킬수가 한 게임 사람 수 보다 많을 수 없음, 행 제거
df=df[df['num']>df['max']]

## 참가 인원에 비해 팀 수가 너무 적은 경우 조정
# 팀 수 의 이상치 열 생성
df.loc[:,'numDuo']=df['num']//2
df.loc[:,'numSquad']=df['num']//4

df.loc[df.duo == 1.0,'Gap'] = abs(df['numDuo']-df['numGroups'])
df.loc[df.squad == 1.0,'Gap'] = abs(df['numSquad']-df['numGroups'])
df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap']=abs(df['num']-df['numGroups'])

# df['Gap'].value_counts().to_frame().sort_index()

# solo나 event에서 numGroups 이상치 탐색
# df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap'].value_counts()
# len(df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0)&(df.Gap>9),'Gap']) # 24907
# print("%.2f" % ((24907/663429)*100)) # 3.75%
# solo나 event인데 Gap이 9보다 큰 경우들 평균값(반올림해서 3)되도록 maxPlace와 numGroup조정
df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0) & (df.Gap > 9),['maxPlace','numGroups']] = df['num']-3
# df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap']=abs(df['num']-df['numGroups'])
# df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap'].value_counts()

# duo에서 numGroups 이상치 탐색
# df.loc[(df.matchType.str.contains('duo')),['Gap']].value_counts()
# Gap이 5이상 차이 나면 groupId에 문제가 있는 것으로 판단, 사실 그 이하도 문제 존재
# df.loc[(df.matchType.str.contains('duo'))&(df.Gap==5.0),['matchId']].value_counts()
# df.loc[df.matchId== '35c26cc0a5212a','groupId'].value_counts() # 한팀에 2명 이상인 팀 들 5팀 이상
# 보통 API가 꼬인 경우로 보인다. -> Gap이 5이상인 경우 가장 많은 값인 1이 되도록 조정한다.
df.loc[(df.matchType.str.contains('duo'))&(df.Gap>=5),['maxPlace','numGroups']] = df['numDuo']-1
# df.loc[df.duo == 1.0,'Gap']=abs(df['numDuo']-df['numGroups'])
# df.loc[(df.matchType.str.contains('duo')),'Gap'].value_counts()

#squad에서 numgroups이상치 탐색
# df.loc[(df.matchType.str.contains('squad')),['Gap']].value_counts()
# Gap이 8이상 차이 나는 것은 groupId에 이상이 있는 것으로 판단, 사실 그 이하도 문제 존재
# Gap이 8이상인 경우 가장 많은 값인 4가 되도록 조정한다.
df.loc[(df.matchType.str.contains('squad'))&(df.Gap>=8),['maxPlace','numGroups']] = df['numSquad']-4
# df.loc[df.squad == 1.0,'Gap']=abs(df['numSquad']-df['numGroups'])
# df.loc[(df.matchType.str.contains('squad')),'Gap'].value_counts()

## maxPlace가 numGroups와 많이 차이나는 것 조정
df.loc[:,'GrpError']=df['maxPlace']-df['numGroups']
# df.loc[df['GrpError'] > df['GrpError'].quantile(0.99),'GrpError'].value_counts() # 7, 8, 9
# df['GrpError'].mean() # 1.445498e+00

# maxPlace와 numGroups가 많이 차이나는 것(7 이상)은 matchId의 오류로 보고 평균값(반올림해서 2)만큼의 차이를 둬서(maxPlace값 - 2) 조정한다.
df.loc[df.GrpError>=7,'numGroups'] = df['maxPlace']-2
# df.loc[:,'GrpError']=df['maxPlace']-df['numGroups']
# df['GrpError'].value_counts()

# 마지막 확인
# df.loc[df.duo == 1.0,'Gap'] = abs(df['numDuo']-df['numGroups'])
# df.loc[df.squad == 1.0,'Gap'] = abs(df['numSquad']-df['numGroups'])
# df.loc[~(df.duo == 1.0)& ~(df.squad == 1.0),'Gap']=abs(df['num']-df['numGroups'])

# df.loc[df.event ==0.0,'Gap'].value_counts().to_frame().sort_index()
# Gap = 8.0 에서 20명이 존재하는데 이는 matchType이 crash 여서 무시한다.

# 만든열 드랍
df=df.drop(['num', 'max', 'numDuo','numSquad','Gap','GrpError'],axis=1)
## end - 안희수
df.isna().sum()


# In[8]:


# 파생 변수 생성
df['num']=df.groupby('matchId')['Id'].transform('count')
df['healthitems'] = df['heals'] + df['boosts']
df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
df["skill"] = df["headshotKills"] + df["roadKills"]

df['Distance_over_heals'] = df['totalDistance']/df['healthitems']
df['Distance_over_weapon'] = df['totalDistance']/df['weaponsAcquired']

# maxGrp: 그룹 별로 능력치는 다르지만 같은 그룹이면 같은 등수를 나타내는 것이기 때문에 가장 잘하는 사람 기준으로 해서 보기 위한 column
df['maxGrp_kills'] = df.groupby('groupId')['kills'].transform('max')
df['maxGrp_totalDistance'] = df.groupby('groupId')['totalDistance'].transform('max')

# sniper : 얻은 무기 수 대비 kill 거리의 비율
df['sniper'] = df['longestKill']/100*df['weaponsAcquired']

# kill 기준
# 킬 대비 연속킬 비율
df['killStreakRate'] = df['killStreaks'] / df['kills']
df['headshotrate'] = df['kills']/df['headshotKills']
df['Distance_over_kills'] = df['totalDistance'] / df['kills']
df['skillRate'] = df['skill'] / df['kills']

# 킬수/매치인원 비율 피쳐
df['kill_per_matchnum']= df['kills']  / df['num']

# kill수가 0이어서 inf가 나오는 값들 0으로 대체
df[df == np.Inf] = np.NaN
df[df == np.NINF] = np.NaN
df.fillna(0, inplace=True)

# 이들의 상관관계 확인
derived_corr = df[['Distance_over_heals','Distance_over_weapon','maxGrp_kills','maxGrp_totalDistance',
                    'sniper', 'killStreakRate','headshotrate', 'Distance_over_kills',
                    'skillRate','winPlacePerc','kill_per_matchnum']].corr()
derived_corr
# maxGrp_killPlace가 0.8이상으로 높게 나옴


# In[9]:


df = df.drop(['Id','matchId','groupId','matchType'], axis = 1)


# In[21]:


# from sklearn.preprocessing import MinMaxScaler

# features = df.drop(['Id','matchId','groupId','matchType'], axis = 1)
# features_name = list(df.columns)
# features_name.remove("Id")
# features_name.remove("matchId")
# features_name.remove("groupId")
# features_name.remove("matchType")

# scaler = MinMaxScaler()
# scaler.fit(features)
# scaled = scaler.transform(features)
# train = pd.DataFrame(data=scaled, columns=features_name)
# train

# 안한게 결과가 더 좋음


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(df.drop(TARGET, axis=1), df[TARGET],
                                                    test_size=1-TRAIN_SIZE, random_state=0)


# In[11]:


# n_estimators는 250 설정, objective = MAE
lgbm_wrapper = LGBMRegressor(objective='mae', n_estimators=250,  
                     learning_rate=0.3, num_leaves=200, 
                     n_jobs=-1,  random_state=156, verbose=0)


# LightGBM도 XGBoost와 동일하게 조기 중단 수행 가능. 
lgbm_wrapper.fit(X_train, y_train,
         eval_set=[(X_test, y_test)], 
         eval_metric='mae', early_stopping_rounds=10, 
         verbose=0)
preds = lgbm_wrapper.predict(X_test)


# In[12]:


# 모델과 학습/테스트 데이터 셋을 입력하면 성능 평가 수치를 반환
def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1 :
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print('###',model.__class__.__name__,'###')
    evaluate_regr(y_test, pred)


# In[13]:


print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, preds)))


# In[14]:


# plot_importance( )를 이용하여 feature 중요도 시각화
from lightgbm import plot_importance
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(lgbm_wrapper, ax=ax)


# In[16]:




