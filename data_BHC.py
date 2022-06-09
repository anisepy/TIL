#!/usr/bin/env python
# coding: utf-8

# ## 데이터 불러오기 및 전처리

# In[6]:


# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataframe setting
print("pandas version: ", pd.__version__)
pd.set_option('display.max_row', 30)
pd.set_option('display.max_columns', 100)

# data load
data = pd.read_csv("/Users/krc/Documents/dev/pubgML12/train_V2.csv")
df = data.copy()

df


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df[df.winPlacePerc.isna()]


# In[ ]:


df['matchType'].value_counts()


# In[3]:


# preprocessing
df = df.dropna() # remove missing value
name_list = ['squad-fpp', 'duo', 'solo-fpp', 'squad', 'duo-fpp', 'solo'] # remove custom games' data
df = df[df["matchType"].isin(name_list) == True]
df


# ### matchType이 커스텀 게임인 부류를 행 삭제 하는 이유

# In[7]:


# 한 팀에 팀원이 4명을 초과하는 팀 존재 
t2 = df.loc[:,'groupId'].value_counts().to_frame()
t2[t2.groupId>4]


# In[9]:


# df[df.groupId == '14d6b54cdec6bc'].head() 
# matchId -> 'b30f3d87189aa6', matchType -> normal-squad-fpp
df.loc[df.matchId== 'b30f3d87189aa6','groupId'].value_counts()
# 2팀 밖에 없는데 팀 당 인원이 비정상적 -> 커스텀 게임


# In[11]:


# df[df.groupId == 'b8275198faa03b'].head()
# matchId -> '3e029737889ce9', matchType -> duo-fpp	
df.loc[df.matchId== '3e029737889ce9','groupId'].value_counts()
# api 문제로 보임


# 5명이상인 이유
# 1. 커스텀 게임 - 대부분의 경우
# 2. api문제 : 두 팀을 한 그룹으로 묶어버림
# 3. 드문 확률로 한 팀이 5명 이상으로 플레이 - 버그

# ### 테스트 중

# In[23]:


t1 = df.groupby('matchId')['Id'].transform('count').to_frame()
# t1
# https://steadiness-193.tistory.com/42 - transform함수 설명
plt.figure(figsize=(30,10))
sns.countplot(data = t1, x="Id")
plt.show()


# In[30]:


df.groupby('matchId')['kills'].max()


# In[28]:


t2 = df.groupby('matchId')['kills'].transform('max').to_frame()
# t2
plt.figure(figsize=(30,10))
sns.countplot(data = t2, x="kills")
plt.show()


# In[18]:


t1 = df.groupby('matchId')['Id'].transform('count').to_frame()
plt.figure(figsize=(30,10))
sns.countplot(data = t1, x="Id")
plt.show()


# ### maxPlace 분석

# In[ ]:


plt.figure(figsize=(30,10))
sns.countplot(data = df, x="maxPlace")
plt.show()


# In[14]:


df[df.maxPlace==2]


# 마지막 등수를 나타낸 column. 마지막 등수가 10이하 인 것은 특이하지만 단순 시작 오류 등으로 보이며 정상 게임으로 진행되었다.

# ## points 분석

# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap=sns.color_palette("RdBu", 20))

plt.show()


# ### killPoints & winPoints 분석

# In[ ]:


df['killPoints'].value_counts()


# In[ ]:


df['winPoints'].value_counts()


# In[ ]:


idx_nm = df[(df.killPoints>0)&(df.winPoints>0)]
idx_nm


# In[ ]:


idx_nm[['rankPoints','killPoints','winPoints']].describe()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
sns.scatterplot(data = idx_nm, x="killPoints", y='kills')
plt.xlabel('killPoints',fontsize=10)
plt.ylabel('kills',fontsize=10)

plt.subplot(2,2,2)
sns.scatterplot(data = idx_nm, x="killPoints", y='killStreaks')
plt.xlabel('killPoints',fontsize=10)
plt.ylabel('killStreaks',fontsize=10)

plt.subplot(2,2,3)
sns.scatterplot(data = idx_nm, x="killPoints", y='damageDealt')
plt.xlabel('killPoints',fontsize=10)
plt.ylabel('damageDealt',fontsize=10)

plt.subplot(2,2,4)
sns.scatterplot(data = idx_nm, x="killPoints", y='longestKill')
plt.xlabel('killPoints',fontsize=10)
plt.ylabel('longestKill',fontsize=10)

plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
sns.scatterplot(data = idx_nm, x="winPoints", y='kills')
plt.xlabel('winPoints',fontsize=10)
plt.ylabel('kills',fontsize=10)

plt.subplot(2,2,2)
sns.scatterplot(data = idx_nm, x="winPoints", y='killStreaks')
plt.xlabel('winPoints',fontsize=10)
plt.ylabel('killStreaks',fontsize=10)

plt.subplot(2,2,3)
sns.scatterplot(data = idx_nm, x="winPoints", y='damageDealt')
plt.xlabel('winPoints',fontsize=10)
plt.ylabel('damageDealt',fontsize=10)

plt.subplot(2,2,4)
sns.scatterplot(data = idx_nm, x="winPoints", y='longestKill')
plt.xlabel('winPoints',fontsize=10)
plt.ylabel('longestKill',fontsize=10)

plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
plt.scatter(idx_nm['winPoints'],idx_nm['killPoints'])
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel('winPoints')
plt.ylabel('killPoints')
plt.show()


# In[ ]:


kp = df[(df.killPoints==1000)]


# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(data = kp, x="kills")
plt.show()


# - kill과 연관은 killPoints보다는 winPoints가 많아보인다.
# - killPoints는 1000점 대, winPoints는 1500점대가 가장 기본 점수로 보인다.

# ### rankPoints와 killPoints,winPoints와의 관계 분석

# In[4]:


df['rankPoints'].value_counts()


# In[5]:


idx_nm_1 = df[df.rankPoints>0]
idx_nm_1['rankPoints'].value_counts()


# In[6]:


idx_nm_1[['rankPoints','killPoints','winPoints']].describe()


# In[7]:


plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.scatter(idx_nm_1['rankPoints'],idx_nm_1['kills'])
plt.xlabel('rankPoints')
plt.ylabel('kills')

plt.subplot(2,2,2)
plt.scatter(idx_nm_1['rankPoints'],idx_nm_1['killStreaks'])
plt.xlabel('rankPoints')
plt.ylabel('killStreaks')

plt.subplot(2,2,3)
plt.scatter(idx_nm_1['rankPoints'],idx_nm_1['damageDealt'])
plt.xlabel('rankPoints')
plt.ylabel('damageDealt')

plt.subplot(2,2,4)
plt.scatter(idx_nm_1['rankPoints'],idx_nm_1['longestKill'])
plt.xlabel('rankPoints')
plt.ylabel('longestKill')

plt.show()


# - -1(none),0(none),1500(중간값)
# - 상위 랭커들은 운영적으로 해서 많이 분산된 반면 1500점대 유저들은 전략이 부족하다보니 운이나 실력등이 산재되어 있는듯하다. 
# - 또한 시즌이 바뀌면서 오랜만에 들어오지만 게임을 잘하는 경우는 제대로 rankPoints가 반영이 안되어 있는 듯 하다.
# - 상위 랭커들은 winPlace를 통해 다시 분석할 필요가 있다.

# In[8]:


df[(df.killPoints==0)&(df.rankPoints==-1)]


# In[9]:


df[(df.winPoints==0)&(df.rankPoints==-1)]


# In[10]:


df[(df.killPoints!=0)&(df.rankPoints>0)]


# In[11]:


df[(df.winPoints!=0)&(df.rankPoints>0)]


# killPoints와 winPoints가 있는데 rankPoints가 -1이나 0이 아닌 경우는 없다.
