#!/usr/bin/env python
# coding: utf-8

# ### 정리
# - 평균 : np.average()
# - 분산 : np.var() 기댓값에서 떨어진 정도, 편차제곱 총합의 평균
# - 표준편차 : np.std() 데이터가 퍼진정도, 분산의 제곱근
# - 공분산 : 직접 함수 작성, 2개 확률변수의 상관정도, 편차끼리 곱의 총합 평균
# - 상관계수 : np.corrcoef(data1,data2)[0][1] 또는 df1.corrwith(df2)
#   - -1~1까지 상관정도 표현 0 가까울수록 상관도 적음
#   - $ correlation-coefficient = \frac{공분산}{\sqrt{{x분산} \cdot {y분산}}} $
# - 결정계수 : np.corrcoef(data1,data2)[0][1]**2
#   - x로부터 y를 예측할수 있는 정도
#   - 상관계수의 제곱 (상관계수를 양수화)
#   - 수치가 클수록 회귀분석을 통해 예측할수 있는 수치의 정도가 더 정확

# #### 피클
# - pickle은 파이썬에서 사용하는 딕셔너리, 리스트, 클래스 등의 자료형을 변환 없이 그대로 파일로 저장하고 이를 불러올 때 사용하는 모듈이다.
# - raw text 에 있는 내용을 파싱하여 필요한 객체를 구성하고, 그 자체를 바이너리로 저장
# - pickle.dump(객체, 파일) 로 저장하고
# - pickle.load(파일) 로 로딩

# In[9]:


import pandas as pd
df = pd.read_csv("./data/premierleague.csv")
datas = df.values # df의 value들만 array 객체로 지정


# In[10]:


import pickle
## Save pickle
with open("./data/premierleague.pkl", "wb") as f:
    pickle.dump(datas, f)


# In[11]:


## Load pickle
with open("./data/premierleague.pkl", "rb") as f:
    datas = pickle.load(f)


# In[12]:


datas[:3]


# #### 상관계수, 결정계수 실습

# In[14]:


import numpy as np

# 득점
gf = datas[:, 1].astype(np.int)
# 실점
ga = datas[:, 2].astype(np.int)
# 승점
points = datas[:, -1].astype(np.int)


# In[16]:


# 득점과 승점의 상관계수 출력
corr_gf = np.corrcoef(gf, points)[0, 1]
corr_gf


# In[17]:


# 실점과 승점의 상관계수 출력
corr_ga = np.corrcoef(ga, points)[0, 1]
corr_ga


# In[19]:


# 결정계수 : coefficient of determination
# np.round(수,자리수) : 소수점 밑 자리수까지 반올림
# zip() 함수 : 순회가능한 객체(리스트 객체)를 인자로 받아서, 튜플 형태로 반환
deter = {key: np.round(value ** 2, 2) for key, value in zip(["gf", "ga"], [corr_gf, corr_ga])}
deter


# In[ ]:




