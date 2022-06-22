import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_feature_importance(model, train):
    # tree based ML model visualization
    feature_series = pd.Series(data=model.feature_importances_, index=train.columns )
    feature_series = feature_series.sort_values(ascending=False)

    plt.figure(figsize=(30,12))
    sns.barplot(x= feature_series, y=feature_series.index)
    # return plt.savefig('./data/feature_importance.png')
    return plt.show()

# Linear based ML model visualization
def get_top_bottom_coef(model, train):
    # coef_ 속성을 기반으로 Series 객체를 생성. index는 컬럼명. 
    coef = pd.Series(model.coef_, index=train.columns)
    
    # + 상위 10개 , - 하위 10개 coefficient 추출하여 반환.
    coef_high = coef.sort_values(ascending=False).head(10)
    coef_low = coef.sort_values(ascending=False).tail(10)
    return coef_high, coef_low

def visualize_coefficient(model):
    # 3개 회귀 모델의 시각화를 위해 3개의 컬럼을 가지는 subplot 생성
    # 입력인자로 받은 list객체인 models에서 차례로 model을 추출하여 회귀 계수 시각화. 
    coef_high, coef_low = get_top_bottom_coef(model)
    coef_concat = pd.concat( [coef_high , coef_low] )
    plt.figure(figsize=(10,12))
    sns.barplot(x=coef_concat.values, y=coef_concat.index)
    return plt.show()