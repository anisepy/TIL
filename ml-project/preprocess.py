import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import utils

def scaling(data):
    scaler = MinMaxScaler()

    if type(data) == pd.DataFrame:
        new_data = scaler.fit_transform(data.values)
    else:
        new_data = scaler.fit_transform(data)

    return new_data


def preprocess_text(data):
    data = data.dropna()
    data["total_item"] = data.boosts + data.heals
    data["total_distance"] = data.rideDistance + data.walkDistance + data.swimDistance
    # boosts
    data = data.drop(data[data.boosts>22][data.winPlacePerc<0.2].index)
    # total_item
    data = data.drop(data[data.winPlacePerc==0][data.total_item>10].index)
    # total_distance
    data = data.drop(data[data.winPlacePerc==1][data.total_distance==0].index)
    data = data.drop(data[data.kills >= 2][data.total_distance == 0].index)
    # weaponsAcquired
    data = data.drop(data[data.weaponsAcquired>30].index)
    data = data.drop(data[data.weaponsAcquired==0][data.winPlacePerc==1].index)

    # 로드디스턴스0 & 로드킬 >0
    data=data.drop(index=data[ (data['rideDistance']==0) & (data['roadKills']>0)  ].index)

    # 딜이없는데 킬이있는경우
    data=data.drop(index=data[ (data['damageDealt']==0) & (data['kills']>0)  ].index)

    # kills, headshotKills, killStreaks, longestKill 이상치 제거 (각 column 별 약 1000개 정도 없어지도록 값을 잡음)
    data=data.drop(index=data[ (data['kills']>15) | (data['headshotKills']>7) | (data['killStreaks']>5) | (data['longestKill']>500)  ].index)

    # DBNOs remove outliers
    data.drop(data[data['DBNOs'] > 6].index, inplace= True)

    # revives remove outliers
    data.drop(data[data['revives'] > 3].index, inplace= True)

    # teamkills drop
    # data=data.drop(['teamKills'],axis=1)

    # Decompose matchType
    data.loc[data.matchType.str.contains('solo'),'solo'] = 1
    data.loc[~data.matchType.str.contains('solo'),'solo'] = 0
    data.loc[data.matchType.str.contains('duo'),'duo'] = 1
    data.loc[~data.matchType.str.contains('duo'),'duo'] = 0
    data.loc[data.matchType.str.contains('squad'),'squad'] = 1
    data.loc[~data.matchType.str.contains('squad'),'squad'] = 0
    data.loc[(data.matchType.str.contains('normal'))|
        (data.matchType.str.contains('crash'))|
        (data.matchType.str.contains('flare')),'event'] = 1
    data.loc[(~data.matchType.str.contains('normal'))&
        (~data.matchType.str.contains('crash'))&
        (~data.matchType.str.contains('flare')),'event'] = 0

    # 최대 킬수가 한 게임 사람 수 보다 많을 수 없음, 행 제거
    data['num']=data.groupby('matchId')['Id'].transform('count')
    data['max']=data.groupby('matchId')['kills'].transform('max')
    data=data[data['num']>data['max']]

    ## 참가 인원에 비해 팀 수가 너무 적은 경우 조정
    data.loc[:,'numDuo']=data['num']//2
    data.loc[:,'numSquad']=data['num']//4

    data.loc[data.duo == 1.0,'Gap'] = abs(data['numDuo']-data['numGroups'])
    data.loc[data.squad == 1.0,'Gap'] = abs(data['numSquad']-data['numGroups'])
    data.loc[~(data.duo == 1.0)& ~(data.squad == 1.0),'Gap']=abs(data['num']-data['numGroups'])


    # solo나 event인데 Gap이 9보다 큰 경우들 평균값(반올림해서 3)되도록 maxPlace와 numGroup조정
    data.loc[~(data.duo == 1.0)& ~(data.squad == 1.0) & (data.Gap > 9),['maxPlace','numGroups']] = data['num']-3

    # duo인데 Gap이 5이상인 경우 가장 많은 값인 1이 되도록 조정한다.
    data.loc[(data.matchType.str.contains('duo'))&(data.Gap>=5),['maxPlace','numGroups']] = data['numDuo']-1

    # squad인데 Gap이 8이상인 경우 가장 많은 값인 4가 되도록 조정한다.
    data.loc[(data.matchType.str.contains('squad'))&(data.Gap>=8),['maxPlace','numGroups']] = data['numSquad']-4

    ## maxPlace가 numGroups와 많이 차이나는 것 조정
    data.loc[:,'GrpError']=data['maxPlace']-data['numGroups']

    # maxPlace와 numGroups가 많이 차이나는 것(7 이상)은 matchId의 오류로 보고 평균값(반올림해서 2)만큼의 차이를 둬서(maxPlace값 - 2) 조정한다.
    data.loc[data.GrpError>=7,'numGroups'] = data['maxPlace']-2


    data['Distance_over_heals'] = data['total_distance']/data['total_item']
    data['Distance_over_weapon'] = data['total_distance']/data['weaponsAcquired']

    # maxGrp: 그룹 별로 능력치는 다르지만 같은 그룹이면 같은 등수를 나타내는 것이기 때문에 가장 잘하는 사람 기준으로 해서 보기 위한 column
    data['maxGrp_kills'] = data.groupby('groupId')['kills'].transform('max')
    data['maxGrp_totalDistance'] = data.groupby('groupId')['total_distance'].transform('max')

    # sniper : 얻은 무기 수 대비 kill 거리의 비율
    data['sniper'] = data['longestKill']/100*data['weaponsAcquired']

    # kill 기준
    # 킬 대비 연속킬 비율
    data['killStreakRate'] = data['killStreaks'] / data['kills']
    data['headshotrate'] = data['kills']/data['headshotKills']
    data['Distance_over_kills'] = data['total_distance'] / data['kills']
    data["skill"] = data["headshotKills"] + data["roadKills"]
    data['skillRate'] = data['skill'] / data['kills']

    # 킬수/매치인원 비율 피쳐
    data['kill_per_matchnum']= data['kills']  / data['num']

    # 새롭게 killplace만들기
    data['killPlace']=data.groupby('matchId')['kills'].transform('rank', ascending=False)

    # map size
    data.loc[data.matchDuration<1625,'map'] = 0
    data.loc[data.matchDuration>=1625,'map'] = 1

    # groupby mean, max
    basicFeatures = ['assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace',
                    'kills','killStreaks','longestKill','matchDuration','numGroups','revives',
                    'rideDistance','roadKills','swimDistance','vehicleDestroys','walkDistance','weaponsAcquired']
    mean =[]
    max = []
    per = []
    for i in basicFeatures:
        data[i+'_mean']=data.groupby('groupId')[i].transform('mean')
        mean.append(i+'_mean')
        data[i+'_per']=data[i] / ( data.groupby('matchId')[i].transform('mean') )
        per.append(i+'_per')

    for i in basicFeatures:
        data[i+'_max']=data.groupby('groupId')[i].transform('max')
        max.append(i+'_max')


    # new features 
    for i in mean :
        data[i+'_place']=data.groupby('matchId')[i].transform('rank', ascending=False)

    for i in max :
        data[i+'_place']=data.groupby('matchId')[i].transform('rank', ascending=False)

    # data['assists_per']= data['assists']  /  ( data.groupby('matchId')['Id'].transform('count') )
    # data['boosts_value']= data['boosts']  * ( data.groupby('matchId')['Id'].transform('count') )
    # data['damageDealt_per']= data['damageDealt']  / ( data.groupby('matchId')['damageDealt'].transform('sum') )
    # data['damageDealt_per'] = data['damageDealt_per'].fillna(0)
    # data['DBNOs_per']= data['DBNOs']  / ( data.groupby('matchId')['Id'].transform('count') )
    # data['headshot_rate'] = data['headshotKills'] / data['kills']
    # data['headshot_rate'] = data['headshot_rate'].fillna(0)
    # data['heals_per']= data['heals']  * ( data.groupby('matchId')['Id'].transform('count') )
    # data['killPlace_per']= data['killPlace']  / ( data.groupby('matchId')['Id'].transform('count') )
    # data['kills_per']= data['kills']  / ( data.groupby('matchId')['Id'].transform('count') )
    # data['killStreaks_per']= data['killStreaks']  / ( data.groupby('matchId')['Id'].transform('count') )
    # data['revives_per']= data['revives']  / data.groupby('groupId')['Id'].transform('count')
    # data['rideDistance_value']= data['rideDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    # data['roadKills_per']= data['roadKills']  / ( data.groupby('matchId')['Id'].transform('count') )
    # data['swimDistance_value']= data['swimDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    # data['teamKills_per']= data['teamKills']  / data.groupby('groupId')['Id'].transform('count')
    # data['vehicleDestroys_per']= data['vehicleDestroys']  / ( data['map']+1 )
    # data['walkDistance_value']= data['walkDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    # data['weaponsAcquired_per']= data['weaponsAcquired']  * ( data.groupby('matchId')['Id'].transform('count') )
    # for i in ['assists_per', 'boosts_value', 'damageDealt_per',
    #     'DBNOs_per', 'headshot_rate', 'heals_per', 'killPlace_per', 'kills_per',
    #     'killStreaks_per', 'revives_per', 'rideDistance_value', 'roadKills_per', 'swimDistance_value', 'teamKills_per', 'vehicleDestroys_per',
    #     'walkDistance_value', 'weaponsAcquired_per']:
    #     data[i+'_mean']=data.groupby('groupId')[i].transform('mean')
    #     per.append(i+'_mean')
    # for i in per :
    #     data[i+'_place']=data.groupby('matchId')[i].transform('rank', ascending=False)
    data['killPlace_per']= data['killPlace']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['kills_per']= data['kills']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['killStreaks_value']= data['killStreaks']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['roadKills_value']= data['roadKills']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['rideDistance_value']= data['rideDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['swimDistance_value']= data['swimDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['walkDistance_value']= data['walkDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['weaponsAcquired_value']= data['weaponsAcquired']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['revives_per']= data['revives']  / data.groupby('groupId')['Id'].transform('count')
    data['teamKills_per']= data['teamKills']/ data.groupby('groupId')['Id'].transform('count')
    data['vehicleDestroys_per']= data['vehicleDestroys']  / ( data['map']+1 )

    per_mean =[]
    for i in per :
        data[i+'_mean']=data.groupby('groupId')[i].transform('mean')
        per_mean.append(i+'_mean')
    for i in per_mean :
        data[i+'_place']=data.groupby('matchId')[i].transform('rank', ascending=False)

    # value 값 0,1은 categorical data입니다!
    index_list = data[data.boosts>=8][data.boosts<=10][data.damageDealt>=302][data.damageDealt<=779][data.longestKill>=53][data.longestKill<=239].index

    for i in index_list:
        data.loc[i,"crawling"] = 1
    data["crawling"].fillna(0)

    # kill수가 0이어서 inf가 나오는 값들 0으로 대체
    data[data == np.Inf] = np.NaN
    data[data == np.NINF] = np.NaN
    data.fillna(0, inplace=True)
    # 필요 없는 열 드랍
    data=data.drop(['num', 'max', 'numDuo','numSquad','Gap','GrpError',
                    'matchType','Id','matchId','groupId','rankPoints',
                    'winPoints','killPoints','teamKills','killPlace'],axis=1)
    data=data.drop(['assists', 'boosts', 'headshotKills','weaponsAcquired','headshotKills_mean',
                'revives_mean','swimDistance_mean_place','assists_max','DBNOs_max',
                'headshotKills_max','headshotKills_max_place','revives_max_place',
                'roadKills_max_place','swimDistance_value',
                'revives_per_mean_place','revives', 'swimDistance_max',
                'roadKills_per','assists_mean','headshotKills_mean_place',
                'weaponsAcquired_max','assists_per_mean_place'], 
                axis=1)
    return data

def preprocess_text_test(data):
    data["total_item"] = data.boosts + data.heals
    data["total_distance"] = data.rideDistance + data.walkDistance + data.swimDistance
    # # boosts
    # data = data.drop(data[data.boosts>22][data.winPlacePerc<0.2].index)
    # # total_item
    # data = data.drop(data[data.winPlacePerc==0][data.total_item>10].index)
    # # total_distance
    # data = data.drop(data[data.winPlacePerc==1][data.total_distance==0].index)
    # data = data.drop(data[data.kills >= 2][data.total_distance == 0].index)
    # # weaponsAcquired
    # data = data.drop(data[data.weaponsAcquired>30].index)
    # data = data.drop(data[data.weaponsAcquired==0][data.winPlacePerc==1].index)

    # # 로드디스턴스0 & 로드킬 >0
    # data=data.drop(index=data[ (data['rideDistance']==0) & (data['roadKills']>0)  ].index)

    # # 딜이없는데 킬이있는경우
    # data=data.drop(index=data[ (data['damageDealt']==0) & (data['kills']>0)  ].index)

    # # kills, headshotKills, killStreaks, longestKill 이상치 제거 (각 column 별 약 1000개 정도 없어지도록 값을 잡음)
    # data=data.drop(index=data[ (data['kills']>15) | (data['headshotKills']>7) | (data['killStreaks']>5) | (data['longestKill']>500)  ].index)

    # # DBNOs remove outliers
    # data.drop(data[data['DBNOs'] > 6].index, inplace= True)

    # # revives remove outliers
    # data.drop(data[data['revives'] > 3].index, inplace= True)

    # # teamkills drop
    # data=data.drop(['teamKills'],axis=1)

    # Decompose matchType
    data.loc[data.matchType.str.contains('solo'),'solo'] = 1
    data.loc[~data.matchType.str.contains('solo'),'solo'] = 0
    data.loc[data.matchType.str.contains('duo'),'duo'] = 1
    data.loc[~data.matchType.str.contains('duo'),'duo'] = 0
    data.loc[data.matchType.str.contains('squad'),'squad'] = 1
    data.loc[~data.matchType.str.contains('squad'),'squad'] = 0
    data.loc[(data.matchType.str.contains('normal'))|
        (data.matchType.str.contains('crash'))|
        (data.matchType.str.contains('flare')),'event'] = 1
    data.loc[(~data.matchType.str.contains('normal'))&
        (~data.matchType.str.contains('crash'))&
        (~data.matchType.str.contains('flare')),'event'] = 0

    # 최대 킬수가 한 게임 사람 수 보다 많을 수 없음, 행 제거
    data['num']=data.groupby('matchId')['Id'].transform('count')
    data['max']=data.groupby('matchId')['kills'].transform('max')
    # data=data[data['num']>data['max']]

    ## 참가 인원에 비해 팀 수가 너무 적은 경우 조정
    data.loc[:,'numDuo']=data['num']//2
    data.loc[:,'numSquad']=data['num']//4

    data.loc[data.duo == 1.0,'Gap'] = abs(data['numDuo']-data['numGroups'])
    data.loc[data.squad == 1.0,'Gap'] = abs(data['numSquad']-data['numGroups'])
    data.loc[~(data.duo == 1.0)& ~(data.squad == 1.0),'Gap']=abs(data['num']-data['numGroups'])


    # solo나 event인데 Gap이 9보다 큰 경우들 평균값(반올림해서 3)되도록 maxPlace와 numGroup조정
    data.loc[~(data.duo == 1.0)& ~(data.squad == 1.0) & (data.Gap > 9),['maxPlace','numGroups']] = data['num']-3

    # duo인데 Gap이 5이상인 경우 가장 많은 값인 1이 되도록 조정한다.
    data.loc[(data.matchType.str.contains('duo'))&(data.Gap>=5),['maxPlace','numGroups']] = data['numDuo']-1

    # squad인데 Gap이 8이상인 경우 가장 많은 값인 4가 되도록 조정한다.
    data.loc[(data.matchType.str.contains('squad'))&(data.Gap>=8),['maxPlace','numGroups']] = data['numSquad']-4

    ## maxPlace가 numGroups와 많이 차이나는 것 조정
    data.loc[:,'GrpError']=data['maxPlace']-data['numGroups']

    # maxPlace와 numGroups가 많이 차이나는 것(7 이상)은 matchId의 오류로 보고 평균값(반올림해서 2)만큼의 차이를 둬서(maxPlace값 - 2) 조정한다.
    data.loc[data.GrpError>=7,'numGroups'] = data['maxPlace']-2


    data['Distance_over_heals'] = data['total_distance']/data['total_item']
    data['Distance_over_weapon'] = data['total_distance']/data['weaponsAcquired']

    # maxGrp: 그룹 별로 능력치는 다르지만 같은 그룹이면 같은 등수를 나타내는 것이기 때문에 가장 잘하는 사람 기준으로 해서 보기 위한 column
    data['maxGrp_kills'] = data.groupby('groupId')['kills'].transform('max')
    data['maxGrp_totalDistance'] = data.groupby('groupId')['total_distance'].transform('max')

    # sniper : 얻은 무기 수 대비 kill 거리의 비율
    data['sniper'] = data['longestKill']/100*data['weaponsAcquired']

    # kill 기준
    # 킬 대비 연속킬 비율
    data['killStreakRate'] = data['killStreaks'] / data['kills']
    data['headshotrate'] = data['kills']/data['headshotKills']
    data['Distance_over_kills'] = data['total_distance'] / data['kills']
    data["skill"] = data["headshotKills"] + data["roadKills"]
    data['skillRate'] = data['skill'] / data['kills']

    # 킬수/매치인원 비율 피쳐
    data['kill_per_matchnum']= data['kills']  / data['num']

    # 새롭게 killplace만들기
    data['killPlace']=data.groupby('matchId')['kills'].transform('rank', ascending=False)

    # map size
    data.loc[data.matchDuration<1625,'map'] = 0
    data.loc[data.matchDuration>=1625,'map'] = 1

    # groupby mean, max
    basicFeatures = ['assists','boosts','damageDealt','DBNOs','headshotKills','heals','killPlace',
                    'kills','killStreaks','longestKill','matchDuration','numGroups','revives',
                    'rideDistance','roadKills','swimDistance','vehicleDestroys','walkDistance','weaponsAcquired']
    mean =[]
    max = []
    per = []
    for i in basicFeatures:
        data[i+'_mean']=data.groupby('groupId')[i].transform('mean')
        mean.append(i+'_mean')
        data[i+'_per']=data[i] / ( data.groupby('matchId')[i].transform('mean') )
        per.append(i+'_per')

    for i in basicFeatures:
        data[i+'_max']=data.groupby('groupId')[i].transform('max')
        max.append(i+'_max')


    # new features 
    for i in mean :
        data[i+'_place']=data.groupby('matchId')[i].transform('rank', ascending=False)

    for i in max :
        data[i+'_place']=data.groupby('matchId')[i].transform('rank', ascending=False)

    data['killPlace_per']= data['killPlace']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['kills_per']= data['kills']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['killStreaks_value']= data['killStreaks']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['roadKills_value']= data['roadKills']  / ( data.groupby('matchId')['Id'].transform('count') )
    data['rideDistance_value']= data['rideDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['swimDistance_value']= data['swimDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['walkDistance_value']= data['walkDistance']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['weaponsAcquired_value']= data['weaponsAcquired']  * ( data.groupby('matchId')['Id'].transform('count') )
    data['revives_per']= data['revives']  / data.groupby('groupId')['Id'].transform('count')
    data['teamKills_per']= data['teamKills']/ data.groupby('groupId')['Id'].transform('count')
    data['vehicleDestroys_per']= data['vehicleDestroys']  / ( data['map']+1 )

    per_mean =[]
    for i in per :
        data[i+'_mean']=data.groupby('groupId')[i].transform('mean')
        per_mean.append(i+'_mean')
    for i in per_mean :
        data[i+'_place']=data.groupby('matchId')[i].transform('rank', ascending=False)

    # value 값 0,1은 categorical data입니다!
    index_list = data[data.boosts>=8][data.boosts<=10][data.damageDealt>=302][data.damageDealt<=779][data.longestKill>=53][data.longestKill<=239].index

    for i in index_list:
        data.loc[i,"crawling"] = 1
    data["crawling"].fillna(0)

    # kill수가 0이어서 inf가 나오는 값들 0으로 대체
    data[data == np.Inf] = np.NaN
    data[data == np.NINF] = np.NaN
    data.fillna(0, inplace=True)
    # 필요 없는 열 드랍
    data=data.drop(['num', 'max', 'numDuo','numSquad','Gap','GrpError',
                    'matchType','Id','matchId','groupId','rankPoints',
                    'winPoints','killPoints','teamKills','killPlace'],axis=1)
    data=data.drop(['assists', 'boosts', 'headshotKills','weaponsAcquired','headshotKills_mean',
                'revives_mean','swimDistance_mean_place','assists_max','DBNOs_max',
                'headshotKills_max','headshotKills_max_place','revives_max_place',
                'roadKills_max_place','swimDistance_value',
                'revives_per_mean_place','revives', 'swimDistance_max',
                'roadKills_per','assists_mean','headshotKills_mean_place',
                'weaponsAcquired_max','assists_per_mean_place'], 
                axis=1)
    return data

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

