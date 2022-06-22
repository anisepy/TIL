from dataloader import load_train
from dataloader import load_test
from utils import load_config
from preprocess import scaling
from preprocess import reduce_mem_usage
from preprocess import preprocess_text
from preprocess import preprocess_text_test
from visualization import visualize_feature_importance
from visualization import get_top_bottom_coef
from visualization import visualize_coefficient

import catboost   
from xgboost.sklearn import XGBRegressor           
from lightgbm.sklearn import LGBMRegressor   
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def main():
    # TODO python argparse package(library)
    # python main.py --config rf.json
    # python main.py --config cb.json
    # python main.py --config xgb.json
    # python main.py --config lgbm.json
    
    # load data
    df, target = load_train()
    test, submission = load_test()
    df = df.sample(n=500000)
    print("success data loading")
    
    # json file option
    cfg = load_config(model_name='cb')
    preprocess_cfg = cfg['preprocess']
    model_cfg = cfg['params']

    # preprocessing
    if preprocess_cfg['text']:
        df = preprocess_text(df)
        test = preprocess_text_test(test)

    reduce_mem_usage(df)
    reduce_mem_usage(test)

    # data split
    X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), df[target],
                                                    test_size=0.1, random_state=42)
    X_test = test

    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    # feature scaling
    if preprocess_cfg['scaling']:
        X_train_sc = scaling(X_train)
        X_val = scaling(X_val)
        X_test = scaling(X_test)

    # training 
    # model = LGBMRegressor()
    # model = LGBMRegressor(**model_cfg)
    # model = XGBRegressor()
    # model = XGBRegressor(**model_cfg)
    model = catboost.CatBoostRegressor()
    # model = catboost.CatBoostRegressor(**model_cfg)
    model.fit(X_train_sc, y_train)
    pred_train = model.predict(X_train_sc)
    pred_val = model.predict(X_val)

    mae_train = mean_absolute_error(y_train, pred_train)
    mae_val = mean_absolute_error(y_val, pred_val)

    # print("Light GBM MAE result,\t train = %.5f, val = %.5f" % (mae_train, mae_val))
    # print("XGBoost MAE result,\t train = %.5f, val = %.5f" % (mae_train, mae_val))
    print("CatBoost MAE result,\t train = %.5f, val = %.5f" % (mae_train, mae_val))


    # test prediction & maiking submission file
    result = model.predict(X_test)
    submission[target] = result
    submission.to_csv("./data/submission.csv", index = False)

    # visualization
    if preprocess_cfg['importance']:
        visualize_feature_importance(model,X_train)
    if preprocess_cfg['coefficient']:
        get_top_bottom_coef(model, X_train)
        visualize_coefficient(model)
    # TODO evaluate


if __name__ == "__main__":
    main()