from dataloader import load_train
from dataloader import load_test
from utils import load_config
from preprocess import scaling
from preprocess import reduce_mem_usage
from preprocess import preprocess_text


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression  # 1. Linear Regression
from sklearn.linear_model import Lasso             # 2. Lasso
from sklearn.linear_model import Ridge             # 3. Ridge
from xgboost.sklearn import XGBRegressor           # 4. XGBoost
from lightgbm.sklearn import LGBMRegressor         # 5. LightGBM
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils

def main():
    # TODO python argparse package(library)
    # python main.py --config rf.json
    # python main.py --config svm.json
    # python main.py --config lr.json
    # python main.py --config lgbm.json

    df, target = load_train()
    # test = load_test()

    cfg = load_config(model_name='lgbm')
    preprocess_cfg = cfg['preprocess']
    model_cfg = cfg['params']

    if preprocess_cfg['text']:
        preprocess_text(df)
        # preprocess_text(test)

    reduce_mem_usage(df)
    # reduce_mem_usage(test)

    X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), df[target],
                                                    test_size=0.1, random_state=42)
    # X_test = test
                                                   
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)


    if preprocess_cfg['scaling']:
        X_train = scaling(X_train)
        X_val = scaling(X_val)
        # X_test = scaling(X_test)
        # lab_enc = preprocessing.LabelEncoder()
        # X_train = lab_enc.fit_transform(X_train)
        # X_val = lab_enc.fit_transform(X_val)

    model = LGBMRegressor()
    # model = LGBMRegressor(**model_cfg)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    mae_train = mean_absolute_error(y_train, pred_train)
    mae_val = mean_absolute_error(y_val, pred_val)

    print("LightGBM MAE result,\t train = %.4f, val = %.4f" % (mae_train, mae_val))
    # print("Random Forest MAE result,\t train = %.4f, val = %.4f" % (mae_train, mae_val))
    
    # result = model.predict(X_test)
    # print("---------- RandomForest ---------")
    # print("---------- LightGBM ---------")
    # print('MSE in training: %.4f' % mean_absolute_error(y_test, result))
    # print(" ")
    # print("Done")
    # TODO evaluate


if __name__ == "__main__":
    main()