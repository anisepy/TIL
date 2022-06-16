import warnings

from dataloader import load_data
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# 참고: https://ywkim92.github.io/machine_learning/feature_selection/#sequencial-feature-selection

df, target = load_data()

X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target],
                                                    test_size=0.1, random_state=0)
print(X_train.shape)

model = LogisticRegression()
# forward  : [x1, x2, x3]
# backward : [x1, x2, x3]
sfs = SequentialFeatureSelector(estimator=model,
                                n_features_to_select=5, direction='backward')

result = sfs.fit(train_X, train_y)
selected_features = sfs.support_

print(data.columns[selected_features])