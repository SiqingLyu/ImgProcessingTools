import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def metric(pred, lab, savepath):
    MAE = np.mean(np.abs(pred - lab))
    RMSE = np.mean(np.power((pred-lab), 2))
    print(f"MAE={MAE}, RMSE={RMSE}")
    np.savetxt(savepath, pred, delimiter=',')


def main(df, savepath=r''):
    x = df.iloc[:, 2:].values
    y_c = df.iloc[:, 1].values
    y_r = df.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y_c, test_size=0.3, random_state=0)
    _, _, y_train_r, y_test_r = train_test_split(x, y_r, test_size=0.3, random_state=0)

    forest_c = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=16, max_depth=3, max_features=370, max_leaf_nodes=2)
    forest_c.fit(x_train, y_train)

    forest_r = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=16, max_depth=3, max_features=370, max_leaf_nodes=2)
    forest_r.fit(x_train, y_train_r)

    res_c = forest_c.predict(x_test)
    res_r = forest_r.predict(x_test)
    metric(res_c, y_test, savepath=savepath + f'Classifier.csv')
    metric(res_r, y_test_r, savepath=savepath + f'Regressor.csv')

    for i in range(3):
        x_tmp = x_train[y_train == i+1]
        y_tmp = y_train_r[y_train == i+1]
        forest_r_tmp = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=16, max_depth=3, max_features=370, max_leaf_nodes=2)
        forest_r_tmp.fit(x_tmp, y_tmp)

        x_test_tmp = x_test[res_c == i+1]
        y_test_tmp = y_test_r[res_c == i+1]
        res_tmp = forest_r_tmp.predict(x_test_tmp)
        metric(res_tmp, y_test_tmp, savepath=savepath + f'predGroup{i+1}.csv')




if __name__ == '__main__':
    csv_path = r'/media/shihaoze/lsq/Bi-Temporal/BiTemporalData.csv'
    df = pd.read_csv(csv_path, header=None)
    print(df.shape)  # (1678899, 482)
    print(df.head(5))

    # '''
    #    0(r) 1(c)     2      3    ...      478      479      480      481
    # 0  2.0  1.0  0.1341  0.1154  ...  15239.0  21775.0  10963.0  10339.0
    # 1  2.0  1.0  0.1864  0.1339  ...  15247.0  22053.0  12265.0  10539.0
    # 2  2.0  1.0  0.2425  0.2363  ...  14583.0  21773.0  12315.0  10873.0
    # 3  2.0  1.0  0.2076  0.1748  ...  14341.0  21217.0  12685.0  10803.0
    # 4  2.0  1.0  0.2238  0.1955  ...  15473.0  20693.0  12249.0  10807.0
    # '''

    main(df, savepath=r'/media/shihaoze/lsq/Bi-Temporal/')