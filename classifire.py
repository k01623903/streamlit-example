import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skms
import sklearn.ensemble as skens
import seaborn as sn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle


def main():
    task1()
    # test()

def reduce_data(original):
    reduced = original.copy()
    reduced = reduced.drop(["Evaporation"], axis="columns")  # sehr viele nan values
    reduced = reduced.drop(['Sunshine'], axis='columns')  # sehr viele nan values
    reduced = reduced.drop(['Date'], axis='columns')  # zu viele verschiedene werte, dafür monat
    reduced = reduced.drop(['Location'], axis='columns')  # sehr hohe importance
    reduced = reduced.drop(['month'], axis='columns')  # just for fun
    reduced = reduced.drop(['RainToday'], axis='columns') #just for fun
    reduced = reduced.drop(['Rainfall'], axis='columns') #just for fun
    reduced = reduced.drop(['WindGustDir'], axis='columns') #just for fun
    return reduced


def convert(to_convert):
    converted = pd.DataFrame()
    converters = dict()

    for col, dtype in zip(to_convert.columns, to_convert.dtypes):
        if dtype != np.object_:
            converted[col] = to_convert[col]
        else:
            encoder = OneHotEncoder(sparse=False)
            encoder.fit(to_convert[col].values.reshape(-1, 1))
            converters[col] = encoder
            temp = encoder.transform(to_convert[col].values.reshape(-1, 1))
            for i in range(temp.shape[1]):
                converted[f'{col}_{i}'] = temp[:, i]
    filename = 'converters'
    pickle.dump(converters, open(filename, 'wb'))
    return converted


def task1():
    import time
    task_start = time.time()

    start = 0
    end = 113753
    dfTrainData = pd.read_csv("data/clean_weather_train_data.csv")
    dfTrainData = dfTrainData[start:end]
    dfTrainTarget = pd.read_csv("data/weather_train_label.csv")
    dfTrainTarget = dfTrainTarget[start:end]
    dfTrainTarget = dfTrainTarget.values.flatten()
    dfTrainData = reduce_data(dfTrainData)
    dfTrainData = convert(dfTrainData)
    dfTrainData = dfTrainData.drop(['Unnamed: 0'], axis='columns')
    x_train, x_test, y_train, y_test = skms.train_test_split(dfTrainData, dfTrainTarget, test_size=0.2,
                                                             random_state=np.random.RandomState(98475987))

    estimator = skens.RandomForestClassifier(n_estimators=10, random_state=np.random.RandomState(1635183))
    estimator.fit(x_train, y_train)
    y_hat_train = estimator.predict(x_train)
    y_hat_test = estimator.predict(x_test)

    print('x_train.shape: ', x_train.shape)
    print('x_test.shape: ', x_test.shape)
    print('y_train.shape: ', y_train.shape)
    print('y_test.shape: ', y_test.shape)

    # jetzt kommt immer nein raus, weil nein die besten chancen hat, hohe trefferquoten zu erzielen.
    # vielleicht kommt in den Trainingsdaten einfach zu oft "no" vor.
    # zu viele gleiche Werte durch unser cleaning.
    # Daten encodet, vielleicht Fehlerhaft. Evtl Date Spalte rausschmeißen, weil strings unbearbeitbar sind.
    # Montatsspalte hama eh, jetzt is die Datespalte ned so wichtig.
    # Nur Date hilft nix, weil auch andere spalten Strings enthalten.


    acc_train = accuracy_score(y_train, y_hat_train)
    acc_test = accuracy_score(y_test, y_hat_test)

    print('acc_train: ', acc_train)
    print('acc_test: ', acc_test)

    cm_train = confusion_matrix(y_train, y_hat_train)
    cm_test = confusion_matrix(y_test, y_hat_test)

    # 0 no, 1 yes
    # prediction senkrecht, truth waagrecht

    dfTrainData.to_csv('data/dfTrainData_find_Unknown.csv')
    print(str(dfTrainData.keys()))

    print('cm_train: ')
    print(cm_train)
    print('cm_test: ')
    print(cm_test)
    print(estimator.feature_importances_)
    mi_features_idx = np.argsort(estimator.feature_importances_)
    mi_features_idx = mi_features_idx[-10:]
    mi_features = estimator.feature_importances_[mi_features_idx]
    fig, ax = plt.subplots()
    ax.vlines(
        np.arange(len(mi_features)),
        np.zeros(len(mi_features)),
        mi_features)
    ax.set_xticks(np.arange(len(mi_features)))
    ax.set_xticklabels(dfTrainData.columns[mi_features_idx], rotation=90)
    plt.tight_layout()
    plt.show()

    filename = 'finalized_model.sav'
    pickle.dump(estimator, open(filename, 'wb'))

    task_end = time.time()
    task_diff = task_end - task_start
    print('took ', task_diff)


if __name__ == "__main__":
    main()
