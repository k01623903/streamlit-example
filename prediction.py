import statistics
from sklearn import impute
import pandas as pd
# import plotly.graph_objs as go
import numpy as np
import random
import pickle
import classifire


def main():
    task1()
    #test()

def predict(x_test):
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y_test = loaded_model.predict(x_test)
    return y_test

def reduce_data(original):
    to_return = classifire.reduce_data(original)
    return to_return

def convert(to_convert):
    converted = pd.DataFrame()
    filename = 'converters'
    encoder_file = pickle.load(open(filename, 'rb'))

    for col, dtype in zip(to_convert.columns, to_convert.dtypes):
        if dtype != np.object_:
            converted[col] = to_convert[col]
        else:
            encoder = encoder_file[col]
            encoder.transform(to_convert[col].values.reshape(-1, 1))
            temp = encoder.transform(to_convert[col].values.reshape(-1, 1))
            for i in range(temp.shape[1]):
                converted[f'{col}_{i}'] = temp[:, i]
    return converted



def task1():
    dfTestData = pd.read_csv("data/clean_weather_test_data.csv")
    dfTestData = reduce_data(dfTestData)
    dfTestData = convert(dfTestData)
    dfTestData = dfTestData.drop(['Unnamed: 0'], axis='columns')
    prediction = predict(dfTestData)
    df_prediction = pd.DataFrame()
    df_prediction["prediction"] = prediction
    #df_prediction = df_prediction['prediction']
    df_prediction.to_csv("data/prediction.csv", index=False, header=False)



if __name__ == "__main__":
    main()
