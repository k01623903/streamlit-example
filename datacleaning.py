import statistics
from sklearn import impute
import pandas as pd
# import plotly.graph_objs as go
import numpy as np
import random


def main():
    task1()

def add_month_column(original):
    result_df = original.copy()

    conditions = [
        result_df['Date'].str.contains('-01-'),
        result_df['Date'].str.contains('-02-'),
        result_df['Date'].str.contains('-03-'),
        result_df['Date'].str.contains('-04-'),
        result_df['Date'].str.contains('-05-'),
        result_df['Date'].str.contains('-06-'),
        result_df['Date'].str.contains('-07-'),
        result_df['Date'].str.contains('-08-'),
        result_df['Date'].str.contains('-09-'),
        result_df['Date'].str.contains('-10-'),
        result_df['Date'].str.contains('-11-'),
        result_df['Date'].str.contains('-12-')
    ]
    months_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    result_df['month'] = np.select(conditions, months_numbers)
    return result_df

def create_lookup_table(original):

    # Lookup Table für die einzusetzenden Avg und Mod Werte.
    group_columns = ['Location', 'month']
    categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    numerical_columns = []
    correct_windDir = ['WNW', 'E', 'NE', 'SW', 'W', 'S', 'NW', 'NNW', 'SSE', 'SE', 'SSW', 'WSW', 'N', 'ESE', 'NNE', 'ENE']
    correct_rainToday = ['Yes', 'No']

    for col in original.keys():
        if original[col].dtypes == "float64":
            numerical_columns.append(col)

    am = original[group_columns + numerical_columns].groupby(group_columns, as_index=False).agg(np.nanmean)
    avg_month = pd.DataFrame(am)

    mm = original[group_columns + categorical_columns].groupby(group_columns, as_index=False).agg(statistics.mode)
    mm[{'WindGustDir', 'WindDir9am', 'WindDir3pm'}].replace(to_replace=np.nan, value=random.choice(correct_windDir))
    mm['RainToday'].replace(to_replace=np.nan, value=random.choice(correct_rainToday))
    mode_month = pd.DataFrame(mm)

    df_avg_mod = pd.merge(
        left=avg_month,
        right=mode_month,
        how='inner',
        left_on=group_columns,
        right_on=group_columns
    )
    df_avg_mod.set_index(group_columns, inplace=True)
    df_avg_mod.to_csv('data/AvgModLookup.csv')
    return df_avg_mod


def clean_data_frame(dirty, lookup):
    clean = dirty.copy()
    df_avg_mod = lookup

    group_columns = ['Location', 'month']
    categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    numerical_columns = []

    for col in clean.keys():
        if clean[col].dtypes == "float64":
            numerical_columns.append(col)

    import time
    t_start = time.time()
    for rowindex, row in clean.iterrows():
        for column in numerical_columns + categorical_columns:
            if pd.isna(row[column]):
                location, month = tuple(row[group_columns])
                replacement = np.nan
                # replace the missing vlue with value in lookuptable
                if (location, month) in df_avg_mod.index:
                    replacement = df_avg_mod.loc[(location, month), column]

                # if theres no valid entry in the table
                if pd.isna(replacement):
                    if column in numerical_columns:
                        replacement = df_avg_mod.loc[location, column].mean()
                        if pd.isna(replacement):
                            replacement = dirty[column].mean()
                    else:
                        replacement = df_avg_mod.loc[location, column].mode()
                        if len(replacement) == 0 or pd.isna(replacement).any():
                            replacement = dirty[column].mode()
                # print(replacement)

                if type(replacement) is pd.core.series.Series:
                    for k in replacement:
                        replacement = k

                clean.at[rowindex, column] = replacement




    t_end = time.time()
    t_diff = t_end - t_start
    print('took ', t_diff)
    return clean



def task1():
    import time
    task_start = time.time()
    df_train_data = pd.read_csv("data/weather_train_data.csv")
    df_test_data = pd.read_csv("data/weather_test_data.csv")
    df_train_data_month = add_month_column(df_train_data)
    df_test_data_month = add_month_column(df_test_data)
    lookup_table = create_lookup_table(original=df_train_data_month)
    df_train_data_clean = clean_data_frame(dirty=df_train_data_month, lookup=lookup_table)
    df_test_data_clean = clean_data_frame(dirty=df_test_data_month, lookup=lookup_table)

    df_test_data_clean.to_csv('data/clean_weather_test_data.csv')
    print(df_train_data_clean)
    df_train_data_clean.to_csv('data/clean_weather_train_data.csv')

    print(pd.isna(df_test_data_clean).sum())
    print(pd.isna(df_train_data_clean).sum())

    task_end = time.time()
    task_diff = task_end - task_start
    print('took ', task_diff)

def test():

    dfuc = pd.read_csv("data/weather_train_data.csv")
    df = pd.read_csv("data/clean_weather_train_data.csv")

    null_counts = df.isnull().sum()
    null_counts[null_counts > 0].sort_values(ascending=False)
    null_countsUC = dfuc.isnull().sum()
    null_countsUC[null_countsUC > 0].sort_values(ascending=False)
    print(null_counts)
    print(null_countsUC)


if __name__ == "__main__":
    main()
