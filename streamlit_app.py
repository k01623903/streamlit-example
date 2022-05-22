import streamlit as st
from lists import location_options, comp_dir_options
import pandas as pd
import numpy as np
import pickle
import classifire
import datacleaning
import prediction

st.write("""
# How's the weather?
Will it rain tomorrow?
""")


def location_format_func(option):
    return location_options[option]

def dir_format_func(option):
    return comp_dir_options[option]



st.balloons()
form = st.form(key='my_form')
date = form.date_input(label='Date')
location = form.selectbox("Location", options=list(location_options.keys()), format_func=location_format_func)
min_temp = form.number_input(label='Min Temp')
max_temp = form.number_input(label='Max Temp')
rainfall = form.number_input(label='Rainfall')
evaporation = form.number_input(label='Evaporation')
sunshine = form.number_input(label='Sunshine')
wind_gust_dir = form.selectbox("WindGustDir", options=list(comp_dir_options.keys()), format_func=dir_format_func)
wind_gust_speed = form.number_input(label='WindGustSpeed')
wind_dir_9am = form.selectbox("WindDir9am", options=list(comp_dir_options.keys()), format_func=dir_format_func)
wind_dir_3pm = form.selectbox("WindDir3pm", options=list(comp_dir_options.keys()), format_func=dir_format_func)
wind_speed_9am = form.number_input(label='WindSpeed9am')
wind_speed_3pm = form.number_input(label='WindSpeed3pm')
humidity_9am = form.number_input(label='Humidity9am')
humidity_3pm = form.number_input(label='Humidity3pm')
pressure_9am = form.number_input(label='Pressure9am')
pressure_3pm = form.number_input(label='Pressure3pm')
cloud_9am = form.number_input(label='Cloud9am')
cloud_3pm = form.number_input(label='Cloud3pm')
temp_9am = form.number_input(label='Temp9am')
temp_3pm = form.number_input(label='Temp3pm')
submit_form = form.form_submit_button(label='Submit')

if (submit_form):
    if rainfall > 0:
        rt = 'Yes'
    else:
        rt = 'No'

    lookup_table = pd.read_csv("data/AvgModLookup.csv")

    df_to_predict = pd.DataFrame({"Date":[str(date)],
                               "Location": [str(location_options[location])],
                               "MinTemp":[min_temp],
                               "MaxTemp":[max_temp],
                               "Rainfall":[rainfall],
                               "Evaporation":[evaporation],
                               "Sunshine":[sunshine],
                               "WindGustDir":[str(comp_dir_options[wind_gust_dir])],
                               "WindGustSpeed":[wind_gust_speed],
                               "WindDir9am":[str(comp_dir_options[wind_dir_9am])],
                               "WindDir3pm":[str(comp_dir_options[wind_dir_3pm])],
                               "WindSpeed9am":[wind_speed_9am],
                               "WindSpeed3pm":[wind_speed_3pm],
                               "Humidity9am":[humidity_9am],
                               "Humidity3pm":[humidity_3pm],
                               "Pressure9am":[pressure_9am],
                               "Pressure3pm":[pressure_3pm],
                               "Cloud9am":[cloud_9am],
                               "Cloud3pm":[cloud_3pm],
                               "Temp9am":[temp_9am],
                               "Temp3pm":[temp_3pm],
                               "RainToday":[rt]})

    df_to_predict_month = datacleaning.add_month_column(df_to_predict)
    df_to_predict_clean = datacleaning.clean_data_frame(dirty=df_to_predict_month, lookup=lookup_table)

    df_to_predict = prediction.reduce_data(df_to_predict_clean)
    df_to_predict = prediction.convert(df_to_predict)

    prediction = prediction.predict(df_to_predict)
    df_to_predict["predicion"] = prediction




    if (prediction == 'Yes'):
        st.write('It will rain in ' + location_format_func(location))
        st.write('Better take an umbrella with you!')
    else:
        st.write('Predicting no rain tomorrow! Sun will shine in ' + location_format_func(location))
