import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from owslib.wms import WebMapService

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import altair as alt

import sys
sys.path.append('../app/pages/utils')
import functions_st


st.set_page_config(layout="wide", page_title="Sensitivity Analysis", page_icon=":compass:")

st.markdown("<h1 style='color:#5DB44C'>Sensitivity Analysis</h1>", unsafe_allow_html=True)

# Load Data

@st.cache(allow_output_mutation=True)
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data("../data/df_st.csv")

# The rest has to be updated !!!!!!!!!!!!

# Check Box
PAW = st.checkbox('PortAventura World', True)
TG = st.checkbox('Tivoli Gardens')

########################### ATTENDANCE ################
filtered_df_PAW = df_attendance[df_attendance["FACILITY_NAME"] == "PortAventura World"] if PAW else None
filtered_df_TG = df_attendance[df_attendance["FACILITY_NAME"] == "Tivoli Gardens"] if TG else None

if filtered_df_PAW is not None and filtered_df_TG is not None:
    # Merge the two dataframes
    merged_df = pd.merge(filtered_df_PAW, filtered_df_TG, on='USAGE_DATE', how='outer')
    # Plot both on the same graph
    grouped = merged_df.groupby('USAGE_DATE')['attendance_x', 'attendance_y'].mean()
    st.line_chart(grouped)
elif filtered_df_PAW is not None:
    # Plot only PAW
    grouped_PAW = filtered_df_PAW.groupby('USAGE_DATE')['attendance'].mean().reset_index()
    #st.line_chart(grouped_PAW)
    chart = alt.Chart(grouped_PAW).mark_line(color="#5DB44C").encode(
        x= alt.X('USAGE_DATE', title="Date"),
        y=alt.Y('attendance', title='Attendance'),
    ).properties(
        title='Total attendance over time', width=1250)
    st.write(chart)
elif filtered_df_TG is not None:
    # Plot only TG
    grouped_TG = filtered_df_TG.groupby('USAGE_DATE')['attendance'].mean().reset_index()
    #st.line_chart(grouped_TG)
    chart = alt.Chart(grouped_TG).mark_line(color="#A34CB4").encode(
        x= alt.X('USAGE_DATE', title="Date"),
        y=alt.Y('attendance', title='Attendance'),
    ).properties(
        title='Total attendance over time', width=1250)
    st.write(chart)


######################### DELTA #################################
df = load_data("../data/data_merged.csv")
#Preprocess
df = functions_st.get_data_ready(df)

# Dropdown
attraction_names = df['ENTITY_DESCRIPTION_SHORT'].unique()
attraction_names = ["Select All"] + list(attraction_names)

## create a dropdown menu for the user to select the server name
selected_attraction = st.selectbox('Select attraction:', attraction_names)

if selected_attraction == "Select All":
    df = df
else:
    df = df[df['ENTITY_DESCRIPTION_SHORT'] == selected_attraction]

col1, col2, col3 = st.columns(3)


years = df['year'].unique()
years =  ["Select All"] + list(years)

with col1:
    selected_year = st.selectbox('Select year:', years)

months = df['month'].unique()
months =  ["Select All"] + list(months)

with col2:
    selected_month = st.selectbox('Select month:', months)

days = df['day'].unique()
days = ["Select All"] + list(days)

with col3:
    selected_day = st.selectbox('Select day:', days)

col1, col2 = st.columns([2, 2])
col1.subheader(":family: :green[Total number of visitors]")
col2.subheader(':clock1: :green[Average Waiting Times] _(in minutes)_')

avg_wait_time = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[0]
capacity_utilization = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[1]
avg_adjust_capacity_utilization = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[2]
sum_attendance = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[3]

delta = None
delta1 = None
delta2 = None
delta3 = None

delta = functions_st.calculate_delta(df, selected_year, selected_month, selected_day, avg_wait_time, capacity_utilization, avg_adjust_capacity_utilization, sum_attendance, delta, delta1, delta2, delta3)[0]
delta3 = functions_st.calculate_delta(df, selected_year, selected_month, selected_day, avg_wait_time, capacity_utilization, avg_adjust_capacity_utilization, sum_attendance, delta, delta1, delta2, delta3)[3]

col1.metric("", "{:,.0f}".format(sum_attendance), delta= delta3)
col2.metric("" , round(avg_wait_time, 2), delta=delta, delta_color="inverse")

################################################### WAITING TIMES ###################################################

st.markdown("""---""")

st.markdown("<h2 style='color:#5DB44C'>Let's dive into Waiting Times</h2>", unsafe_allow_html=True)

if selected_year == "Select All" and selected_month == "Select All" and selected_day == "Select All":
    filtered_df = df
elif selected_year != "Select All" and selected_month == "Select All" and selected_day == "Select All":
    filtered_df = df[(df['year'] == selected_year)]
elif selected_year == "Select All" and selected_month != "Select All" and selected_day == "Select All":
    filtered_df = df[(df['month'] == selected_month)]
elif selected_year == "Select All" and selected_month == "Select All" and selected_day != "Select All":
    filtered_df = df[(df['day'] == selected_day)]
elif selected_year != "Select All" and selected_month != "Select All" and selected_day == "Select All":
    filtered_df = df[(df['year'] == selected_year) &
                    (df['month'] == selected_month)]
elif selected_year != "Select All" and selected_month == "Select All" and selected_day != "Select All":
    filtered_df = df[(df['year'] == selected_year) &
                    (df['day'] == selected_day)]
elif selected_year == "Select All" and selected_month != "Select All" and selected_day != "Select All":
    filtered_df = df[(df['day'] == selected_day) &
                    (df['month'] == selected_month)]
elif selected_year != "Select All" and selected_month != "Select All" and selected_day != "Select All":
    filtered_df = df[(df['year'] == selected_year) &
                    (df['month'] == selected_month) &
                    (df['day'] == selected_day)]

#st.line_chart(filtered_df.groupby("WORK_DATE")["WAIT_TIME_MAX"].mean())
time_df = filtered_df.groupby("WORK_DATE")["WAIT_TIME_MAX"].mean().reset_index()
chart = alt.Chart(time_df).mark_line(color="#5DB44C").encode(
        x= alt.X('WORK_DATE', title="Date"),
        y=alt.Y('WAIT_TIME_MAX', title='Wait Time'),
    ).properties(
        title='Average Wait Time over time', width=1250)
st.write(chart)

col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.subheader(":rocket: :green[Minimum Waiting Times] _(in minutes)_")
col2.subheader(":clock1: Date of the Min Waiting Times")
col3.subheader(':turtle: :green[Maximum Waiting Times] _(in minutes)_')
col4.subheader(":clock1: Date of the Max Waiting Times")

min_wait_time = filtered_df['WAIT_TIME_MAX'].min()
min_date = pd.Timestamp(filtered_df.loc[filtered_df['WAIT_TIME_MAX'] == min_wait_time, 'WORK_DATE'].iloc[0]).date().strftime("%d/%m/%Y")
max_wait_time = filtered_df['WAIT_TIME_MAX'].max()
max_date = pd.Timestamp(filtered_df.loc[filtered_df['WAIT_TIME_MAX'] == max_wait_time, 'WORK_DATE'].iloc[0]).date().strftime("%d/%m/%Y")

col1.metric("", min_wait_time)
col2.metric("", min_date)
col3.metric("" , max_wait_time)
col4.metric("", max_date)

hourly_df = filtered_df.groupby("hour")["WAIT_TIME_MAX"].mean().reset_index()

st.markdown('##')

chart = alt.Chart(hourly_df).mark_bar(color="#D8FAD9").encode(
    x=alt.X('hour:O', title='Hour'),
    y=alt.Y('WAIT_TIME_MAX:Q', title='Avg Wait Time')
).properties(
    width=1250
)

line = chart.mark_line(color='#5DB44C').encode(
    x='hour:O',
    y='WAIT_TIME_MAX:Q'
)

st.write(chart + line)
