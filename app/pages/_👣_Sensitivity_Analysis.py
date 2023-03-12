import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from owslib.wms import WebMapService

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import altair as alt
import json

import sys

print(sys.path)
sys.path.append('../app/pages/utils')
# sys.path.append('utils')
import functions_st
# print(sys.path)
# import functions_st

path_conf = '../params/config.json'
scenario ="scenario1"
conf = json.load(open(path_conf, "r"))

st.set_page_config(layout="wide", page_title="Sensitivity Analysis", page_icon=":compass:")

st.markdown("<h1 style='color:#FFF'>Sensitivity Analysis</h1>", unsafe_allow_html=True)

# Load Data

@st.cache(allow_output_mutation=True)
def load_data(path):
    df = pd.read_csv(path)
    return df

# The rest has to be updated !!!!!!!!!!!!

# # Check Box
# PAW = st.checkbox('PortAventura World', True)
# TG = st.checkbox('Tivoli Gardens')

# ########################### ATTENDANCE ################
# filtered_df_PAW = df_attendance[df_attendance["FACILITY_NAME"] == "PortAventura World"] if PAW else None
# filtered_df_TG = df_attendance[df_attendance["FACILITY_NAME"] == "Tivoli Gardens"] if TG else None

# if filtered_df_PAW is not None and filtered_df_TG is not None:
#     # Merge the two dataframes
#     merged_df = pd.merge(filtered_df_PAW, filtered_df_TG, on='USAGE_DATE', how='outer')
#     # Plot both on the same graph
#     grouped = merged_df.groupby('USAGE_DATE')['attendance_x', 'attendance_y'].mean()
#     st.line_chart(grouped)
# elif filtered_df_PAW is not None:
#     # Plot only PAW
#     grouped_PAW = filtered_df_PAW.groupby('USAGE_DATE')['attendance'].mean().reset_index()
#     #st.line_chart(grouped_PAW)
#     chart = alt.Chart(grouped_PAW).mark_line(color="#5DB44C").encode(
#         x= alt.X('USAGE_DATE', title="Date"),
#         y=alt.Y('attendance', title='Attendance'),
#     ).properties(
#         title='Total attendance over time', width=1250)
#     st.write(chart)
# elif filtered_df_TG is not None:
#     # Plot only TG
#     grouped_TG = filtered_df_TG.groupby('USAGE_DATE')['attendance'].mean().reset_index()
#     #st.line_chart(grouped_TG)
#     chart = alt.Chart(grouped_TG).mark_line(color="#A34CB4").encode(
#         x= alt.X('USAGE_DATE', title="Date"),
#         y=alt.Y('attendance', title='Attendance'),
#     ).properties(
#         title='Total attendance over time', width=1250)
#     st.write(chart)


######################### DELTA #################################
df = load_data("../data/df_st.csv")

# Dropdown
length_to_use ='longest_line'
if (length_to_use not in ['longest_line', 'diameter', 'length_max']):
    length_to_use = 'longest_line'
length_to_use = length_to_use

length_to_use = ['longest_line', 'diameter', 'length_max']
length_display = ['Longest Line', 'Diameter', 'Length Max']

## create a dropdown menu for the user to select the server name
selected_length_display = st.selectbox('Select department length calculation method:', length_display)
selected_length = length_to_use[length_display.index(selected_length_display)]

# calculate distance function

# Autonomy Market Share
autonomy_high_ms = int(conf[scenario]['market_share'][0])
autonomy_medium_ms = int(conf[scenario]['market_share'][1])
autonomy_low_ms = int(conf[scenario]['market_share'][2])

autonomy_high_km = int(conf['autonomy_share'][0])
autonomy_medium_km = int(conf['autonomy_share'][1])
autonomy_low_km = int(conf['autonomy_share'][2])
    
# Autonomy km
col1, col2, col3 = st.columns(3)
with col1:
    autonomy_high_km = st.number_input('Enter autonomy of first trucks', min_value=1, value=autonomy_high_km)
with col2:
    autonomy_medium_km = st.number_input('Enter autonomy of second trucks', min_value=1, value=autonomy_medium_km)
with col3:
    autonomy_low_km = st.number_input('Enter autonomy of third trucks', min_value=1, value=autonomy_low_km)
    
col1, col2, col3 = st.columns(3)
with col1:
    autonomy_high_ms = st.number_input('Enter market share of first trucks', min_value=0, max_value=1, value=autonomy_high_ms)
with col2:
    autonomy_medium_ms = st.number_input('Enter market share of second trucks', min_value=0, max_value=1, value=autonomy_medium_ms)
with col3:
    autonomy_low_ms = st.number_input('Enter market share of third trucks', min_value=0, max_value=1, value=autonomy_low_ms)


# Value entry & Selection
demand_share_2030 = float(conf[scenario]['demand_share_2030'])
demand_share_2040 = float(conf[scenario]['demand_share_2040'])

col1, col2 = st.columns(2)
with col1:
    demand_share_2030 = st.slider('H2 Trucks Target 2030', min_value=0.0, max_value=3.0, value=demand_share_2030, step=0.01)
with col2:
    demand_share_2040 = st.slider('H2 Trucks Target 2030', min_value=0.0, max_value=3.0, value=demand_share_2040, step=0.01)

# Value entry & Selection
truck_tank_size_high = conf["truck_tank_size"][0]
truck_tank_size_medium = conf["truck_tank_size"][1]
truck_tank_size_low = conf["truck_tank_size"][2]

col1, col2, col3 = st.columns(3)
with col1:
    truck_tank_size_high = st.slider('Tank size of first tank', min_value=1, max_value=200, value=truck_tank_size_high, step=1)
with col2:
    truck_tank_size_medium = st.slider('Tank size of second tank', min_value=1, max_value=200, value=truck_tank_size_medium, step=1)
with col3:
    truck_tank_size_low = st.slider('Tank size of third tank', min_value=1, max_value=200, value=truck_tank_size_low, step=1)

truck_tank_size = [truck_tank_size_high, truck_tank_size_medium, truck_tank_size_low]

# Value entry & Selection
station_tank_size_high = conf["station_tank_size"][0]
station_tank_size_medium = conf["station_tank_size"][1]
station_tank_size_low = conf["station_tank_size"][2]

col1, col2, col3 = st.columns(3)
with col1:
    station_tank_size_high = st.slider('Number of rows to display', min_value=1, max_value=20, value=station_tank_size_high, step=1)
with col2:
    station_tank_size_medium = st.slider('Number of rows to display', min_value=1, max_value=20, value=station_tank_size_medium, step=1)
with col3:
    station_tank_size_low = st.slider('Number of rows to display', min_value=1, max_value=20, value=station_tank_size_low, step=1)

station_tank_size = [station_tank_size_high, station_tank_size_medium, station_tank_size_low]


df, H2_trucks_num_2030, H2_trucks_num_2040 = functions_st.calculate_number_stations(df, 
                                            selected_length,
                                            demand_share_2030 = demand_share_2030,
                                            demand_share_2040 = demand_share_2040,
                                            autonomy_high_ms = autonomy_high_ms,
                                            autonomy_medium_ms = autonomy_medium_ms,
                                            autonomy_low_ms = autonomy_low_ms,
                                            autonomy_high_km = autonomy_high_km,
                                            autonomy_medium_km = autonomy_medium_km,
                                            autonomy_low_km = autonomy_low_km,
                                            truck_tank_size = truck_tank_size,
                                            station_tank_size = station_tank_size)

scenario_name = 'scenario_name_ex'

col1, col2 = st.columns(2)
with col1:
    if st.button('Save Scenario'):
        scenario_name = st.text_input('Enter the name to save: ', 'scenario_ex')
        functions_st.save_scenario(df, scenario_name)
        st.write(('Scenario saved as ' + scenario_name))
with col2:
    if st.button('Save Predictions'):
        functions_st.save_predictions(df, scenario_name)
        st.write('Predictions saved !')
        
# st.dataframe(grouped.head(n_rows).style.highlight_max(color='#D8FAD9', axis=0).highlight_min(color = '#FAD8F9', axis=0), use_container_width=True)


# col1, col2 = st.columns([2, 2])
# col1.subheader(":family: :green[Total number of visitors]")
# col2.subheader(':clock1: :green[Average Waiting Times] _(in minutes)_')

# avg_wait_time = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[0]
# capacity_utilization = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[1]
# avg_adjust_capacity_utilization = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[2]
# sum_attendance = functions_st.calculate_metrics(df, selected_year, selected_month, selected_day)[3]

# delta = None
# delta1 = None
# delta2 = None
# delta3 = None

# delta = functions_st.calculate_delta(df, selected_year, selected_month, selected_day, avg_wait_time, capacity_utilization, avg_adjust_capacity_utilization, sum_attendance, delta, delta1, delta2, delta3)[0]
# delta3 = functions_st.calculate_delta(df, selected_year, selected_month, selected_day, avg_wait_time, capacity_utilization, avg_adjust_capacity_utilization, sum_attendance, delta, delta1, delta2, delta3)[3]

# col1.metric("", "{:,.0f}".format(sum_attendance), delta= delta3)
# col2.metric("" , round(avg_wait_time, 2), delta=delta, delta_color="inverse")

# ################################################### WAITING TIMES ###################################################

# st.markdown("""---""")

# st.markdown("<h2 style='color:#5DB44C'>Let's dive into Waiting Times</h2>", unsafe_allow_html=True)

# if selected_year == "Select All" and selected_month == "Select All" and selected_day == "Select All":
#     filtered_df = df
# elif selected_year != "Select All" and selected_month == "Select All" and selected_day == "Select All":
#     filtered_df = df[(df['year'] == selected_year)]
# elif selected_year == "Select All" and selected_month != "Select All" and selected_day == "Select All":
#     filtered_df = df[(df['month'] == selected_month)]
# elif selected_year == "Select All" and selected_month == "Select All" and selected_day != "Select All":
#     filtered_df = df[(df['day'] == selected_day)]
# elif selected_year != "Select All" and selected_month != "Select All" and selected_day == "Select All":
#     filtered_df = df[(df['year'] == selected_year) &
#                     (df['month'] == selected_month)]
# elif selected_year != "Select All" and selected_month == "Select All" and selected_day != "Select All":
#     filtered_df = df[(df['year'] == selected_year) &
#                     (df['day'] == selected_day)]
# elif selected_year == "Select All" and selected_month != "Select All" and selected_day != "Select All":
#     filtered_df = df[(df['day'] == selected_day) &
#                     (df['month'] == selected_month)]
# elif selected_year != "Select All" and selected_month != "Select All" and selected_day != "Select All":
#     filtered_df = df[(df['year'] == selected_year) &
#                     (df['month'] == selected_month) &
#                     (df['day'] == selected_day)]

# #st.line_chart(filtered_df.groupby("WORK_DATE")["WAIT_TIME_MAX"].mean())
# time_df = filtered_df.groupby("WORK_DATE")["WAIT_TIME_MAX"].mean().reset_index()
# chart = alt.Chart(time_df).mark_line(color="#5DB44C").encode(
#         x= alt.X('WORK_DATE', title="Date"),
#         y=alt.Y('WAIT_TIME_MAX', title='Wait Time'),
#     ).properties(
#         title='Average Wait Time over time', width=1250)
# st.write(chart)

# col1, col2, col3, col4 = st.columns([1,1,1,1])
# col1.subheader(":rocket: :green[Minimum Waiting Times] _(in minutes)_")
# col2.subheader(":clock1: Date of the Min Waiting Times")
# col3.subheader(':turtle: :green[Maximum Waiting Times] _(in minutes)_')
# col4.subheader(":clock1: Date of the Max Waiting Times")

# min_wait_time = filtered_df['WAIT_TIME_MAX'].min()
# min_date = pd.Timestamp(filtered_df.loc[filtered_df['WAIT_TIME_MAX'] == min_wait_time, 'WORK_DATE'].iloc[0]).date().strftime("%d/%m/%Y")
# max_wait_time = filtered_df['WAIT_TIME_MAX'].max()
# max_date = pd.Timestamp(filtered_df.loc[filtered_df['WAIT_TIME_MAX'] == max_wait_time, 'WORK_DATE'].iloc[0]).date().strftime("%d/%m/%Y")

# col1.metric("", min_wait_time)
# col2.metric("", min_date)
# col3.metric("" , max_wait_time)
# col4.metric("", max_date)

# hourly_df = filtered_df.groupby("hour")["WAIT_TIME_MAX"].mean().reset_index()

# st.markdown('##')

# chart = alt.Chart(hourly_df).mark_bar(color="#D8FAD9").encode(
#     x=alt.X('hour:O', title='Hour'),
#     y=alt.Y('WAIT_TIME_MAX:Q', title='Avg Wait Time')
# ).properties(
#     width=1250
# )

# line = chart.mark_line(color='#5DB44C').encode(
#     x='hour:O',
#     y='WAIT_TIME_MAX:Q'
# )

# st.write(chart + line)
