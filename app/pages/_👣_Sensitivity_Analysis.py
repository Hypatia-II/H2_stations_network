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

path_conf = '../app/pages/utils/params/config_st.json'
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

keys_l = list((conf.keys()))
scenario_list = [ck for ck in keys_l if ck.startswith('scenario')]
scenario = scenario_list[0]

path_output_sc = '../data/output_' + scenario + '.json'
conf_o_sc = json.load(open(path_output_sc, "r"))
H2_stations_2030 = int(sum(conf_o_sc['num_stations_2030'].values()))
H2_stations_2040 = int(sum(conf_o_sc['num_stations_2040'].values()))

scenario = st.selectbox('Select department length calculation method:', scenario_list)

# Dropdown
selected_length ='longest_line'

# ## create a dropdown menu for the user to select the server name
# length_to_use = ['longest_line', 'diameter', 'length_max']
# length_display = ['Longest Line', 'Diameter', 'Length Max']
# selected_length_display = st.selectbox('Select department length calculation method:', length_display)
# selected_length = length_to_use[length_display.index(selected_length_display)]

# calculate distance function

# Autonomy Market Share
autonomy_high_ms = float(conf[scenario]['market_share'][0])
autonomy_medium_ms = float(conf[scenario]['market_share'][1])
autonomy_low_ms = float(conf[scenario]['market_share'][2])

autonomy_high_km = int(conf['autonomy_share'][0])
autonomy_medium_km = int(conf['autonomy_share'][1])
autonomy_low_km = int(conf['autonomy_share'][2])
    
# # Autonomy km
# col1, col2, col3 = st.columns(3)
# with col1:
#     autonomy_high_km = st.number_input('Enter autonomy of first trucks', min_value=1, value=autonomy_high_km)
# with col2:
#     autonomy_medium_km = st.number_input('Enter autonomy of second trucks', min_value=1, value=autonomy_medium_km)
# with col3:
#     autonomy_low_km = st.number_input('Enter autonomy of third trucks', min_value=1, value=autonomy_low_km)
    
# Market Share
col1, col2, col3 = st.columns(3)
with col1:
    autonomy_high_ms = st.number_input('Enter market share of first trucks', min_value=0.00, max_value=1.00, value=autonomy_high_ms)
with col2:
    autonomy_medium_ms = st.number_input('Enter market share of second trucks', min_value=0.00, max_value=1.00, value=autonomy_medium_ms)
with col3:
    autonomy_low_ms = st.number_input('Enter market share of third trucks', min_value=0.00, max_value=1.00, value=autonomy_low_ms)

# Fix this part
# if (autonomy_high_ms+autonomy_medium_ms+autonomy_low_ms)!=1.0:
#     if (autonomy_high_ms+autonomy_medium_ms)>1:
#         autonomy_medium_ms = 1 - autonomy_high_ms
#     elif (autonomy_medium_ms+autonomy_low_ms)>1:
#         autonomy_low_ms = 1 - autonomy_medium_ms
#     elif (autonomy_high_ms+autonomy_low_ms)>1:
#         autonomy_low_ms = 1 - autonomy_high_ms
#     else:
#         autonomy_low_ms = 1 - autonomy_high_ms - autonomy_medium_ms
        
if (autonomy_high_ms+autonomy_medium_ms+autonomy_low_ms)>1.0:
    st.write('The sum of market shares exceeds 1!')

# Value entry & Selection
demand_share_2030 = float(conf[scenario]['demand_share_2030'])
demand_share_2040 = float(conf[scenario]['demand_share_2040'])

col1, col2 = st.columns(2)
with col1:
    demand_share_2030 = st.slider('H2 Trucks Target 2030', min_value=0.0, max_value=3.0, value=demand_share_2030, step=0.01)
with col2:
    demand_share_2040 = st.slider('H2 Trucks Target 2040', min_value=0.0, max_value=3.0, value=demand_share_2040, step=0.01)

# Value entry & Selection
truck_tank_size_high = conf["truck_tank_size"][0]
truck_tank_size_medium = conf["truck_tank_size"][1]
truck_tank_size_low = conf["truck_tank_size"][2]

# col1, col2, col3 = st.columns(3)
# with col1:
#     truck_tank_size_high = st.slider('Tank size of first tank', min_value=1, max_value=200, value=truck_tank_size_high, step=1)
# with col2:
#     truck_tank_size_medium = st.slider('Tank size of second tank', min_value=1, max_value=200, value=truck_tank_size_medium, step=1)
# with col3:
#     truck_tank_size_low = st.slider('Tank size of third tank', min_value=1, max_value=200, value=truck_tank_size_low, step=1)

truck_tank_size = [truck_tank_size_high, truck_tank_size_medium, truck_tank_size_low]

# Value entry & Selection
station_tank_size_high = conf["station_tank_size"][0]
station_tank_size_medium = conf["station_tank_size"][1]
station_tank_size_low = conf["station_tank_size"][2]

# col1, col2, col3 = st.columns(3)
# with col1:
#     station_tank_size_high = st.slider('Select the tank size of the first small stations', min_value=1, max_value=20, value=station_tank_size_high, step=1)
# with col2:
#     station_tank_size_medium = st.slider('Select the tank size of the medium stations', min_value=1, max_value=20, value=station_tank_size_medium, step=1)
# with col3:
#     station_tank_size_low = st.slider('Select the tank size of the largest stations', min_value=1, max_value=20, value=station_tank_size_low, step=1)

station_tank_size = [station_tank_size_high, station_tank_size_medium, station_tank_size_low]

delta1 = None
delta2 = None

df, H2_stations_2030, H2_stations_2040, delta1, delta2 = functions_st.calculate_number_stations(df, 
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
                                            station_tank_size = station_tank_size,
                                            H2_stations_2030=H2_stations_2030, 
                                            H2_stations_2040=H2_stations_2040,
                                            delta1=delta1,
                                            delta2=delta2)


scenario_name_ex = 'scenario_name_ex'

col1, col2 = st.columns(2)
with col1:
    if st.button('Save Scenario'):
        scenario_name = st.text_input('Enter the name to save: ', scenario_name_ex)
        functions_st.save_scenario(df, scenario_name)
        st.write(('Scenario saved as ' + scenario_name))
with col2:
    if st.button('Save Predictions'):
        functions_st.save_predictions(df, scenario_name)
        st.write('Predictions saved !')

cols_keep = ['region', 'R_2030_total', 'R_2040_total', 'h2_num_2030', 'h2_num_2040', 'num_stations_2030', 'num_stations_2040']
cols_show = ['Region', 'Refills 2030', 'Refills 2040', 'H2 Trucks 2030', 'H2 Trucks 2040', 'Number of Stations 2030', 'Number of Stations 2040']
df_show = df[cols_keep]
df_show.sort_values(by='num_stations_2040', ascending=False, inplace=True)
df_show.columns = cols_show
df_show.set_index('Region', drop=True, inplace=True)

col1, col2 = st.columns([2, 2])
col1.subheader(":family: :green[by 2030]")
col2.subheader(":clock1: :green[by 2040]")

if delta1!=None:
    delta1 = int(delta1)
    
if delta2!=None:
    delta2 = int(delta2)
    
col1.metric("", H2_stations_2030, delta=delta1)
col2.metric("", H2_stations_2040, delta=delta2, delta_color="inverse")

st.dataframe(df_show.style.highlight_max(color='#74CD67', axis=0).highlight_min(color = '#E57760', axis=0), use_container_width=True)

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
