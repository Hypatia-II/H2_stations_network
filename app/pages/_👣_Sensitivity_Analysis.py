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
# if 'conf' not in st.session_state:
conf = json.load(open(path_conf, "r"))

st.set_page_config(layout="wide", page_title="Sensitivity Analysis", page_icon=":compass:")

st.markdown("<h1 style='color:#FFF'>🤌 Sensitivity Analysis</h1>", unsafe_allow_html=True)

# Load Data
keys_l = list((conf.keys()))
scenario_list = [ck for ck in keys_l if ck.startswith('Scenario ')]
if 'scenario' not in st.session_state:
    st.session_state.scenario = scenario_list[0]
# scenario = scenario_list[0]
scenario_file = st.session_state.scenario.replace(" ", "_")
path_output_sc = '../data/output_' + scenario_file + '.json'
conf_o_sc = json.load(open(path_output_sc, "r"))
if 'H2_stations_2030' not in st.session_state:
    st.session_state.H2_stations_2030 = int(sum(conf_o_sc['num_stations_2030'].values()))
if 'H2_stations_2040' not in st.session_state:
    st.session_state.H2_stations_2040 = int(sum(conf_o_sc['num_stations_2040'].values()))

@st.cache(allow_output_mutation=True)
def load_data(path):
    df = pd.read_csv(path)
    return df


# def handle_click(new_var):
#     st.session_state.scenario = new_var
#     return None

def handle_click_no_button():
    if st.session_state['scenario_change']:
        st.session_state.scenario = 'Scenario ' + st.session_state.scenario_change
        st.session_state.autonomy_high_ms = float(conf[st.session_state.scenario]['market_share'][0])
        st.session_state.autonomy_medium_ms = float(conf[st.session_state.scenario]['market_share'][1])
        st.session_state.autonomy_low_ms = float(conf[st.session_state.scenario]['market_share'][2])
        st.session_state.demand_share_2030 = float(conf[st.session_state.scenario]['demand_share_2030'])
        st.session_state.demand_share_2040 = float(conf[st.session_state.scenario]['demand_share_2040'])
        
def handle_click_no_button_sc():
    if st.session_state['sc_name_change']:
        st.session_state.scenario_name = 'Scenario ' + st.session_state.sc_name_change
        
def handle_click_all_vars():
    if st.session_state['demand_share_2030_change']:
        st.session_state.demand_share_2030 = st.session_state.demand_share_2030_change
    if st.session_state['demand_share_2040_change']:
        st.session_state.demand_share_2040 = st.session_state.demand_share_2040_change  
    if st.session_state['autonomy_high_ms_change']:
        st.session_state.autonomy_high_ms = st.session_state.autonomy_high_ms_change
    if st.session_state['autonomy_medium_ms_change']:
        st.session_state.autonomy_medium_ms = st.session_state.autonomy_medium_ms_change
    if st.session_state['autonomy_low_ms_change']:
        st.session_state.autonomy_low_ms = st.session_state.autonomy_low_ms_change  

######################### DELTA #################################
# if 'df' not in st.session_state:
#     st.session_state.df = load_data("../data/df_st.csv")
df = load_data("../data/df_st.csv")

scenario_list_show = [sc.replace("Scenario ", "") for sc in scenario_list]
scenario = st.selectbox('Select preset scenario if wanted:', scenario_list_show, on_change=handle_click_no_button, key='scenario_change')
# scenario = 'Scenario ' + scenario

selected_length ='longest_line'

# Trucks Market Share
if 'autonomy_high_ms' not in st.session_state:
    st.session_state.autonomy_high_ms = float(conf[st.session_state.scenario]['market_share'][0])
if 'autonomy_medium_ms' not in st.session_state:
    st.session_state.autonomy_medium_ms = float(conf[st.session_state.scenario]['market_share'][1])
if 'autonomy_low_ms' not in st.session_state:
    st.session_state.autonomy_low_ms = float(conf[st.session_state.scenario]['market_share'][2])

# Trucks Autonomy
autonomy_high_km = int(conf['autonomy_share'][0])
autonomy_medium_km = int(conf['autonomy_share'][1])
autonomy_low_km = int(conf['autonomy_share'][2])

# Demand Share
if 'demand_share_2030' not in st.session_state:
    st.session_state.demand_share_2030 = float(conf[st.session_state.scenario]['demand_share_2030'])
if 'demand_share_2040' not in st.session_state:
    st.session_state.demand_share_2040 = float(conf[st.session_state.scenario]['demand_share_2040'])

truck_tank_size_high = conf["truck_tank_size"][0]
truck_tank_size_medium = conf["truck_tank_size"][1]
truck_tank_size_low = conf["truck_tank_size"][2]
truck_tank_size = [truck_tank_size_high, truck_tank_size_medium, truck_tank_size_low]

station_tank_size_high = conf["station_tank_size"][0]
station_tank_size_medium = conf["station_tank_size"][1]
station_tank_size_low = conf["station_tank_size"][2]

station_tank_size = [station_tank_size_high, station_tank_size_medium, station_tank_size_low]

# ## create a dropdown menu for the user to select the server name
# length_to_use = ['longest_line', 'diameter', 'length_max']
# length_display = ['Longest Line', 'Diameter', 'Length Max']
# selected_length_display = st.selectbox('Select department length calculation method:', length_display)
# selected_length = length_to_use[length_display.index(selected_length_display)]

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
    autonomy_high_ms = st.number_input('Enter market share of first trucks', min_value=0.00, max_value=1.00, 
                                       value=st.session_state.autonomy_high_ms, on_change=handle_click_all_vars, 
                                       key='autonomy_high_ms_change')
with col2:
    autonomy_medium_ms = st.number_input('Enter market share of second trucks', min_value=0.00, max_value=1.00, 
                                         value=st.session_state.autonomy_medium_ms, on_change=handle_click_all_vars, 
                                         key='autonomy_medium_ms_change')
with col3:
    autonomy_low_ms = st.number_input('Enter market share of third trucks', min_value=0.00, max_value=1.00, 
                                      value=st.session_state.autonomy_low_ms, on_change=handle_click_all_vars, 
                                      key='autonomy_low_ms_change')

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
        
if (st.session_state.autonomy_high_ms+st.session_state.autonomy_medium_ms+st.session_state.autonomy_low_ms)>1.0:
    st.write('The sum of market shares exceeds 1!')

# H2 target by 2030 and 2040
col1, col2 = st.columns(2)
with col1:
    demand_share_2030 = st.slider('H2 Trucks Target 2030', min_value=0.0, max_value=3.0, value=st.session_state.demand_share_2030, 
                                  step=0.01, on_change=handle_click_all_vars, key='demand_share_2030_change')
with col2:
    demand_share_2040 = st.slider('H2 Trucks Target 2040', min_value=0.0, max_value=3.0, value=st.session_state.demand_share_2040, 
                                  step=0.01, on_change=handle_click_all_vars, key='demand_share_2040_change')

col1, col2 = st.columns(2)
col1.metric("", "{:,.0f}".format(int(round((st.session_state.demand_share_2030*10000), 2))))
col2.metric("", "{:,.0f}".format(int(round((st.session_state.demand_share_2040*60000), 2))))

# Value entry & Selection


# col1, col2, col3 = st.columns(3)
# with col1:
#     truck_tank_size_high = st.slider('Tank size of first tank', min_value=1, max_value=200, value=truck_tank_size_high, step=1)
# with col2:
#     truck_tank_size_medium = st.slider('Tank size of second tank', min_value=1, max_value=200, value=truck_tank_size_medium, step=1)
# with col3:
#     truck_tank_size_low = st.slider('Tank size of third tank', min_value=1, max_value=200, value=truck_tank_size_low, step=1)

# col1, col2, col3 = st.columns(3)
# with col1:
#     station_tank_size_high = st.slider('Select the tank size of the first small stations', min_value=1, max_value=20, value=station_tank_size_high, step=1)
# with col2:
#     station_tank_size_medium = st.slider('Select the tank size of the medium stations', min_value=1, max_value=20, value=station_tank_size_medium, step=1)
# with col3:
#     station_tank_size_low = st.slider('Select the tank size of the largest stations', min_value=1, max_value=20, value=station_tank_size_low, step=1)

if 'delta1' not in st.session_state:
    st.session_state.delta1 = None
if 'delta2' not in st.session_state:
    st.session_state.delta2 = None

df, st.session_state.H2_stations_2030, st.session_state.H2_stations_2040, st.session_state.delta1, st.session_state.delta2 = functions_st.calculate_number_stations(df, 
                                            selected_length,
                                            demand_share_2030 = st.session_state.demand_share_2030,
                                            demand_share_2040 = st.session_state.demand_share_2040,
                                            autonomy_high_ms = st.session_state.autonomy_high_ms,
                                            autonomy_medium_ms = st.session_state.autonomy_medium_ms,
                                            autonomy_low_ms = st.session_state.autonomy_low_ms,
                                            autonomy_high_km = autonomy_high_km,
                                            autonomy_medium_km = autonomy_medium_km,
                                            autonomy_low_km = autonomy_low_km,
                                            truck_tank_size = truck_tank_size,
                                            station_tank_size = station_tank_size,
                                            H2_stations_2030=st.session_state.H2_stations_2030, 
                                            H2_stations_2040=st.session_state.H2_stations_2040,
                                            del1=st.session_state.delta1,
                                            del2=st.session_state.delta2)


scenario_name_ex = 'Example'
if 'scenario_name' not in st.session_state:
    st.session_state.scenario_name = 'Scenario Example'
# scenario_name = 'Scenario Example'
col1, col2 = st.columns(2)
with col1:
    if st.button('Save Scenario'):
        scenario_name = st.text_input('Enter the name to save: ', scenario_name_ex, on_change=handle_click_no_button_sc, key='sc_name_change')
        # scenario_name = 'Scenario ' + scenario_name
        functions_st.save_scenario(scenario_name=st.session_state.scenario_name)
        st.write(('Scenario saved as ' + st.session_state.scenario_name))
with col2:
    if st.button('Save Predictions'):
        scenario_file_save = st.session_state.scenario_name.replace(" ", "_")
        functions_st.save_predictions(df, scenario_name=scenario_file_save)
        st.write('Predictions saved !')

cols_keep = ['region', 'R_2030_total', 'R_2040_total', 'h2_num_2030', 'h2_num_2040', 'num_stations_2030', 'num_stations_2040']
cols_show = ['Region', 'Refills 2030', 'Refills 2040', 'H2 Trucks 2030', 'H2 Trucks 2040', 'Total Number of Stations 2030', 'Total Number of Stations 2040']
df_show = df[cols_keep].copy()
df_show.sort_values(by='num_stations_2040', ascending=False, inplace=True)
df_show.columns = cols_show
df_show.set_index('Region', drop=True, inplace=True)
df_show = df_show.apply(np.floor)
df_show = df_show.astype(int)

st.markdown('##')

st.markdown("<h2 style='color:#FFF'>Number of H2 Stations by:</h2>", unsafe_allow_html=True)
if st.session_state.delta1!=None:
    st.session_state.delta1 = int(st.session_state.delta1)
    
if st.session_state.delta2!=None:
    st.session_state.delta2 = int(st.session_state.delta2)
col1, col2 = st.columns([2, 2])
col1.subheader(":clock3: :red[2030]")
col2.subheader(":hourglass_flowing_sand: :red[2040]")
col1.metric("", st.session_state.H2_stations_2030, delta=st.session_state.delta1)
col2.metric("", st.session_state.H2_stations_2040, delta=st.session_state.delta2)

st.dataframe(df_show.style.highlight_max(color='#74CD67', axis=0).highlight_min(color = '#E57760', axis=0), use_container_width=True)


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
