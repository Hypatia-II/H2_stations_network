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

sys.path.append('../app/pages/utils')
import functions_st

path_conf = '../app/pages/utils/params/config_st.json'
conf = json.load(open(path_conf, "r"))

st.set_page_config(page_title="Scenario Comparison", page_icon=":chart:", layout="wide")

st.markdown(
    """
    # 📊 Scenario Comparison
    
    This page will be used to compare different scenarios, either recommended by us to you, or your own created scenarios to have better insights on the H2 stations created.
    """
)

def handle_click_sc_names():
    if st.session_state['selected_scenarios2_change']:
        st.session_state.selected_scenarios2 = st.session_state.selected_scenarios2_change
        
# Load Data
df = functions_st.load_scenario_data()

scenarios2_names = list(df.index)
scenarios2_names_wo_sa = list(df.index)
scenarios2_names = ["Select All"] + scenarios2_names
if 'selected_scenarios2' not in st.session_state:
    st.session_state.selected_scenarios2 = "Select All"
selected_scenarios2 = st.multiselect('Select Scenarios of interest:', scenarios2_names, on_change=handle_click_sc_names, key='selected_scenarios2_change')
df_show = df.copy()
if (len(st.session_state.selected_scenarios2)==0):
    st.session_state.selected_scenarios2 = scenarios2_names_wo_sa
elif (not isinstance(st.session_state.selected_scenarios2, list)) and ("Select All" in st.session_state.selected_scenarios2):
    st.session_state.selected_scenarios2 = scenarios2_names_wo_sa
elif (isinstance(st.session_state.selected_scenarios2, list)) and ((len(st.session_state.selected_scenarios2)==1) and ("Select All" in st.session_state.selected_scenarios2)):
    st.session_state.selected_scenarios2 = scenarios2_names_wo_sa
else:
    if (((len(st.session_state.selected_scenarios2)>1)) & ("Select All" in st.session_state.selected_scenarios2)):
        st.session_state.selected_scenarios2.remove("Select All")
    df_show = df_show.loc[st.session_state.selected_scenarios2].copy()
df_show.sort_values('num_stations_2040', inplace=True, ascending=False)
cols_show = ['Total Number of Stations 2030', 'Total Number of Stations 2040']
df_show.columns = cols_show
index_names = [(sc_n.replace('Scenario ', '')+' Scenario') for sc_n in st.session_state.selected_scenarios2]
df_show.index = index_names

st.markdown('##')
df_plot = df_show.copy()
df_plot['scenario2'] = df_show.index
df_plot.reset_index(drop=True, inplace=True)
df_plot.sort_values('Total Number of Stations 2040', inplace=True)
chart_1 = alt.Chart(df_plot).mark_bar(color="#D8FAD9").encode(
    x=alt.X('scenario2:O', title='Scenario'),
    y=alt.Y('Total Number of Stations 2030:Q', title='Total Number of Stations')
).properties(
    width=500
)
chart_2 = alt.Chart(df_plot).mark_bar(color="#D8FAD9").encode(
    x=alt.X('scenario2:O', title='Scenario'),
    y=alt.Y('Total Number of Stations 2040:Q', title='Total Number of Stations')
).properties(
    width=500
)

line_1 = chart_1.mark_line(color='#5DB44C').encode(
    x='scenario2:O',
    y='Total Number of Stations 2030:Q'
)

line_2 = chart_2.mark_line(color='#5DB44C').encode(
    x='scenario2:O',
    y='Total Number of Stations 2040:Q'
)

col1, col2 = st.columns(2)

col1.subheader(":clock3: :red[2030]")
col2.subheader(":hourglass_flowing_sand: :red[2040]")

with col1:
    st.write(chart_1 + line_1)
with col2:
    st.write(chart_2 + line_2)




# st.write(chart + line)

st.markdown('##')

if len(df_show)>1:
    st.dataframe(df_show.style.highlight_max(color='#74CD67', axis=0).highlight_min(color = '#E57760', axis=0), use_container_width=True)
else:
    st.dataframe(df_show, use_container_width=True)
