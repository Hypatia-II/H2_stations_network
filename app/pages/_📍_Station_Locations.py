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
import os

sys.path.append('../app/pages/utils')
import functions_st

path_conf = '../app/pages/utils/params/config_st.json'
conf = json.load(open(path_conf, "r"))
path_scenario_html = '../app/pages/utils/html_output'

st.set_page_config(page_title="Station Locations", page_icon=":detective:", layout="wide")

st.markdown(
    """
    # 📍 Station Locations
    
    This section is to help you detect churners, based on a specific period of time. The output will be a list of the clients with their client id, and potentially personal information.
    """
)

def handle_click_no_button():
    if st.session_state['scenario_change']:
        st.session_state.scenario = st.session_state.scenario_change
        st.session_state.path_to_html = path_scenario_html + "/output_Scenario_" + st.session_state.scenario.replace(" ", "_") + ".html" 
        # st.session_state.autonomy_high_ms = float(conf[st.session_state.scenario]['market_share'][0])
        # st.session_state.autonomy_medium_ms = float(conf[st.session_state.scenario]['market_share'][1])
        # st.session_state.autonomy_low_ms = float(conf[st.session_state.scenario]['market_share'][2])
        # st.session_state.demand_share_2030 = float(conf[st.session_state.scenario]['demand_share_2030'])
        # st.session_state.demand_share_2040 = float(conf[st.session_state.scenario]['demand_share_2040'])
        
scenarios_list = [sc.replace("output_Scenario_", "").replace(".html","").replace("_", " ") for sc in os.listdir(path_scenario_html)]

# Load Data
# df = functions_st.load_scenario_data()

if 'scenario' not in st.session_state:
    st.session_state.scenario = scenarios_list[0]
if 'path_to_html' not in st.session_state:
    st.session_state.path_to_html = path_scenario_html + "/output_Scenario_" + st.session_state.scenario.replace(" ", "_") + ".html" 
    
scenario = st.selectbox('Select Scenarios of interest:', scenarios_list, on_change=handle_click_no_button, key='scenario_change')
print(st.session_state.path_to_html)
# df_show = df.copy()
    
# path_to_html = path_scenario_html + "/output_" + scenario.replace(" ", "_") + ".html" 
with open(st.session_state.path_to_html,'r') as f: 
    html_data = f.read()

## Show in webpage
# st.header("Show an external HTML")
# st.components.v1.html(html_data, width=500, height=500)

col1, col2 = st.columns(2)
with col1:
    st.header("H2 Stations Locations")
    st.components.v1.html(html_data, width=500, height=500)
with col2:
    st.header("Density Map")
    st.components.v1.html(html_data, width=500, height=500)