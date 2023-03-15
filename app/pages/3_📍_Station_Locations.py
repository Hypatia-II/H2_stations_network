import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import altair as alt
import json
import sys
import os
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import pyproj


sys.path.append('../app/pages/utils')

path_conf = '../app/pages/utils/params/config_st.json'
conf = json.load(open(path_conf, "r"))
path_scenario1_html = '../app/pages/utils/html_output'

st.set_page_config(page_title="Station Locations", page_icon=":detective:", layout="wide")

st.markdown(
    """
    # 📍 Station Locations
    
    This section helps visualize the stations locations based on the basic scenarios provided.
    """
)

def handle_click_no_button():
    if st.session_state['scenario1_change']:
        st.session_state.scenario1 = st.session_state.scenario1_change
        st.session_state.path_to_html = path_scenario1_html + "/output_Scenario_" + st.session_state.scenario1.replace(" ", "_") + ".html" 
        st.session_state.path_output_sc = '../data/output_Scenario_' + st.session_state.scenario1.replace(" ", "_") + '.json'
        st.session_state.path_output_map = path_scenario1_html + '/' + st.session_state.scenario1.replace(" ", "_")

scenarios1_list = [sc.replace("output_Scenario_", "").replace(".html","").replace("_", " ") for sc in os.listdir(path_scenario1_html) if sc.startswith("output_Scenario_")]

if 'scenario1' not in st.session_state:
    st.session_state.scenario1 = scenarios1_list[0]
if 'path_to_html' not in st.session_state:
    st.session_state.path_to_html = path_scenario1_html + "/output_Scenario_" + st.session_state.scenario1.replace(" ", "_") + ".html" 
if 'path_output_sc' not in st.session_state:
    st.session_state.path_output_sc = '../data/output_Scenario_' + st.session_state.scenario1.replace(" ", "_") + '.json'
if 'path_output_map' not in st.session_state:
    st.session_state.path_output_map = path_scenario1_html + '/' + st.session_state.scenario1.replace(" ", "_")
    
scenario1 = st.selectbox('Select Scenarios of interest:', scenarios1_list, on_change=handle_click_no_button, key='scenario1_change')

scenario_names_l = ['Baseline', 'Best Case', 'Worst Case', 'Truck Type Sensitivity']
ind_sc = scenario_names_l.index(st.session_state.scenario1)

num_stations_small_l = [92, 50, 47, 55]
num_stations_medium_l = [69, 68, 66, 45]
num_stations_large_l = [148, 149, 137, 198]

num_stations_small = num_stations_small_l[ind_sc]
num_stations_medium = num_stations_medium_l[ind_sc]
num_stations_large = num_stations_large_l[ind_sc]
    
col11, col22, col33 = st.columns(3)

col11.subheader(":house: Small Stations")
col22.subheader(":office: Medium Stations")
col33.subheader(":european_castle: Large Station")

col11.metric("", '{:,.0f}'.format(num_stations_small))
col22.metric("", '{:,.0f}'.format(num_stations_medium))
col33.metric("", '{:,.0f}'.format(num_stations_large))    

with open(st.session_state.path_to_html,'r') as f: 
    html_data = f.read()

col1, col2 = st.columns([2,2])
with col1:
    st.subheader(":world_map: Density Map")
    image1 = Image.open(st.session_state.path_output_map + '_2030' + '.png')
    st.image(image1, caption='2030', width=400)
    
    image2 = Image.open(st.session_state.path_output_map + '_2040' + '.png')
    st.image(image2, caption='2040', width=400)
    
with col2:
    st.subheader(":fuelpump: H2 Stations Locations")
    st.components.v1.html(html_data, width=700, height=700)

