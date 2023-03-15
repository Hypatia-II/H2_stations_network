import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys
import os
import geopandas as gpd
import matplotlib.pyplot as plt


sys.path.append('../app/pages/utils')

path_conf = '../app/pages/utils/params/config_st.json'
conf = json.load(open(path_conf, "r"))
path_scenario5 = '../app/pages/utils/part4files/'

st.set_page_config(page_title="Hydrogen Production Sites", page_icon=":factory:", layout="wide")

st.markdown(
    """
    # 🏭 Hydrogen Production Sites
    
    This section pinpoints optimzed locations for hydrogen production sites based on the competitive scenarios .
    """
)

scenario_names = ['Air Liquide is the only player in the market', 
                  'Air Liquide and Red Team are entering simultaneously',
                  'Air Liquide enters after Red Team']

def name_scenario(name):

    if name=='Air Liquide is the only player in the market':
        sc = 'scenario1'
    elif name=='Air Liquide and Red Team are entering simultaneously':
        sc = 'scenario2'
    elif name=='Air Liquide enters after Red Team':
        sc = 'scenario3'
    return sc

def handle_click_no_button():
    if st.session_state['case_change']:
        st.session_state.scenario5_name = st.session_state.case_change
        st.session_state.scenario5 = name_scenario(st.session_state.scenario5_name)
        st.session_state.path_to_production_sites = path_scenario5 + "production_sites_" + st.session_state.scenario5 + ".csv" 
        st.session_state.path_to_plot_sc5 = path_scenario5 + "plot_" + st.session_state.scenario5 + ".jpg" 
        st.session_state.path_to_scenario_html = path_scenario5 + st.session_state.scenario5 + "part4.html" 

scenarios5_list = [sc.replace("production_sites_", "").replace(".csv","") for sc in os.listdir(path_scenario5) if sc.startswith("production_sites_")]

if 'scenario5_name' not in st.session_state:
    st.session_state.scenario5_name = scenario_names[0]
if 'scenario5' not in st.session_state:
    st.session_state.scenario5 = name_scenario(st.session_state.scenario5_name)
if 'path_to_production_sites' not in st.session_state:
    st.session_state.path_to_production_sites = path_scenario5 + "production_sites_" + st.session_state.scenario5 + ".csv"  
if 'path_to_plot_sc5' not in st.session_state:
    st.session_state.path_to_plot_sc5 = path_scenario5 + "plot_" + st.session_state.scenario5 + ".jpg" 
if 'path_to_scenario_html' not in st.session_state:
    st.session_state.path_to_scenario_html = path_scenario5 + st.session_state.scenario5 + "part4.html" 

df_production = pd.read_csv(st.session_state.path_to_production_sites)

scenario5 = st.selectbox('Select Case of interest:', scenario_names, on_change=handle_click_no_button, key='case_change')
    
with open(st.session_state.path_to_scenario_html,'r') as f: 
    html_data = f.read()

col1, col2 = st.columns([2,2])
with col1:
    st.header("Cluster Comparison")
    image1 = Image.open(st.session_state.path_to_plot_sc5)
    st.image(image1, caption='Clusters', width=700)
    
with col2:
    st.header("H2 Production Sites and Stations Locations")
    st.components.v1.html(html_data, width=700, height=700)