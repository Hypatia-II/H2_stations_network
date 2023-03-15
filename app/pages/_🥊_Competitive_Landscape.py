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
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import pyproj
from plotly.subplots import make_subplots
from shapely import wkt
import folium
import branca.colormap as cm
from streamlit_folium import st_folium



sys.path.append('../app/pages/utils')
import functions_st

path_conf = '../app/pages/utils/params/config_st.json'
conf = json.load(open(path_conf, "r"))
path_scenario4 = '../app/pages/utils/cases_csv_part3/'

colors = ['#2C3E50',
          '#C0392B',
          '#E74C3C',
          '#9B59B6',
          '#8E44AD',
          '#2980B9',
          '#3498DB',
          '#27AE60',
          '#F1C40F',
          '#E67E22',
          '#D35400']
france_center = [47.0276, 1.687]


st.set_page_config(page_title="Competitive Influence on Station Locations", page_icon=":detective:", layout="wide")

st.markdown(
    """
    # 📍 Competitive Influence on Station Locations
    
    This section is to help you detect churners, based on a specific period of time. The output will be a list of the clients with their client id, and potentially personal information.
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
        st.session_state.scenario4_name = st.session_state.case_change
        st.session_state.scenario4 = name_scenario(st.session_state.scenario4_name)
        st.session_state.path_to_cost_yearly = path_scenario4 + "cost_yearly_" + st.session_state.scenario4 + ".csv" 
        st.session_state.path_to_final_points = path_scenario4 + "final_points_" + st.session_state.scenario4 + ".csv" 

scenarios4_list = [sc.replace("cost_yearly_", "").replace(".csv","") for sc in os.listdir(path_scenario4) if sc.startswith("cost_yearly_")]

if 'scenario4_name' not in st.session_state:
    st.session_state.scenario4_name = scenario_names[0]
if 'scenario4' not in st.session_state:
    st.session_state.scenario4 = name_scenario(st.session_state.scenario4_name)
if 'path_to_cost_yearly' not in st.session_state:
    st.session_state.path_to_cost_yearly = path_scenario4 + "cost_yearly_" + st.session_state.scenario4 + ".csv"  
if 'path_to_final_points' not in st.session_state:
    st.session_state.path_to_final_points = path_scenario4 + "final_points_" + st.session_state.scenario4 + ".csv"
    
df_cost = pd.read_csv(st.session_state.path_to_cost_yearly)
df_final_points = pd.read_csv(st.session_state.path_to_final_points)
df_cost.drop('Unnamed: 0', axis=1, inplace=True)
df_final_points.drop('Unnamed: 0', axis=1, inplace=True)
df_final_points['points'] = df_final_points['points'].apply(wkt.loads)

df_points = gpd.GeoDataFrame(df_final_points, geometry=df_final_points.points).set_crs('2154')
df_points = df_points[~ df_points.year.isna()]
df_points['year'].astype(int)
df_points.reset_index(inplace=True)

df_bar = pd.DataFrame(index=df_points['year'].value_counts().index.astype(int))
df_bar['Count'] = df_points['year'].value_counts().values
df_bar['year'] = df_bar.index
df_bar.reset_index(inplace=True, drop=True)

scenario4 = st.selectbox('Select Case of interest:', scenario_names, on_change=handle_click_no_button, key='case_change')
    
    
m = folium.Map(location=france_center, zoom_start=6, tiles='cartodbpositron')
for idx, year_i in enumerate(set(df_points.year.astype(int))):
    type_color = colors[idx]
    df_p = df_points[df_points.year==year_i]
    geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in df_p.geometry]

    top_2030 = folium.GeoJson(df_p.geometry,
                              marker = folium.CircleMarker(
                                      radius = 4,
                                      weight = 0,
                                      fill_color = colors[idx], 
                                      fill_opacity = 0.8)                            
                                  ).add_to(m)

colormap = cm.LinearColormap(colors=colors,
                             index=set(df_points.year.astype(int)), vmin=int(2030), vmax=int(2040),
                             caption='Year of Implementation',
                             ).add_to(m)

chart_1 = alt.Chart(df_bar).mark_bar(color="#D8FAD9").encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('Count:Q', title='Count')
).properties(
    width=600
)

line_1 = chart_1.mark_line(color='#5DB44C').encode(
    x='year:O',
    y='Count:Q'
)

chart_2 = alt.Chart(df_cost).mark_bar(color="#D8FAD9").encode(
    x=alt.X('year:O', title='Year'),
    y=alt.Y('cumsumwithgrowth_profit:Q', title='Cumulative Sum with Growth Profit')
).properties(
    width=600
)

line_2 = chart_2.mark_line(color='#5DB44C').encode(
    x='year:O',
    y='cumsumwithgrowth_profit:Q'
)
profitable_year = df_cost.loc[df_cost["total_profit"].gt(0).idxmax(),'year']
reimbursed_year = df_cost.loc[df_cost["total_profit_cumsum"].gt(0).idxmax(),'year']
profitable_tot_profit = int(df_cost.loc[df_cost["total_profit"].gt(0).idxmax(),'total_profit'])
reimbursed_tot_profit_cumsum = int(df_cost.loc[df_cost["total_profit_cumsum"].gt(0).idxmax(),'total_profit_cumsum'])

col11, col22, col33, col44 = st.columns(4)
col11.subheader(":clock3: :white[2030]")
col11.metric("", profitable_year)#, delta=st.session_state.delta1)
col22.metric("", profitable_tot_profit)#, delta=st.session_state.delta1)
col33.subheader(":hourglass_flowing_sand: :white[2040]")
col33.metric("", reimbursed_year)#, delta=st.session_state.delta2)
col44.metric("", reimbursed_tot_profit_cumsum)#, delta=st.session_state.delta2)

col1, col2 = st.columns([2,2])
with col1:
    if st.session_state.scenario4=='scenario1':
        str_sc = "**Worst Case scenario**"
        st.write("Count of number of stations built per year - " + str_sc)
    if st.session_state.scenario4=='scenario2':
        str_sc = "**Baseline Scenario**"
        st.write("Count of number of stations built per year - " + str_sc)
    if st.session_state.scenario4=='scenario3':
        str_sc = "yo"
        st.write("Count of number of stations built per year - " + str_sc)
    st.write(chart_1 + line_1)
        
    st.write("Operational profit per year using a price of **5 € per kgH₂**")
    st.write(chart_2 + line_2)
with col2:
    st_folium(m, width=600)