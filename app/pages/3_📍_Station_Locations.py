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
import functions_st

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
print(st.session_state.path_to_html)
# df_show = df.copy()
    
# path_to_html = path_scenario_html + "/output_" + scenario1.replace(" ", "_") + ".html" 
with open(st.session_state.path_to_html,'r') as f: 
    html_data = f.read()

# col1, col2 = st.columns(2)
# with col1:
#     st.header("Density Map 2030")
#     image1 = Image.open(st.session_state.path_output_map + '_2030' + '.png')
#     st.image(image1, caption='2030')
# with col2:
#     st.header("Density Map 2040")
#     image2 = Image.open(st.session_state.path_output_map + '_2040' + '.png')
#     st.image(image2, caption='2040')
        
# st.header("H2 Stations Locations")
# st.components.v1.html(html_data, width=700, height=700)

col1, col2 = st.columns([2,2])
with col1:
    st.header("Density Map")
    image1 = Image.open(st.session_state.path_output_map + '_2030' + '.png')
    st.image(image1, caption='2030', width=400)
    
    image2 = Image.open(st.session_state.path_output_map + '_2040' + '.png')
    st.image(image2, caption='2040', width=400)
    
with col2:
    st.header("H2 Stations Locations")
    st.components.v1.html(html_data, width=700, height=700)

# conf_o_sc = json.load(open(st.session_state.path_output_sc, "r"))
# df = sum(list(conf_o_sc['num_stations_2030'].values())), sum(list(conf_o_sc['num_stations_2040'].values()))



# regions_pd_sh = gpd.read_file('../data/FRA_adm1.shp')
# regions_pd_sh = regions_pd_sh[['NAME_1', 'geometry']]
# regions_pd_sh.columns = ['region', 'geometry']
# regions_pd_sh.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

# path_output_sc = '../data/output_Scenario_' + st.session_state.scenario1.replace(" ", "_") + '.json'
# conf_o_sc = json.load(open(path_output_sc, "r"))

# col_names = list(conf_o_sc.keys())
# index_list = list(conf_o_sc[col_names[0]].keys())
# df = pd.DataFrame(columns=col_names, index=index_list)

# df.loc[:,col_names[0]] = list(conf_o_sc[col_names[0]].values())
# df.loc[:,col_names[1]] = list(conf_o_sc[col_names[1]].values())
# df['region'] = df.index
# df.reset_index(inplace=True, drop=True)
# # df.set_index('region', inplace=True)
# df_f = regions_pd_sh.set_index('region').join(df.set_index('region'), on='region', how='left')
# df_f.dropna(inplace=True)

# fig1 = px.choropleth(df_f,
#                    geojson=df_f.geometry,
#                    locations=df_f.index,
#                    color="num_stations_2030",
#                    height=600,
#                    width=600,
#                    range_color=(0, 50),
#                    color_continuous_scale='GnBu',
#                    scope="europe",
#                    labels={'num_stations_2030':'Number of Stations'})
# fig1.update_geos(fitbounds="geojson", visible=False)
# fig1.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
#                   plot_bgcolor = "rgba(0,0,0,0)")
# # st.plotly_chart(fig1)

# fig2 = px.choropleth(df_f,
#                    geojson=df_f.geometry,
#                    locations=df_f.index,
#                    color="num_stations_2030",
#                    height=600,
#                    width=600,
#                    range_color=(0, 50),
#                    color_continuous_scale='GnBu',
#                    scope="europe",
#                    labels={'num_stations_2030':'Number of Stations'})
# fig2.update_geos(fitbounds="geojson", visible=False)
# fig2.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
#                   plot_bgcolor = "rgba(0,0,0,0)")
# # st.plotly_chart(fig2)

# col1, col2 = st.columns(2, gap="small")
# with col1:
#     st.header("H2 Stations Density Map")
#     st.plotly_chart(fig1)
#     st.plotly_chart(fig2)
    
# with col2:
#     st.header("H2 Stations Locations")
#     st.components.v1.html(html_data, width=700, height=700)
    
    
    
    
    
    
# path_output_sc = '../data/output_Scenario_' + ("Best Case").replace(" ", "_") + '.json'
# conf_o_sc = json.load(open(path_output_sc, "r"))
# regions_pd_sh = gpd.read_file('../data/FRA_adm1.shp')
# regions_pd_sh = regions_pd_sh[['NAME_1', 'geometry']]
# regions_pd_sh.columns = ['region', 'geometry']
# regions_pd_sh.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

# col_names = list(conf_o_sc.keys())
# index_list = list(conf_o_sc[col_names[0]].keys())
# df = pd.DataFrame(columns=col_names, index=index_list)
# df.loc[:,col_names[0]] = list(conf_o_sc[col_names[0]].values())
# df.loc[:,col_names[1]] = list(conf_o_sc[col_names[1]].values())
# df['region'] = df.index
# df.reset_index(inplace=True, drop=True)
# df_f = regions_pd_sh.merge(df, on='region', how='left')
# df_f.dropna(inplace=True)

# choro_json = json.loads(df_f.to_json())
# choro_data = alt.Data(values=choro_json['features'])

# df_f['centroid_lon'] = df_f['geometry'].centroid.x
# df_f['centroid_lat'] = df_f['geometry'].centroid.y

# def gen_map(geodata, color_column, title):
#     '''Generates DC ANC map with population choropleth and ANC labels'''
#     # Add Base Layer
#     base = alt.Chart(geodata, title = title).mark_geoshape(
#         stroke='black',
#         strokeWidth=1
#     ).encode(
#     ).properties(
#         width=400,
#         height=400
#     )
#     # Add Choropleth Layer
#     choro = alt.Chart(geodata).mark_geoshape(
#         fill='lightgray',
#         stroke='black'
#     ).encode(
#         alt.Color(color_column, 
#                   type='quantitative', 
#                   scale=alt.Scale(scheme='bluegreen'),
#                   title = "DC Population")
#     )
#     # Add Labels Layer
#     labels = alt.Chart(geodata).mark_text(baseline='top'
#      ).properties(
#         width=400,
#         height=400
#      ).encode(
#          longitude='properties.centroid_lon:Q',
#          latitude='properties.centroid_lat:Q',
#         #  text='properties.ANC_ID:O',
#          size=alt.value(8),
#          opacity=alt.value(1)
#      )

#     return base + choro + labels

# pop_2000_map = gen_map(geodata=choro_data, color_column='properties.num_stations_2030', title='2030')
# st.altair_chart(pop_2000_map)