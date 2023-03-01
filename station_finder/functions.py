import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from load_preprocess.functions import Data

path = 'data/'
data = Data(path)
shapefiles = data.get_shapefiles()

traffic = shapefiles['TMJA2019'].set_crs('2154')
traffic['PL_traffic'] = traffic['TMJA'] * (traffic['ratio_PL']/100)
data = traffic

G = nx.Graph()

for index, row in data.iterrows():
    G.add_node(index, geometry=row['geometry'], PL_traffic=row['PL_traffic'])
    if index > 0:
        G.add_edge(index-1, index)

betweenness = nx.betweenness_centrality(G, weight='PL_traffic')

best_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:20]

# Print the best locations for hydrogen stations
print('Best locations for hydrogen stations:')
for node in best_nodes:
    index = node[0]
    location = data.iloc[index]['geometry'].centroid
    print(f'Location {index+1}: ({location.x}, {location.y})')
    
