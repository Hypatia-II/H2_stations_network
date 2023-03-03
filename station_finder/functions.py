import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiLineString, Point, LineString
from shapely import ops
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class StationLocator():
    def __init__(self,
                 shapefiles: dict,
                 csvs: dict,
                 crs: str = '2154') -> None:
        """Create necessary datasets to calculate grid-search
        
        Args:
            shapefiles: dict containing all shapefiles loaded from Data()
            csvs: dict containting all csvs loaded from Data()
        
        """
        self.crs = crs
        # From shapefiles
        self.data = shapefiles['TMJA2019'].set_crs(self.crs)
        self.data = self.data.explode('geometry')
        self.data['PL_traffic'] = self.data['TMJA'] * (self.data['ratio_PL']/100)
        self.data['PL_traffic'] = (self.data['PL_traffic'] - self.data['PL_traffic'].min()) / \
                                    (self.data['PL_traffic'].max() - self.data['PL_traffic'].min())
    
        
        self.road_segments = self.data.geometry
        self.traffic_values = gpd.GeoDataFrame(self.data[['PL_traffic', 'geometry']], geometry='geometry').set_crs(self.crs)
        self.traffic_values['geometry'] = self.traffic_values.geometry.centroid
        
        self.air_logis = pd.concat([shapefiles['Aires_logistiques_elargies'], shapefiles['Aires_logistiques_denses']])
        
        # From csvs
        self.stations = csvs['pdv'].dropna(subset='latlng')
        self.stations[['lat', 'long']] = self.stations['latlng'].str.split(',', expand=True).astype(float)
        self.stations['geometry'] = self.stations.apply(lambda row: Point(row['long'], row['lat']), axis=1)
        self.stations = gpd.GeoDataFrame(self.stations[['id', 'typeroute', 'services', 'geometry']]).set_crs(self.crs)
        
        self.air_logis_info = csvs['aire_loqistique'].rename(columns={'Surface totale': 'surface_totale'})
        self.air_logis_info.columns.values[0] = 'e1'
        
        # Combined csv and shapefile
        self.air_logis = gpd.GeoDataFrame(pd.merge(self.air_logis_info[['e1', 'surface_totale']], self.air_logis, on='e1', how='inner')).set_crs(self.crs)
        self.air_logis['surface_totale'] = (self.air_logis['surface_totale'] - self.air_logis['surface_totale'].min()) / \
                                            (self.air_logis['surface_totale'].max() - self.air_logis['surface_totale'].min())
        self.air_logis['geometry'] = self.air_logis.geometry.centroid
    

    def create_network(self, 
                       road_segments: list[object]) -> MultiLineString:
        """Combine all geometric segments into one large MultiLineString
        
        Args:
            road_segments: list of linestrings
            
        Returns:
            network: combined linestrings into multilinestring
            
        """
        segments = [MultiLineString([segment]) if isinstance(segment, LineString) else segment for segment in road_segments]
        network = ops.unary_union(segments)
        
        return network
    
   
    def score(self, 
              candidate: Point, 
              road_network: MultiLineString) -> float:
        """Compute score for candidate location
        
        Args:
            candidate: geometric Point
            networks: MultiLineString to compare to candidate
            weights: traffic values and other values to include
            
        Returns:
            score: cumulative score
            
        """
        max_road = 500 # Maximum distance for traffic & roads
        max_distance = 10_000 # Maximum distance to consider for  aires
        
        proximity_weight = 2 # weight for proximity score
        traffic_weight = 5 # weight for traffic score
        aires_weight = 10 # weight for the aires logistique
        score = 0
        
        # Calculating road distance
        for i, network in enumerate(road_network):
            distance = candidate.distance(network)
            
            proximity_score = 0.0
            if distance <= max_road/2:
                proximity_score = (max_road - distance) / max_road
            elif distance <= max_road:
                proximity_score = (max_road - distance) / max_road / 2
            else:
                continue 
                       
            score += proximity_weight * proximity_score

        # Calculating traffic nearby
        for i, point in enumerate(self.traffic_values.geometry):
            distance = candidate.distance(point)
            
            traffic_score = 0.0
            if distance <= max_road/2:
                traffic_score = self.traffic_values.PL_traffic[i]
            if distance <= max_road:
                traffic_score = self.traffic_values.PL_traffic[i] / 2
            else:
                continue
            
            score += traffic_score * traffic_weight
        
        # Calculating proximity to logistic centers
        for i, point in enumerate(self.air_logis.geometry):
            distance = candidate.distance(point)
            
            aires_score = 0.0
            if distance <= max_distance/2:
                aires_score = (max_distance - distance) / max_distance
                aires_score += self.air_logis.surface_totale[i]
            elif distance <= max_distance:
                aires_score = (max_distance - distance) / max_distance / 2
                aires_score += self.air_logis.surface_totale[i] / 2
            else: 
                continue
            
            score += aires_score * aires_weight

        return score


    def score_faster(self, 
                     candidate: Point, 
                     road_network: MultiLineString) -> float:
        
        max_road = 500
        max_distance = 10_000
        proximity_weight = 2
        traffic_weight = 5
        aires_weight = 10
        score = 0
        
        # Pre-compute distances between candidate and network vertices
        network_distances = np.array([candidate.distance(network) for network in road_network])
        
        # Compute proximity scores
        proximity_scores = np.zeros_like(network_distances)
        proximity_scores[network_distances <= max_road / 2] = (max_road - network_distances[network_distances <= max_road / 2]) / max_road
        proximity_scores[(max_road / 2 < network_distances) & (network_distances <= max_road)] = (max_road - network_distances[(max_road / 2 < network_distances) & (network_distances <= max_road)]) / max_road / 2
        
        # Add proximity scores to total score
        score += np.sum(proximity_scores) * proximity_weight
        
        # Compute traffic scores
        traffic_distances = np.array([candidate.distance(point) for point in self.traffic_values.geometry])
        traffic_scores = np.zeros_like(traffic_distances)
        traffic_scores[traffic_distances <= max_road / 2] = self.traffic_values.PL_traffic[traffic_distances <= max_road / 2]
        traffic_scores[(max_road / 2 < traffic_distances) & (traffic_distances <= max_road)] = self.traffic_values.PL_traffic[(max_road / 2 < traffic_distances) & (traffic_distances <= max_road)] / 2
        
        # Add traffic scores to total score
        score += np.sum(traffic_scores) * traffic_weight
        
        # Compute aires scores
        aires_distances = np.array([candidate.distance(point) for point in self.air_logis.geometry])
        aires_scores = np.zeros_like(aires_distances)
        aires_scores[aires_distances <= max_distance / 2] = (max_distance - aires_distances[aires_distances <= max_distance / 2]) / max_distance + self.air_logis.surface_totale[aires_distances <= max_distance / 2]
        aires_scores[(max_distance / 2 < aires_distances) & (aires_distances <= max_distance)] = (max_distance - aires_distances[(max_distance / 2 < aires_distances) & (aires_distances <= max_distance)]) / max_distance / 2 + self.air_logis.surface_totale[(max_distance / 2 < aires_distances) & (aires_distances <= max_distance)] / 2
        
        # Add aires scores to total score
        score += np.sum(aires_scores) * aires_weight
        
        return score

    def grid_searcher(self,
                      grid_size: int = 100_000,
                      num_locations: int = 10) -> list:
        """Identify top X locations on map based on pre-defined parameters
        
        Args:
            grid_size = distance between points on map, in meters
            num_locations = number of top locations to be returned
            
        Returns:
            sorted_locations: coordinates, weighted_score of top X locations
        """
              
        network = self.create_network(self.road_segments)
        # creating the boundary of our grid
        xmin, ymin, xmax, ymax = network.bounds
        x_coords = np.arange(xmin, xmax + grid_size, grid_size)
        y_coords = np.arange(ymin, ymax + grid_size, grid_size)
        
        # setting up the grid points
        grid_points = np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])
        candidate_locations = [Point(x, y) for x, y in grid_points]

        # defining the grid weights
        #weights = [float(tv) / sum(self.traffic_values) for tv in self.traffic_values]
        
        # computing the weighted distances in each grid. Not finished, the network size is larger than the weight size and will need to be adjusted
        weighted_distances  = [self.score_faster(candidate, network) for candidate in tqdm(candidate_locations)]
            

        sorted_locations = sorted(zip(candidate_locations, weighted_distances), key=lambda x: x[1], reverse=True)[:num_locations]
        
        return sorted_locations
    
    def visualize_results(self,
                          sorted_locations: list,
                          colors: list[str] = None) -> None:
        """Visualize top locations on map
        
        Args:
            sorted_locations: list of coordinates, weighted score of locations
            colors: list of colors for traffic heatmap
            
        """
        
        france_center = [46.2276, 2.2137]
        m = folium.Map(location=france_center, zoom_start=6, tiles='cartodbpositron')

        values = np.quantile(self.data['PL_traffic'], [np.linspace(0, 1, 7)])
        values = values[0]
        if colors is None:
            colors = ['#00ae53', '#86dc76', '#daf8aa', '#ffe6a4', '#ff9a61', '#ee0028']
            
        colormap_dept = cm.StepColormap(colors=colors,
                                        vmin=min(self.data['PL_traffic']),
                                        vmax=max(self.data['PL_traffic']),
                                        index=values)

        style_function = lambda x: {'color': colormap_dept(x['properties']['PL_traffic']),
                                    'weight': 2.5,
                                    'fillOpacity': 1}
        
        geojson = folium.GeoJson(self.data,
                                 name='Routes',
                                 style_function=style_function
                                )
        grid = folium.GeoJson(gpd.GeoDataFrame(sorted_locations, geometry=0).set_crs(self.crs),
                              style_function=lambda x: {'color': 'red',
                                                        'weight': 2}
                              )
        geojson.add_to(m)
        grid.add_to(m)
        
        m.save('map.html')

