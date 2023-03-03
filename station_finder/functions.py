import branca.colormap as cm
import folium
import geopandas as gpd
from shapely.geometry import MultiLineString, Point, LineString
from shapely import ops
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

class StationLocator():
    def __init__(self,
                 shapefiles: dict) -> None:
        """Create necessary datasets to calculate grid-search
        
        Args:
            shapefiles: dict containing all shapefiles loaded from Data()
        
        """
        
        self.data = shapefiles['TMJA2019'].set_crs('2154')
        self.data['PL_traffic'] = self.data['TMJA'] * (self.data['ratio_PL']/100)
        
        self.road_segments = self.data.geometry
        self.traffic_values = self.data.PL_traffic
    
    
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
              networks: MultiLineString, 
              weights: list) -> float:
        """Compute score for candidate location
        
        Args:
            candidate: geometric Point
            networks: MultiLineString to compare to candidate
            weights: traffic values and other values to include
            
        Returns:
            score: cumulative score
            
        """
        
        max_distance = 250 # maximum distance to consider for roads
        proximity_weight = 2 # weight for proximity score
        traffic_weight = 10 # weight for traffic score
        score = 0
        
        for i, network in enumerate(networks.geoms[:4695]):
            distance = candidate.distance(network)
            
            # Only consider roads within 0-75 km range
            if distance <= 75:
                proximity_score = (max_distance - distance) / max_distance
            elif distance <= max_distance:
                proximity_score = (max_distance - distance) / max_distance / 2
            else:
                continue # skip roads beyond 250 km range
            
            traffic_score = 1 / (weights[i] + 1) ** 2
            
            score += proximity_weight * proximity_score + traffic_weight * traffic_score
        
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
        weights = [float(tv) / sum(self.traffic_values) for tv in self.traffic_values]
        
        # computing the weighted distances in each grid. Not finished, the network size is larger than the weight size and will need to be adjusted
        weighted_distances  = [self.score(candidate, network, weights) for candidate in tqdm(candidate_locations)]
            

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
        grid = folium.GeoJson(gpd.GeoDataFrame(sorted_locations, geometry=0).set_crs('2154'),
                              style_function=lambda x: {'color': 'red',
                                                        'weight': 2}
                              )
        geojson.add_to(m)
        grid.add_to(m)
        
        m.save('map.html')
