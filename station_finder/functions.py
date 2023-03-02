from ast import Mult
from venv import create
from shapely.geometry import MultiLineString, Point, LineString
from shapely import ops
import numpy as np
from tqdm import tqdm
sys.path.append('..')
from load_preprocess.functions import Data

class StationLocator():
    def __init__(self,
                 shapefiles: dict) -> None:
        self.data = shapefiles['TMJA2019'].set_crs('2154')
        self.data['PL_traffic'] = self.data['TMJA'] * (self.data['ratio_PL']/100)
        
        self.road_segments = self.data.geometry
        self.traffic_values = self.data.PL_traffic
    
    
    def create_network(self, 
                       road_segments: list[object]) -> MultiLineString:
        segments = [MultiLineString([segment]) if isinstance(segment, LineString) else segment for segment in road_segments]
        network = ops.unary_union(segments)
        
        return network
    
    @classmethod
    def grid_searcher(cls,
                      grid_size: int = 100_000,
                      num_locations: int = 10) -> list:        
        network = cls.create_network(cls.road_segments)
        # creating the boundary of our grid
        xmin, ymin, xmax, ymax = network.bounds
        x_coords = np.arange(xmin, xmax + grid_size, grid_size)
        y_coords = np.arange(ymin, ymax + grid_size, grid_size)
        
        # setting up the grid points
        grid_points = np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])
        candidate_locations = [Point(x, y) for x, y in grid_points]

        # defining the grid weights
        weights = [float(tv) / sum(cls.traffic_values) for tv in cls.traffic_values]
        
        # computing the weighted distances in each grid. Not finished, the network size is larger than the weight size and will need to be adjusted
        weighted_distances = [sum([weights[i] * candidate.distance(network) for i, network in enumerate(network.geoms[0:4695])]) for candidate in tqdm(candidate_locations)]

        sorted_locations = sorted(zip(candidate_locations, weighted_distances), key=lambda x: x[1])[:num_locations]
        
        return sorted_locations
    