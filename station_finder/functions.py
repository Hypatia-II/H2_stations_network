import branca.colormap as cm
import json
import folium
import geopandas as gpd
from itertools import chain
import numpy as np
from shapely.geometry import MultiLineString, Point, LineString, Polygon
from shapely import ops
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Optional

import warnings
warnings.filterwarnings("ignore")


class WeightedLineString(LineString):
    
    """Add weight to regular shapely LineString
    
    Args:
        weight: traffic value
    """
    def __init__(self, *args, **kwargs):
        self.weight = kwargs.pop('weight', 0.0)
        super().__init__(*args, **kwargs)
        
class WeightedMultiLineString(MultiLineString):
    
    """Add weight to shapely MultiLineString
    
    Args:
        weight: traffic value
    """
    def __init__(self, lines, weights=None, **kwargs):
        if weights is None:
            weights = [0.0] * len(lines)
        self.lines = [WeightedLineString(line, weight=weight) for line, weight in zip(lines, weights)]
        super().__init__(self.lines, **kwargs)

    @property
    def weights(self):
        return [line.weight for line in self.lines]

class StationLocator():
    def __init__(self,
                 shapefiles: dict,
                 csvs: dict,
                 crs: str = '2154') -> None:
        
        """Create datasets to calculate grid-search
        
        Args:
            shapefiles: dict containing all shapefiles loaded from Data()
            csvs: dict containting all csvs loaded from Data()
            crs: EPSG rules to follow for geometric data, default is RGF93 v1 / Lambert-93
        
        """
        # init global variables
        self.crs = crs
        
        ## Loading road and traffic data
        self.data = shapefiles['TMJA2019'].set_crs(self.crs)
        self.data = self.data.explode('geometry')
        # Clean ratio_PL
        self.data['ratio_PL'] = np.where(self.data['ratio_PL'] > 40, self.data['ratio_PL'] / 10, self.data['ratio_PL'])
        # defining traffic values
        self.data['PL_traffic'] = self.data['TMJA'] * (self.data['ratio_PL']/100)
        self.data['PL_traffic'] = (self.data['PL_traffic'] - self.data['PL_traffic'].min()) / \
                                    (self.data['PL_traffic'].max() - self.data['PL_traffic'].min())
        self.road_segments = self.data.geometry
        self.traffic_only = self.data.PL_traffic

        # Loading gas station data
        self.stations = csvs['pdv'].dropna(subset=['latlng'])
        self.stations[['lat', 'long']] = self.stations['latlng'].str.split(',', expand=True).astype(float)
        self.stations['geometry'] = self.stations.apply(lambda row: Point(row['long'], row['lat']), axis=1)
        self.stations = gpd.GeoDataFrame(self.stations[['id', 'typeroute', 'services', 'geometry']]).set_crs(self.crs)
        
        ## Loading production hub data
        self.air_logis = pd.concat([shapefiles['Aires_logistiques_elargies'], shapefiles['Aires_logistiques_denses']])
        self.air_logis_info = csvs['aire_loqistique'].rename(columns={'Surface totale': 'surface_totale'})
        self.air_logis_info.columns.values[0] = 'e1'
        self.air_logis = gpd.GeoDataFrame(pd.merge(self.air_logis_info[['e1', 'surface_totale']], self.air_logis, on='e1', how='inner')).set_crs(self.crs)
        self.air_logis['surface_totale'] = (self.air_logis['surface_totale'] - self.air_logis['surface_totale'].min()) / \
                                            (self.air_logis['surface_totale'].max() - self.air_logis['surface_totale'].min())
        self.air_logis['geometry'] = self.air_logis.geometry.centroid
        
        ## Region & departments
        self.regions = gpd.GeoDataFrame(shapefiles['FRA_adm1']).to_crs(self.crs)
    
    def create_network(self,
                       road_segments: list[object], 
                       traffic_values: list[float]) -> MultiLineString:
        
        """Combine all geometric segments into one large MultiLineString with custom weights
        
        Args:
            road_segments: list of linestrings & multilinestrings
            traffic_values: list of traffic values for each segment
        Returns:
            network: combined linestrings into multilinestring with custom weights
        """
        segments = []
        for segment, traffic in zip(road_segments, traffic_values):
            if isinstance(segment, WeightedLineString):
                segment_with_weight = segment
            elif isinstance(segment, LineString):
                segment_with_weight = WeightedLineString(segment.coords, weight=traffic)
            elif isinstance(segment, MultiLineString):
                sub_segments_with_weight = []
                for sub_segment in segment.geoms:
                    if isinstance(sub_segment, WeightedLineString):
                        sub_segment_with_weight = sub_segment
                    elif isinstance(sub_segment, LineString):
                        sub_segment_with_weight = WeightedLineString(sub_segment.coords, weight=traffic)
                    sub_segments_with_weight.append(sub_segment_with_weight)
                segment_with_weight = WeightedMultiLineString(sub_segments_with_weight, weight=traffic)
            segments.append(segment_with_weight)
        
        network = ops.unary_union(segments)
        return network

    def score_locations(self,
                        candidate: Point, 
                        road_network: MultiLineString,
                        gas_stations: bool = False) -> float:
        
        """Compute score for candidate location
        
        Args:
            candidate: geometric Point
            networks: road network with coordinates and weights
            gas_stations: include gas station locations into calculation
                        
        Returns:
            score: cumulative score
            
        """
        max_road = 500 # Maximum distance for traffic & roads
        max_distance = 10_000 # Maximum distance to consider for aires
        
        proximity_weight = 2 # weight for proximity score
        traffic_weight = 5 # weight for traffic score
        aires_weight = 10 # weight for the aires logistique
        station_weight = -2 # weight for gas stations
        
        score = 0
        
        # Calculating road distance & traffic 
        for i, network in enumerate(road_network):
            distance = candidate.distance(network)

            proximity_score = 0.0
            if distance <= max_road/2:
                proximity_score = (max_road - distance) / max_road
                if isinstance(network, WeightedLineString):
                    traffic_score = network.weight
                elif isinstance(network, MultiLineString):
                    traffic_score = np.mean([line.weight for line in network])
                else:
                    continue
            elif distance <= max_road:
                proximity_score = (max_road - distance) / max_road / 2
                if isinstance(network, WeightedLineString):
                    traffic_score = network.weight / 2
                elif isinstance(network, MultiLineString):
                    traffic_score = np.mean([line.weight for line in network]) / 2
                else:
                    continue
            else:
                continue           
            score += proximity_weight * proximity_score + traffic_score * traffic_weight
        
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

        if gas_stations:
            # Calculating proximity to existing gas stations
            for i, station in enumerate(self.stations.geometry):
                distance = candidate.distance(station)
                
                station_score = 0.0
                if distance <= max_distance/2:
                    station_score = (max_distance - distance) / max_distance
                elif distance <= max_distance:
                    station_score = (max_distance - distance) / max_distance / 2
                else:
                    continue
                score += station_score * station_weight    
                
        return score
    
    def get_best_location(self,
                          grid_size: int = 100_000,
                          gas_stations: bool = False,
                          candidate_locations: pd.DataFrame = None,
                          save_file: bool = False) -> list:
        
        """Identify top X locations on map based on pre-defined parameters
        
        Args:
            grid_size = distance between points on map, in meters
            gas_stations = whether to include gas_stations into calculations
            candidate_location = predefined locations to calculate scores for
            
        Returns:
            sorted_locations: coordinates, weighted_score of top X locations
        """
        network = self.create_network(self.road_segments, self.traffic_only)

        if candidate_locations is None:
            # creating the boundary of our grid
            xmin, ymin, xmax, ymax = network.bounds
            x_coords = np.arange(xmin, xmax + grid_size, grid_size)
            y_coords = np.arange(ymin, ymax + grid_size, grid_size)
            
            # setting up the grid points
            grid_points = np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])
            candidate_locations = [Point(x, y) for x, y in grid_points]
        else:
            candidate_locations = candidate_locations.geometry
        
        weighted_scores  = [self.score_locations(candidate, network, gas_stations=gas_stations) for candidate in tqdm(candidate_locations)]
            
        sorted_locations = sorted(zip(candidate_locations, weighted_scores), key=lambda x: x[1], reverse=True)
        
        if save_file:
            with open('sorted_locations.pkl', 'wb') as f:
                pickle.dump(sorted_locations, f)
                
        return sorted_locations
    
    def visualize_results(self,
                          sorted_locations: list,
                          num_locations: int = 25,
                          colors: list[str] = None,
                          filename: str = 'map.html') -> None:
        
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
        
        roads = folium.GeoJson(self.data,
                                 name='Routes',
                                 style_function=style_function
                                )
        top_locations = folium.GeoJson(gpd.GeoDataFrame(sorted_locations[:num_locations], geometry=0).set_crs(self.crs),
                              style_function=lambda x: {'color': 'red',
                                                        'weight': 2}
                              )
        
        roads.add_to(m)
        top_locations.add_to(m)
        
        m.save(filename)
        
class Scenarios(StationLocator):
    def __init__(self, 
                 shapefiles: dict, 
                 csvs: dict,
                 jsons: dict, 
                 path_conf: str = 'params/config.json',
                 crs: str = '2154') -> None:
        super().__init__(shapefiles, csvs, crs)
        self.cost_profit = jsons['cost_profit']
        self.conf = json.load(open(path_conf, "r"))

    def distribute_locations(self, 
                             sorted_locations: list[object],
                             region_breakdown: dict) -> pd.DataFrame():
        
        """Distribute hydrogen station location into regions according to pre-defined metrics
        
        Args:
            sorted_locations: list of 'best' locations with Point and score
            region_breakdown: dict of regions and how many hydrogen stations will be needed for each
            
        Returns:
            top_points_by_region: Top X points for each region, X being region specific
        """
        
        sorted_locations = gpd.GeoDataFrame(sorted_locations, geometry=0)
        region_locations = gpd.sjoin(sorted_locations, self.regions, how='inner')

        top_points_by_region = pd.DataFrame()
        for region, stations_num in region_breakdown.items():
            region_data = region_locations[region_locations['NAME_1'] == region]
            top_points = region_data.sort_values(by=1, ascending=False).head(stations_num)
            top_points_by_region = top_points_by_region.append(top_points)
            
        return top_points_by_region
    
    def merge_closest_points(self, top_locations: gpd.GeoDataFrame, distance_min: int=10_000):
        """Merge close points into one station.

        Args:
            top_locations: geodataframe with top locations selected by model and their scores.
        Returns:
            polygones: final list of points with their score and number of merged points.
        """
        top_locations = list(zip(top_locations[0], top_locations[1]))
        distances = {}
        for i in range(len(top_locations)):
            distances.setdefault(i, [])
            for j in range(len(top_locations)):
                if top_locations[i][0].distance(top_locations[j][0]) <= distance_min:
                    distance = top_locations[i][0].distance(top_locations[j][0])
                    distances[i].append((top_locations[j][0].xy[0][0], top_locations[j][0].xy[1][0]))
        
        for key, values in distances.items():
            distances[key] = (values, len(values))
        
        distances = {k:v[0] for k, v in sorted(distances.items(), key=lambda item: item[1][1], reverse=True)}

        distances_reduced = {}
        distances_val = {}
        for i in range(len(distances)):
            if set(distances[i]).isdisjoint(set(list(chain(*list(distances_val.values()))))):
                distances_val[i] = distances[i]
                distances_reduced[i] = [Point(xy) for xy in distances[i]]
        
        polygones = []
        for key, values in distances_reduced.items():
            if len(values) == 1:
                point = values[0]
            elif len(values) == 2:
                line = LineString([values[0], values[1]])
                point = line.centroid
            else:
                point = Polygon(values).centroid
            avg_score = np.mean([item[1] for item in top_locations if item[0] in values])
            #if not any(p.equals(point) for p, _ in polygones):
            polygones.append((point, avg_score, len(values)))
        
        return polygones

    def nearest_part_of_linestrings(self, 
                                    lines: list[object], 
                                    point: Point) -> Point:
        """Find nearest point of linestring or intersection to candidate location
        
        Args:
            lines: road network
            point: candidate location Point
            
        Returns:
            nearest_line: new coordinates for nearest point of road to candidate location
        """
        
        min_distance = float('inf')
        nearest_line = None
        for line in lines:
            distance = line.distance(point)
            if distance < min_distance:
                min_distance = distance
                nearest_line = line
        
        # check if there is an intersection within 3x the distance to the nearest road
        intersection_distance = min_distance * 3
        for line in lines:
            if line == nearest_line:
                continue
            intersection = nearest_line.intersection(line)
            if intersection.geom_type == 'Point':
                intersection_distance_to_point = intersection.distance(point)
                if intersection_distance_to_point <= intersection_distance:
                    nearest_line = line
                    min_distance = intersection_distance_to_point
                    point = intersection
                    intersection_distance = min_distance * 2
        
        min_distance = float('inf')
        nearest_vertex = None
        for i in range(len(nearest_line.coords)):
            vertex = nearest_line.coords[i]
            vertex_point = Point(vertex)
            distance = vertex_point.distance(point)
            if distance < min_distance:
                min_distance = distance
                nearest_vertex = i
        return nearest_line.coords[nearest_vertex]

    def fix_locations(self, 
                      sorted_locations: list[object]) -> list[Point, int]:
        """Attach location to nearest road or intersection
        
        Args:
            sorted_locations: list of locations and weighted scores
            
        Returns:
            new_points: adjusted Point locations and weighted scores
        """
    
        if isinstance(sorted_locations, gpd.GeoDataFrame):
            lines = self.road_segments
            new_points = []
            for i, j, k in tqdm(zip(sorted_locations[0].tolist(), 
                                    sorted_locations[1].tolist(),
                                    sorted_locations[2].tolist()), total=sorted_locations.shape[0]):
                best_point = self.nearest_part_of_linestrings(lines, i)
                new_points.append([Point(best_point), j, k])

        elif isinstance(sorted_locations, list):
            lines = self.road_segments
            new_points = []
            for loc in tqdm(sorted_locations):
                best_point = self.nearest_part_of_linestrings(lines, loc[0])
                new_points.append([Point(best_point), loc[1], loc[2]])
    
        else:
            TypeError('Data must either be GeoDataFrame or list')
            
        return new_points
    
    def get_size_station(self, 
                         new_points: list[object]):
        """Get the size of each station based on its score and number of merged stations.
        Args:
            new_points: list of locations, score and number of stations merged.
        Returns:
            new_points: list of locations, size of station and score
        """
        thresholds = [
            np.percentile([k*j for _, j, k in new_points], 50),
            np.percentile([k*j for _, j, k in new_points], 75)
            ]
        final_points = []
        for i in range(len(new_points)):
            val = new_points[i][1]*new_points[i][2]
            if val <= thresholds[0]:
                final_points.append((new_points[i][0],"small", new_points[i][1]))
            elif val <= thresholds[1]:
                final_points.append((new_points[i][0],"medium", new_points[i][1]))
            else:
                final_points.append((new_points[i][0],"large", new_points[i][1]))
        return final_points
       
    def calculate_cost(self,
                       sorted_locations: list[object]) -> gpd.GeoDataFrame:
        """Calculate costs per station in 2030, 2040
        
        Args:
            sorted_locations: location data with size data
            
        Returns:
            sorted_locations: dataframe with location, size, and costs in 2030 and 2040
        """
        sorted_locations = gpd.GeoDataFrame(sorted_locations, geometry=0).set_crs(self.crs)
        
        for i, row in tqdm(sorted_locations.iterrows(), total=sorted_locations.shape[0]):
            cost_profit = self.cost_profit[row[1]]
            sorted_locations.at[i, 'costs_2030'] = cost_profit['capex'] + (cost_profit['capex'] * cost_profit['yearly_opex'] * (2030-(2023 + cost_profit['construction_time'])))
            sorted_locations.at[i, 'costs_2040'] = 2 * cost_profit['capex'] + (cost_profit['capex'] * cost_profit['yearly_opex'] * (2040-(2023 + cost_profit['construction_time'])))
            
        return sorted_locations
    
    def profitability_by_station(self, final_points: list[object], regions_dem: pd.Series):
        """Compute the profitability of each station based on its attractiveness score and the demand.

        Args:
            final_points: list of stations' location, size and score
            regions_dem: dictionnary of regions and their demand
            path_conf: path of the config file
        Returns:
            stations_final: list of stations' location, size, score, load, profitability (%load) and profitability (binary)
        
        """
        capacity_stations = self.conf["capacity_stations"]
        profitability_stations = self.conf["profitability_stations"]

        regions_dem = regions_dem.to_dict()
        demand_total = sum(regions_dem.values())
        score_total = sum([score for i, j, score in final_points])

        capacity_dict = {
            "small": capacity_stations[0],
            "medium": capacity_stations[1],
            "large": capacity_stations[2]}
        profitability_dict = {
            "small": profitability_stations[0],
            "medium": profitability_stations[1],
            "large": profitability_stations[2]
        }

        stations_final = []
        count = 0
        for i in range(len(final_points)):
            demand = final_points[i][2]/score_total*demand_total
            capacity = capacity_dict[final_points[i][1]]
            profitability_binary = demand/capacity>profitability_dict[final_points[i][1]]
            count += profitability_binary
            stations_final.append([
                final_points[i][0], final_points[i][1],
                final_points[i][2], demand, demand/capacity,
                profitability_binary
            ])

        print(f"{count/len(stations_final)} of stations are profitable")   
        return stations_final
    
    def visualize_scenarios(self,
                            sorted_locations_2030: list,
                            sorted_locations_2040: Optional[list] = None, 
                            colors: list[str] = None, 
                            filename: str = 'map.html') -> None:
        
        """Visualize scenarios for both 2030 and 2040
        
        Args:
            sorted_locations_2030: station locations in 2030
            sorted_locations_2040: station locations in 2040
            colors: list of colors for highways
            filename: name of file
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
        
        roads = folium.GeoJson(self.data,
                                 name='Routes',
                                 style_function=style_function
                                )
        
        # hydrogen location dots
        sorted_locations_2030 = gpd.GeoDataFrame(sorted_locations_2030, geometry=0).set_crs(self.crs)
        top_2030 = folium.GeoJson(sorted_locations_2030,
                                  marker = folium.CircleMarker(
                                      radius = 5,
                                      weight = 0,
                                      fill_color = 'blue', 
                                      fill_opacity = 0.6,)                                  
                                  )
        
        roads.add_to(m)
        top_2030.add_to(m)
        
        if sorted_locations_2040 is not None:
            sorted_locations_2040 = gpd.GeoDataFrame(sorted_locations_2040, geometry=0).set_crs(self.crs)
            top_2040 = folium.GeoJson(sorted_locations_2040,
                                      marker = folium.CircleMarker(
                                          radius = 3, 
                                          weight = 0, 
                                          fill_color = 'red',
                                          fill_opacity = 1,)
                                      )
            top_2040.add_to(m)
            
        m.save(filename)

class Case(Scenarios):
    def __init__(self, shapefiles: dict, csvs: dict, jsons: dict, crs: str = '2154') -> None:
        super().__init__(shapefiles, csvs, jsons, crs)
        self.shapefiles = shapefiles
        self.csvs = csvs
        self.jsons = jsons
        self.crs = crs
        
        # competitor stations
        self.competitors = self.csvs['te_dv']
        self.competitors['H2 Conversion'] = self.competitors['H2 Conversion'].fillna(0)
        self.competitors[['lat', 'long']] = self.competitors['Coordinates'].str.split(',', expand=True).astype(float)
        self.competitors['geometry'] = self.competitors.apply(lambda row: Point(row['long'], row['lat']), axis=1)
        
        self.competitors_starting = self.competitors[self.competitors['H2 Conversion'] == 1]
        competitors_remaining = self.competitors[self.competitors['H2 Conversion'] == 0]
        
        total_gas_stations = super().get_best_location(candidate_locations=competitors_remaining[:30]) # for sake of computations
        self.total_gas_stations = [list(t) for t in total_gas_stations]
        
    def recalculate_locations(self, 
                              locations: list[object, int], 
                              competitor_locations: list[object, int],
                              max_distance: int) -> list[object, int]:
        if type(locations[0]) == tuple:
            locations = [list(t) for t in locations]
        if type(competitor_locations) == list:
            competitor_locations = gpd.GeoDataFrame(competitor_locations, geometry=0)

        
        for point in tqdm(locations):
            station_score = 0.0
            station_weight = -50
            for station in competitor_locations.geometry:
                distance = point[0].distance(station)
                
                if distance < max_distance/2:
                    station_score = (max_distance - distance) / max_distance
                elif distance <= max_distance:
                    station_score = (max_distance - distance) / max_distance / 2
                
            point[1] += station_score * station_weight
            
        return locations
    
    def new_stations_per_region(self,
                                scenario: pd.DataFrame):
        
        avg_increase = (scenario['num_stations_2040'] - scenario['num_stations_2030']) / 10

        for i in range(9, 0, -1):
            scenario[f'num_stations_203{i}'] = scenario['num_stations_2030'] + avg_increase * i
            scenario[f'num_stations_203{i}'] = scenario[f'num_stations_203{i}'].round().astype(int)
        for i in range(9, 0, -1):
            scenario[f'num_stations_203{i}'] = (scenario[f'num_stations_203{i}'] - scenario[f'num_stations_203{i-1}'])
        
        scenario['num_stations_2040'] = scenario['num_stations_2040'] - scenario.drop(columns=['num_stations_2040']).sum(axis=1)
 
        return scenario
    
    def calculate_case3(self, 
                        scored_locations: list[object, int],
                        scenario: pd.DataFrame,
                        max_distance: int = 50_000,
                        final_year: int = 2040,):
        """Simulate scenario 3 for part 3
        
        Args:
            scored_locations:
            scenario:
            final_year:
            
        Returns:
            station_years: dict of new location per year
            existing_locations: a combination of all
        """
        
        total_h2_stations = [[x, 0] for x in self.competitors_starting['geometry']]
        
        all_locations = self.recalculate_locations(scored_locations, self.competitors_starting, max_distance=max_distance)
        locations_2030 = super().distribute_locations(all_locations, scenario['num_stations_2030'])
        locations_2030 = [[x, y] for x, y in zip(locations_2030[0], locations_2030[1])]
        locations_2040 = super().distribute_locations(all_locations, scenario['num_stations_2040'])
        locations_2040 = [[x, y] for x, y in zip(locations_2040[0], locations_2040[1])]

        scenario = self.new_stations_per_region(scenario=scenario)
        
        stations_years = {}
        existing_locations = locations_2030.copy()
        stations_years[2030] = existing_locations
        for year in range(2031, final_year+1):
            new_stations_count = sum(scenario[f'num_stations_{year}'])
            total_h2_stations.extend(self.total_gas_stations[:new_stations_count])
            del self.total_gas_stations[:new_stations_count]
            
            remaining_locations = [sublist for sublist in locations_2040 if sublist not in existing_locations]
            new_stations = self.recalculate_locations(remaining_locations, total_h2_stations, max_distance=max_distance)
            new_stations = super().distribute_locations(new_stations, scenario[f'num_stations_{year}'])
            new_stations = [[x, y] for x, y in zip(new_stations[0], new_stations[1])]
            stations_years[year] = new_stations
            existing_locations.extend(new_stations)
        
        return stations_years
            
         
