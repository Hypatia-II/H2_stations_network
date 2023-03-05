import geopandas as gpd
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import spatial
import shapely
from shapely.geometry import LineString, mapping
import re
from scipy import spatial


class Data():
    def __init__(self, 
                 path: str = '../data/') -> None:
        self.path = path
    
    def get_shapefiles(self) -> dict:
        """
        Load all .shp files in folder to geopandas dataframes
        
        Args:
            path: the folder they are located in
            
        Returns:
            shapes: dict with all geopandas dataframes
        
        """
        shapefiles = glob.glob(self.path + '*.shp')
        shapes = {}
        for shapefile in tqdm(shapefiles):
            name  = shapefile.split('/')[-1].split('\\')[-1].split('.')[0]
            shapes[name] = gpd.read_file(shapefile)
            
        return shapes
    
    def get_csvs(self) -> dict:
        """
        Load all .csv files in folder to geopandas dataframes
        
        Args:
            path: the folder they are located in
            
        Returns:
            shapes: dict with all csv dataframes
        """
        
        csvs = glob.glob(self.path + '*.csv')
        files = {}
        for file in tqdm(csvs):
            name = file.split('/')[-1].split('\\')[-1].split('.')[0]
            files[name] = pd.read_csv(file, sep=None, engine='python')
        
        return files
    
    def calculate_max_length(self, 
                             g: shapely.geometry) -> float:
        """
        Calculate the longest length in a polygon

        Args:
            g (shapely.geometry): geometry of a polygon

        Returns:
            longest_line: the longest distance calculated
        """
        all_coords = str(mapping(g)["coordinates"]) #https://gis.stackexchange.com/questions/287306/list-all-polygon-vertices-coordinates-using-geopandas
        all_xys = re.findall("\d+\.\d+", all_coords) #I know this is ugly, but it works in extracting floats from nested tuples
        all_xys = [float(c) for c in all_xys]
        all_xys = np.array([[a,b] for a,b in zip(all_xys[::2], all_xys[1::2])])
        candidates = all_xys[spatial.ConvexHull(all_xys).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        longest_line = LineString([(float(candidates[i][0]),float(candidates[i][1])), (float(candidates[j][0]),float(candidates[j][1]))]).length
        
        return longest_line
    
    def calculate_length_all(self,
                             g: shapely.geometry) -> float:
        """
        Execute the longest length calculation per polygon, and taking the longest length from all polygons of a multigon

        Args:
            g (shapely.geometry): geometry of a polygon

        Returns:
            float: the longest distance calculated
        """
        max_v = 0
        if g.geom_type == 'MultiPolygon':
            gg = list(g.geoms)
            for i in range(len(gg)):
                gg_v = self.calculate_max_length(gg[i])
                max_v = max(float(max_v), float(gg_v))
        elif g.geom_type == 'Polygon':
            max_v = self.calculate_max_length(g)
            
        return max_v

    def calculate_road_density(self,
                               shapefiles: dict,
                               highways_only: bool = True) -> pd.DataFrame:
        """Calculate road density per region
        
        Args:
            shapefiles: dict of shapefiles loaded in previous 
            highways_only: whether you want to consider only highways or not
            
        Returns:
            df: dataframe breaking down road density for each region
        """
        routes = shapefiles['VSMAP_TOUT']
        
        if highways_only:
            routes = routes[routes.lib_rte.str.startswith('A')]
        
        regions = shapefiles['FRA_adm1']
        regions = regions.to_crs(epsg=2154)
        regions['area_m'] = regions.geometry.area
        regions['longest_line'] = regions['geometry'].apply(self.calculate_length_all)

        joined = gpd.sjoin(routes, regions, predicate='within')
        joined['length_m'] = joined.geometry.length

        total_length_by_region = joined.groupby('NAME_1')['length_m'].sum()
        
        regions = pd.merge(regions, total_length_by_region, on='NAME_1', how='inner')
        #absolute mess of calculations
        temp_a = (joined.groupby(['NAME_1', 'lib_rte'])['length_m'].sum() / 2).groupby('NAME_1').max()
        temp_b = (joined.groupby(['NAME_1', 'lib_rte'])['length_m'].sum() / 2).groupby('NAME_1').mean()

        temp_c = pd.merge(temp_a, temp_b, on='NAME_1').rename(columns={'length_m_x': 'length_max',
                                                                       'length_m_y': 'length_mean'})
        
        regions = pd.merge(regions, temp_c, on='NAME_1', how='inner')
        regions['road_density'] = regions['length_m'] / regions['area_m']
        regions['diameter'] = np.sqrt((regions['area_m'] / np.pi))*2
        
        df = regions[['NAME_1', 'road_density', 'length_m', 'area_m', 'length_max', 'length_mean', 'diameter', 'longest_line']].sort_values(by='road_density', ascending=False)
        df.rename(columns={"NAME_1":"region"}, inplace=True)

        
        return df
    
    def create_df(self,
                  highways_only: bool = False) -> pd.DataFrame:
        """Create the initial dataframe and calculate road density

        Args:
            highways_only: boolean to know whether limit calculation on highways only or include other roads

        Returns:
            pd.DataFrame: dataframe combining road densities to max, min and total road length
        """
        shapefiles = self.get_shapefiles()
        df = self.calculate_road_density(shapefiles, highways_only = highways_only)
        
        return df