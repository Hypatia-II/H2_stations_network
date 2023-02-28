import geopandas as gpd
import glob
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class Data():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_shapefiles(path):
        """
        Load all .shp files in folder to geopandas dataframes
        
        Args:
            path: the folder they are located in
            
        Returns:
            shapes: dict with all geopandas dataframes
        
        """
        shapefiles = glob.glob(path + '*.shp')
        shapes = {}
        for shapefile in tqdm(shapefiles):
            name  = shapefile.split('\\')[-1].split('.')[0]
            
            shapes[name] = gpd.read_file(shapefile)
            
        return shapes
    
    @staticmethod
    def calculate_road_density(shapefiles):
        routes = shapefiles['VSMAP_TOUT']
        routes = routes[routes.lib_rte.str.startswith('A')]
        regions = shapefiles['FRA_adm1']
        regions = regions.to_crs(epsg=2154)
        regions['area_m'] = regions.geometry.area

        joined = gpd.sjoin(routes, regions, predicate='within')
        joined['length_m'] = joined.geometry.length

        total_length_by_region = joined.groupby('NAME_1')['length_m'].sum()
        regions = pd.merge(regions, total_length_by_region, on='NAME_1', how='inner')
        regions['road_density'] = regions['length_m'] / regions['area_m']
        
        df = regions[['NAME_1', 'road_density', 'length_m', 'area_m']].sort_values(by='road_density', ascending=False)
        
        return df    
    
# script
# os.chdir(r"C:\Users\ckunt\OneDrive\Documents\Masters work\HEC\22. Sustainability Challenge\sust_challenge")

def create_df(path = '../data/'):
    
    data = Data()
    shapefiles = data.get_shapefiles(path)
    df = data.calculate_road_density(shapefiles)
    
    return df