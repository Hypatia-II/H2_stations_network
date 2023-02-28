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
    
os.chdir(r"C:\Users\ckunt\OneDrive\Documents\Masters work\HEC\22. Sustainability Challenge\sust_challenge")

path = 'data/'

data = Data()
shapefiles = data.get_shapefiles(path)

shapefiles['ProfilsTravers_RRN_2022']