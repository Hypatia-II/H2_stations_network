import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import json
from shapely.geometry import LineString, mapping
from itertools import combinations
import re
from scipy import spatial



class Number_Stations():
    def __init__(self, 
                 df_data: pd.DataFrame, 
                 path_conf: str = '../params/config.json', 
                 length_to_use: str ='longest_line') -> None:
        self.conf = json.load(open(path_conf, "r"))
        self.autonomy_high_ms = self.conf['market_share'][0]/sum(self.conf['market_share'])
        self.autonomy_medium_ms = self.conf['market_share'][1]/sum(self.conf['market_share'])
        self.autonomy_low_ms = self.conf['market_share'][2]/sum(self.conf['market_share'])
        self.autonomy_high_km = self.conf['autonomy_share'][0]
        self.autonomy_medium_km = self.conf['autonomy_share'][1]
        self.autonomy_low_km = self.conf['autonomy_share'][2]
        self.df_data = df_data
        
        if (length_to_use not in ['longest_line', 'diameter', 'length_max']):
            length_to_use = 'longest_line'
        self.length_to_use = length_to_use

    def clean_freight_df(self, 
                         df: pd.DataFrame, 
                         on_load: bool = True) -> pd.DataFrame:
        """
        Clean up the freight dataset from excel data

        Args:
            df : the onload or offload dataset to be cleaned
            on_load : Check if onload or offload

        Returns:
            df_clean: cleaned dataframe for onloading or offloading
        """
        df = df[['TIME', 'TIME.1', '2021']]
        
        if on_load:
            freight_type = "number_onload"
        else:
            freight_type = "number_offload"
            
        df.rename(columns={"TIME": "geo_code", "TIME.1": "geo_labels", "2021": freight_type}, inplace=True)
        df = df.iloc[1:]
        df_clean = df[df.geo_code.str.startswith("FR").fillna(False)]
        df_clean.reset_index(inplace=True, drop=True)
        
        df_clean[freight_type] = df_clean[freight_type].astype('str')
        df_clean[freight_type] = df_clean[freight_type].str.replace('\.0*$', '', regex=True)
        df_clean.loc[~(df_clean[freight_type].str.isdigit()), freight_type] = '0'
        df_clean[freight_type] = df_clean[freight_type].astype('int')
        
        return df_clean

    def department_region_map(self, 
                              path: str, 
                              df_fr: pd.DataFrame) -> pd.DataFrame:
        """
        Merge region names with labels

        Args:
            path : path of the region label dataframe
            df_fr : dataframe to be merged on, containing region length data

        Returns:
            df_final: merged final dataframe
        """
        df_dpts_region = pd.read_csv(path)
        df_dpts_region.rename(columns={"dep_name":"geo_labels", "region_name":"new_region_name", "old_region_name": "region"}, inplace=True)
        df_final = pd.merge(df_fr, df_dpts_region[["geo_labels", "region"]], how='left', on=['geo_labels'])
        df_final = df_final[~(df_final.geo_labels.str.endswith(" "))]
        
        return df_final

    def merge_freight(self, 
                      path: str, 
                      df_on: pd.DataFrame, 
                      df_off: pd.DataFrame) -> pd.DataFrame:
        df_onload_fr = self.clean_freight_df(df_on, on_load=True)
        df_offload_fr = self.clean_freight_df(df_off, on_load=False)

        df_fr = pd.merge(df_onload_fr, df_offload_fr, how='inner', on=['geo_code', 'geo_labels'])
        df_fr['total_load'] = df_fr.number_offload + df_fr.number_onload 
        
        df_fr['geo_labels'] = [c[0] for c in df_fr['geo_labels'].str.split("(")]
        df_fr = self.department_region_map(path, df_fr)
        df_fr = df_fr.groupby("region")["total_load"].sum().reset_index()
        df_fr["full_load"] = df_fr["total_load"].sum()   
        df_fr["perc_load"] = df_fr["total_load"]/df_fr["full_load"]

        return df_fr

    def calculate_trucks_stations_peryear(self, 
                                          df: pd.DataFrame, 
                                          year: int = 2030) -> pd.DataFrame:
        
        if (year not in [2030, 2040]):
            year = 2030
            
        if year==2030:
            H2_trucks_num = self.conf['H2_trucks_2030']
        else:
            H2_trucks_num = self.conf['H2_trucks_2040']
            
        df["h2_num_"+str(year)] = H2_trucks_num*df["perc_load"]
        df["R_"+str(year)+"_high_aut"] = self.autonomy_high_ms*df["h2_num_"+str(year)]*df["avg_distance_high_aut"] / self.autonomy_high_km
        df["R_"+str(year)+"_mid_aut"] = self.autonomy_medium_ms*df["h2_num_"+str(year)]*df["avg_distance_midlow_aut"] / self.autonomy_medium_km
        df["R_"+str(year)+"_low_aut"] = self.autonomy_low_ms*df["h2_num_"+str(year)]*df["avg_distance_midlow_aut"] / self.autonomy_low_km
        df["R_"+str(year)+"_total"] = df["R_"+str(year)+"_high_aut"] + df["R_"+str(year)+"_mid_aut"] + df["R_"+str(year)+"_low_aut"]
        df["C_"+str(year)] = self.conf['open_time'] / self.conf['avg_time_fill']
        
        return df
        
    @staticmethod
    def calculate_stations(df: pd.DataFrame, 
                           year: int = 2030) -> pd.DataFrame:
        
        if (year not in [2030, 2040]):
            year = 2030
            
        df["num_stations_"+str(year)] = (df["R_"+str(year)+"_total"] / df["C_"+str(year)]).round().astype(int)

        return df

    def calculate_number_stations(self, 
                                  df: pd.DataFrame) -> pd.DataFrame:

        df["max_length_drive"] = self.conf['max_hours_drive'] * self.conf['avg_speed_kmh']
        df[["length_max", "length_mean", "diameter", "longest_line"]] = df[["length_max", "length_mean", "diameter", "longest_line"]]/1e3
        df["avg_distance_high_aut"] = df[["max_length_drive", self.length_to_use]].min(axis=1) # This would be updated either by diameter or longest point
        df["avg_distance_midlow_aut"] = 0.6*df["avg_distance_high_aut"]#df_new[["max_length_drive", "length_mean"]].min(axis=1)
        
        df = self.calculate_trucks_stations_peryear(df, year=2030)
        df = self.calculate_trucks_stations_peryear(df, year=2040)
        
        df = self.calculate_stations(df, year=2030)
        df = self.calculate_stations(df, year=2040)

        return df
    
    def final_station_calculation(self) -> pd.DataFrame:
        
        df_on = pd.read_excel(self.conf['path_on_freight'], sheet_name='Sheet 1', skiprows=8)
        df_off = pd.read_excel(self.conf['path_off_freight'], sheet_name='Sheet 1', skiprows=8)

        df_fr = self.merge_freight(self.conf['path_region_dpt_map'], df_on, df_off)
        df_new = pd.merge(self.df_data, df_fr[["region", "perc_load"]], how="left", on="region")

        df_new = self.calculate_number_stations(df_new)
        
        return df_new