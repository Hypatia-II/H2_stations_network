import pandas as pd
import numpy as np
import json

path_conf = '../app/pages/utils/params/config_st.json'
conf = json.load(open(path_conf, "r"))

keys_l = list((conf.keys()))
scenario_list = [ck for ck in keys_l if ck.startswith('Scenario ')]
scenario = scenario_list[0]

# DONE
# Value entry & Selection
autonomy_high_ms = conf[scenario]['market_share'][0]
autonomy_medium_ms = conf[scenario]['market_share'][1]
autonomy_low_ms = conf[scenario]['market_share'][2]

# DONE
# Value entry & Selection
demand_share_2030 = conf[scenario]['demand_share_2030']
demand_share_2040 = conf[scenario]['demand_share_2040']

# No idea what this is
perc_dist_mid = conf['perc_distance'][0]
perc_dist_low = conf['perc_distance'][1]

# DONE
# Value entry & Selection
autonomy_high_km = conf['autonomy_share'][0]
autonomy_medium_km = conf['autonomy_share'][1]
autonomy_low_km = conf['autonomy_share'][2]

# DONE
# Value entry & Selection
truck_tank_size = conf["truck_tank_size"]

# Value entry & Selection
station_tank_size = conf["station_tank_size"]

# DONE
# Selection
length_to_use ='longest_line'
if (length_to_use not in ['longest_line', 'diameter', 'length_max']):
    length_to_use = 'longest_line'
length_to_use = length_to_use

def calculate_trucks_stations_peryear(df: pd.DataFrame, 
                                      year: int = 2030,
                                      demand_share_2030: float = demand_share_2030,
                                      demand_share_2040: float = demand_share_2040,
                                      autonomy_high_ms: int = autonomy_high_ms,
                                      autonomy_medium_ms: int = autonomy_medium_ms,
                                      autonomy_low_ms: int = autonomy_low_ms,
                                      autonomy_high_km: int = autonomy_high_km,
                                      autonomy_medium_km: int = autonomy_medium_km,
                                      autonomy_low_km: int = autonomy_low_km,
                                      truck_tank_size: list = truck_tank_size,
                                      station_tank_size: list = station_tank_size) -> pd.DataFrame:
    """
    Function to calculate the number of trucks needed per region and truck manufacturer based on the year

    Args:
        df : dataframe containing region specific variables regarding roads and traffic
        year : the year to perform the estimation on

    Returns:
        df: updated dataframe with the number of trucks and the number of refills needed per manufacturer
    """
    if (year not in [2030, 2040]):
        year = 2030
        
    if year==2030:
        H2_trucks_num = conf['H2_trucks_2030']*demand_share_2030
    else:
        H2_trucks_num = conf['H2_trucks_2040']*demand_share_2040
        
    df["h2_num_"+str(year)] = H2_trucks_num*df["perc_load"]
    df["R_"+str(year)+"_high_aut"] = autonomy_high_ms*df["h2_num_"+str(year)]*df["avg_distance_high_aut"] / autonomy_high_km
    df["R_"+str(year)+"_mid_aut"] = autonomy_medium_ms*df["h2_num_"+str(year)]*df["avg_distance_mid_aut"] / autonomy_medium_km
    df["R_"+str(year)+"_low_aut"] = autonomy_low_ms*df["h2_num_"+str(year)]*df["avg_distance_low_aut"] / autonomy_low_km
    df["R_"+str(year)+"_total"] = df["R_"+str(year)+"_high_aut"] + df["R_"+str(year)+"_mid_aut"] + df["R_"+str(year)+"_low_aut"]
    capacity = min(np.sum(station_tank_size)*1000/np.sum(truck_tank_size), conf['open_time'] / conf['avg_time_fill'])
    df["C_"+str(year)] = capacity
    
    return df
    
def calculate_stations(df: pd.DataFrame, 
                        year: int = 2030) -> pd.DataFrame:
    """
    Calculate number of stations based on the year and the number of refills needed per region

    Args:
        df : dataframe containing the number of refills needed per region
        year : the year to perform the estimation on

    Returns:
        df: updated dataframe with calculated number of stations needed
    """
    if (year not in [2030, 2040]):
        year = 2030
        
    df["num_stations_"+str(year)] = (df["R_"+str(year)+"_total"] / df["C_"+str(year)]).round().astype(int)
    num_stations = df["num_stations_"+str(year)].sum()

    return df, num_stations

def calculate_number_stations(df: pd.DataFrame, 
                              length_to_use: str = 'longest_line', 
                              demand_share_2030: float = demand_share_2030,
                              demand_share_2040: float = demand_share_2040,
                              autonomy_high_ms: int = autonomy_high_ms,
                              autonomy_medium_ms: int = autonomy_medium_ms,
                              autonomy_low_ms: int = autonomy_low_ms,
                              autonomy_high_km: int = autonomy_high_km,
                              autonomy_medium_km: int = autonomy_medium_km,
                              autonomy_low_km: int = autonomy_low_km,
                              truck_tank_size: list = truck_tank_size,
                              station_tank_size: list = station_tank_size,
                              H2_stations_2030: int = 0,
                              H2_stations_2040: int = 0,
                              del1 = None,
                              del2 = None) -> pd.DataFrame:
    """
    Estimate the number of stations from the number of trucks present

    Args:
        df: dataframe containing region specific variables regarding roads and traffic

    Returns:
        df: updated dataframe with total number of stations needed per region
    """
    perc_dist_mid = conf['perc_distance'][0]
    perc_dist_low = conf['perc_distance'][1]

    df["max_length_drive"] = conf['max_hours_drive'] * conf['avg_speed_kmh']
    df["avg_distance_high_aut"] = df[["max_length_drive", length_to_use]].min(axis=1)
    df["avg_distance_mid_aut"] = perc_dist_mid*df["avg_distance_high_aut"]
    df["avg_distance_low_aut"] = perc_dist_low*df["avg_distance_high_aut"]
    
    df = calculate_trucks_stations_peryear(df, 
                                           year=2030,
                                           demand_share_2030 = demand_share_2030,
                                           demand_share_2040 = demand_share_2040,
                                           autonomy_high_ms = autonomy_high_ms,
                                           autonomy_medium_ms = autonomy_medium_ms,
                                           autonomy_low_ms = autonomy_low_ms,
                                           autonomy_high_km = autonomy_high_km,
                                           autonomy_medium_km = autonomy_medium_km,
                                           autonomy_low_km = autonomy_low_km,
                                           truck_tank_size = truck_tank_size,
                                           station_tank_size = station_tank_size)
    df = calculate_trucks_stations_peryear(df, 
                                           year=2040,
                                           demand_share_2030 = demand_share_2030,
                                           demand_share_2040 = demand_share_2040,
                                           autonomy_high_ms = autonomy_high_ms,
                                           autonomy_medium_ms = autonomy_medium_ms,
                                           autonomy_low_ms = autonomy_low_ms,
                                           autonomy_high_km = autonomy_high_km,
                                           autonomy_medium_km = autonomy_medium_km,
                                           autonomy_low_km = autonomy_low_km,
                                           truck_tank_size = truck_tank_size,
                                           station_tank_size = station_tank_size)
    
    H2_stations_2030_prev = H2_stations_2030
    H2_stations_2040_prev = H2_stations_2040
    del1_prev = del1
    del2_prev = del2
    
    df, H2_stations_2030 = calculate_stations(df, year=2030)
    df, H2_stations_2040 = calculate_stations(df, year=2040)
    
    if (H2_stations_2030_prev != H2_stations_2030) or (H2_stations_2040_prev != H2_stations_2040):
        del1 = H2_stations_2030 - H2_stations_2030_prev
        del2 = H2_stations_2040 - H2_stations_2040_prev
        if (del1==del1_prev) or (del1==0):
            del1 = None
        if (del2==del2_prev) or (del2==0):
            del2 = None
            
    # if H2_stations_2030_prev != H2_stations_2030:
    #     del1 = H2_stations_2030 - H2_stations_2030_prev
    #     if del1==del1_prev:
    #         del1 = None
        
    # if H2_stations_2040_prev != H2_stations_2040:
    #     del2 = H2_stations_2040 - H2_stations_2040_prev
    #     if del2==del2_prev:
    #         del2 = None

    return df, H2_stations_2030, H2_stations_2040, del1, del2

def save_scenario(scenario_name):
    """Save predictions by region in a json file as dict.
    """
    
    dict_sc= {
    "demand_share_2030": demand_share_2030,
    "demand_share_2040": demand_share_2040,
    "market_share": [autonomy_high_ms, autonomy_medium_ms, autonomy_low_ms]}
    conf.update({scenario_name: dict_sc})
    with open(path_conf, 'w+') as f:
        json.dump(conf, f, ensure_ascii=False)
    return None

def save_predictions(df, scenario_name):
    """Save predictions by region in a json file as dict.
    """
    df_json = df[["region", "num_stations_2030", "num_stations_2040"]].set_index("region").to_dict()
    with open('../data/output_' + scenario_name + '.json', 'w+') as f:
        json.dump(df_json, f, ensure_ascii=False)
    return None