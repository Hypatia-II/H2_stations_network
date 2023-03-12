import pandas as pd
import numpy as np
import json

path_conf = '../params/config.json'
scenario ="scenario1"
conf = json.load(open(path_conf, "r"))

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
    
    return df, H2_trucks_num
    
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

    return df

def calculate_number_stations(df: pd.DataFrame, 
                              length_to_use, 
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
    
    df, H2_trucks_num_2030 = calculate_trucks_stations_peryear(df, 
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
    df, H2_trucks_num_2040 = calculate_trucks_stations_peryear(df, 
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
    
    df = calculate_stations(df, year=2030)
    df = calculate_stations(df, year=2040)

    return df, H2_trucks_num_2030, H2_trucks_num_2040

def final_station_calculation() -> pd.DataFrame:
    """
    Combine all functions to create the new dataframe with the estimated number of refill stations

    Returns:
        df_new: final dataframe with all needed metrics calculated
    """

    df_new = calculate_number_stations(df_new)
    
    return df_new

def save_scenario(df, scenario_name):
    """Save predictions by region in a json file as dict.
    """
    
    dict_sc= {
    "demand_share_2030": demand_share_2030,
    "demand_share_2040": demand_share_2040,
    "market_share": [autonomy_high_ms, autonomy_medium_ms, autonomy_low_ms]}
    conf.update({scenario_name: dict_sc})
    with open('params/config.json', 'w+') as f:
        json.dump(conf, f, ensure_ascii=False)
    return None

def save_predictions(df, scenario_name):
    """Save predictions by region in a json file as dict.
    """
    df_json = df[["region", "num_stations_2030", "num_stations_2040"]].set_index("region").to_dict()
    with open('data/output_' + scenario_name + '.json', 'w+') as f:
        json.dump(df_json, f, ensure_ascii=False)
    return None

# def get_scenario_output(df):
#     """Print the outputs for the scenario.
#     """
#     num_stations_2030 = df["num_stations_2030"].sum()
#     num_stations_2040 = df["num_stations_2040"].sum()
#     print(f"The output for {scenario} is {num_stations_2030} for 2030 and {num_stations_2040} for 2040.")
#     for region in df["region"].values:
#         num_station_2030 = df[df.region==region]["num_stations_2030"].values[0]
#         num_station_2040 = df[df.region==region]["num_stations_2040"].values[0]
#         print(f"Region {region}: {num_station_2030} stations for 2030 and {num_station_2040} for 2040")
#     return None

def calculate_metrics(df, selected_year, selected_month, selected_day):
    if selected_year == "Select All" and selected_month == "Select All" and selected_day == "Select All":
        filtered_df = df
    elif selected_year != "Select All" and selected_month == "Select All" and selected_day == "Select All":
        filtered_df = df[(df['year'] == selected_year)]
    elif selected_year == "Select All" and selected_month != "Select All" and selected_day == "Select All":
        filtered_df = df[(df['month'] == selected_month)]
    elif selected_year == "Select All" and selected_month == "Select All" and selected_day != "Select All":
        filtered_df = df[(df['day'] == selected_day)]
    elif selected_year != "Select All" and selected_month != "Select All" and selected_day == "Select All":
        filtered_df = df[(df['year'] == selected_year) &
                        (df['month'] == selected_month)]
    elif selected_year != "Select All" and selected_month == "Select All" and selected_day != "Select All":
        filtered_df = df[(df['year'] == selected_year) &
                        (df['day'] == selected_day)]
    elif selected_year == "Select All" and selected_month != "Select All" and selected_day != "Select All":
        filtered_df = df[(df['day'] == selected_day) &
                        (df['month'] == selected_month)]
    elif selected_year != "Select All" and selected_month != "Select All" and selected_day != "Select All":
        filtered_df = df[(df['year'] == selected_year) &
                        (df['month'] == selected_month) &
                        (df['day'] == selected_day)]
    
    #Metrics
    avg_wait_time = filtered_df['WAIT_TIME_MAX'].mean()
    guest_carried = filtered_df['GUEST_CARRIED'].mean() 
    avg_adjust_capacity_utilization = (filtered_df['GUEST_CARRIED'].sum() / filtered_df['ADJUST_CAPACITY'].sum()) * 100
    sum_attendance = filtered_df['attendance'].sum()

    not_equal = filtered_df[filtered_df['CAPACITY'] != filtered_df['ADJUST_CAPACITY']]
    per_cap_adj = not_equal.shape[0] / df.shape[0] * 100

    return avg_wait_time, guest_carried, avg_adjust_capacity_utilization, sum_attendance, per_cap_adj
