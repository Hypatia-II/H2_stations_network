import pandas as pd
import numpy as np

path_conf = '../params/config.json'
length_to_use ='longest_line'
scenario ="scenario1"
conf = json.load(open(path_conf, "r"))
scenario = scenario
autonomy_high_ms = conf[scenario]['market_share'][0]
autonomy_medium_ms = conf[scenario]['market_share'][1]
autonomy_low_ms = conf[scenario]['market_share'][2]
demand_share_2030 = conf[scenario]['demand_share_2030']
demand_share_2040 = conf[scenario]['demand_share_2040']
perc_dist_mid = conf['perc_distance'][0]
perc_dist_low = selfconf['perc_distance'][1]
autonomy_high_km = conf['autonomy_share'][0]
autonomy_medium_km = conf['autonomy_share'][1]
autonomy_low_km = conf['autonomy_share'][2]

truck_tank_size = conf["truck_tank_size"]
station_tank_size = conf["station_tank_size"]
    
if (length_to_use not in ['longest_line', 'diameter', 'length_max']):
    length_to_use = 'longest_line'
length_to_use = length_to_use

def clean_freight_df(df: pd.DataFrame, 
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

def department_region_map(path: str, 
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

def merge_freight(path: str, 
                  df_on: pd.DataFrame, 
                  df_off: pd.DataFrame) -> pd.DataFrame:
    """
    Merge freight dataframes with the dataframe about the region's dimensions

    Args:
        path : path to the region specific dataframe
        df_on : onloading dataframe
        df_off : offloading dataframe

    Returns:
        df_fr: updated and merged dataframe with all port actions per year and per region, to the original dataframe
    """
    df_onload_fr = clean_freight_df(df_on, on_load=True)
    df_offload_fr = clean_freight_df(df_off, on_load=False)

    df_fr = pd.merge(df_onload_fr, df_offload_fr, how='inner', on=['geo_code', 'geo_labels'])
    df_fr['total_load'] = df_fr.number_offload + df_fr.number_onload 
    
    df_fr['geo_labels'] = [c[0] for c in df_fr['geo_labels'].str.split("(")]
    df_fr = department_region_map(path, df_fr)
    df_fr = df_fr.groupby("region")["total_load"].sum().reset_index()
    df_fr["full_load"] = df_fr["total_load"].sum()   
    df_fr["perc_load"] = df_fr["total_load"]/df_fr["full_load"]

    return df_fr

def calculate_trucks_stations_peryear(df: pd.DataFrame, 
                                      year: int = 2030) -> pd.DataFrame:
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
    
@staticmethod
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

def calculate_number_stations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the number of stations from the number of trucks present

    Args:
        df: dataframe containing region specific variables regarding roads and traffic

    Returns:
        df: updated dataframe with total number of stations needed per region
    """
    df["max_length_drive"] = conf['max_hours_drive'] * conf['avg_speed_kmh']
    df[["length_max", "length_mean", "diameter", "longest_line"]] = df[["length_max", "length_mean", "diameter", "longest_line"]]/1e3
    df["avg_distance_high_aut"] = df[["max_length_drive", length_to_use]].min(axis=1) # This would be updated either by diameter or longest point
    df["avg_distance_mid_aut"] = perc_dist_mid*df["avg_distance_high_aut"]
    df["avg_distance_low_aut"] = perc_dist_low*df["avg_distance_high_aut"]
    
    df = calculate_trucks_stations_peryear(df, year=2030)
    df = calculate_trucks_stations_peryear(df, year=2040)
    
    df = calculate_stations(df, year=2030)
    df = calculate_stations(df, year=2040)

    return df

def final_station_calculation() -> pd.DataFrame:
    """
    Combine all functions to create the new dataframe with the estimated number of refill stations

    Returns:
        df_new: final dataframe with all needed metrics calculated
    """
    df_on = pd.read_excel(conf['path_on_freight'], sheet_name='Sheet 1', skiprows=8)
    df_off = pd.read_excel(conf['path_off_freight'], sheet_name='Sheet 1', skiprows=8)

    df_fr = merge_freight(conf['path_region_dpt_map'], df_on, df_off)
    df_new = pd.merge(df_data, df_fr[["region", "perc_load"]], how="left", on="region")

    df_new = calculate_number_stations(df_new)
    
    return df_new

def save_predictions(df):
    """Save predictions by region in a json file as dict.
    """
    df_json = df[["region", "num_stations_2030", "num_stations_2040"]].set_index("region").to_dict()
    with open('data/output_' + scenario + '.json', 'w+') as f:
        json.dump(df_json, f, ensure_ascii=False)
    return None

def get_scenario_output(df):
    """Print the outputs for the scenario.
    """
    num_stations_2030 = df["num_stations_2030"].sum()
    num_stations_2040 = df["num_stations_2040"].sum()
    print(f"The output for {scenario} is {num_stations_2030} for 2030 and {num_stations_2040} for 2040.")
    for region in df["region"].values:
        num_station_2030 = df[df.region==region]["num_stations_2030"].values[0]
        num_station_2040 = df[df.region==region]["num_stations_2040"].values[0]
        print(f"Region {region}: {num_station_2030} stations for 2030 and {num_station_2040} for 2040")
    return None

def get_data_ready(df):
    """Perform preprocessing on a dataframe and return and cleaned dataframe.
    Args:
    df (pandas.DataFrame): Input DataFrame

    Returns:
    df (pandas.DataFrame): Cleaned DataFrame
    """
    df['WORK_DATE'] = pd.to_datetime(df['WORK_DATE'])
    df['year'] = df['WORK_DATE'].dt.year
    df['month'] = df['WORK_DATE'].dt.month
    df['day'] = df['WORK_DATE'].dt.day

    df['DEB_TIME'] = pd.to_datetime(df['DEB_TIME'])
    df['hour'] = df['DEB_TIME'].dt.hour
    df['minute'] = df['DEB_TIME'].dt.minute
    df['second'] = df['DEB_TIME'].dt.second 

    #Clean CAPACITY and ADJUST CAPACITY
    #Replace negative Guest carried
    df["GUEST_CARRIED"] = np.where(df["GUEST_CARRIED"] < 0, 0, df["GUEST_CARRIED"])

    #Drop rows with unconsistent behavior
    df = df[~( 
          (df['CAPACITY'] == 0) & 
          (df['ADJUST_CAPACITY'] == 0))]

    #If GUEST_CARRIED not null and ADJUST_CAPACITY = 0, set its value to
    # CAPACITY
    df.loc[(df['GUEST_CARRIED'] != 0) & 
       (df['ADJUST_CAPACITY'] == 0) & 
       (df['CAPACITY'] != 0), 
       'ADJUST_CAPACITY'] = df['CAPACITY']

    # If GUEST_CARRIED not null and CAPACITY = 0, set CAPACITY 
    # to the biggest value between GUEST_CARRIED and ADJUST_CAPACITY
    df.loc[(df['GUEST_CARRIED'] != 0) & 
       (df['CAPACITY'] == 0), 
       'CAPACITY'] = df[['CAPACITY', 'GUEST_CARRIED']].max(axis=1)

    return df


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
