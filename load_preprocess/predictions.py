import numpy as np
import pandas as pd
from tqdm import tqd

def number_refills(df):
    
    df["number_h2_trucks"]
    df["distance_traveled_average"]
    df["autonomy"]
    df["Refills"] = df["number_h2_trucks"]*df["distance_traveled_average"]/df["autonomy"]
    
    return df

def capacity_by_station(df):
    
    df['capacity'] = df['']