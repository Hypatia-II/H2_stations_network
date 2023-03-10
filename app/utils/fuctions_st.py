import pandas as pd
import numpy as np

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
