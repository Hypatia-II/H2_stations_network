"""Identify number of stations, locations and profitability based on competition scenarios"""
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

class Competition_Scenarios():
    def __init__(self, 
                 jsons: dict,
                 path_conf: str = 'params/config.json') -> None:
            self.conf = json.load(open(path_conf, "r"))
            self.jsons = jsons

    def stations_per_year(self, final_points: list[object], scenario: str="scenario1") -> gpd.GeoDataFrame:
        """Location and number of stations to be operational per year.

        Args:
            final_points: Locations with size of station, score and demand in 2040.

        Returns:
            final_point_scenario: geodatframe with locations, size of station, demand in 2040 and operational year.
        """
        final_points_copy = gpd.GeoDataFrame(final_points, geometry=0).set_crs('2154')
        final_points_copy.rename(columns={0:"points", 1:"size", 2:"score", 3:"2040_demand"}, inplace=True)
        final_points_copy.sort_values(by="score", ascending=False, inplace=True)
        final_points_copy.reset_index(inplace=True, drop=True)  
        if scenario == "scenario2":
            n_2030_total = np.sum(self.jsons['output_scenario1']["num_stations_2030"])
            n_2040_total = np.sum(self.jsons['output_scenario1']["num_stations_2040"])
            n_2030=int(np.round(n_2030_total/2))
            n_2040=int(np.round(n_2040_total/2))
            n = n_2030_total + n_2040_total
            final_point_scenario = final_points_copy.iloc[:n_2030_total,:].sample(n=n_2030)
            final_point_scenario = pd.concat(
                [final_point_scenario,
                final_points_copy.iloc[n_2030_total:n,:].sample(n=n_2040)], axis=0)
            final_point_scenario.sort_values(by="score", ascending=False, inplace=True)
            final_point_scenario.reset_index(inplace=True, drop=True)
        elif scenario == "scenario1":
            n_2030 = np.sum(self.jsons['output_scenario2']["num_stations_2030"])
            n_2040 = np.sum(self.jsons['output_scenario2']["num_stations_2040"])
            n_tot = n_2030 + n_2040
            n_2030 = int(len(final_points)*n_2030/n_tot)
            n_2040 = int(len(final_points)*n_2040/n_tot)
            n = n_2030 + n_2040
            final_point_scenario = final_points_copy.iloc[:n, :]
        
        final_point_scenario.loc[:n_2030, "year"] = "2030"
        n_init = n_2030 + 1
        n_yearly = int(n_2040/10) - 1
        for i in range(10):
            year = str(2030 + i + 1)
            n = n_init + n_yearly
            final_point_scenario.loc[n_init:n, "year"] = year
            n_init = n + 1
            
        return final_point_scenario
    
    def calculate_yearly_op_profit(self, final_point_scenario: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate operational profit per year per station
        
        Args:
            final_points: location data with size of station, demand in 2040 and operational year
            
        Returns:
            final_points: dataframe with location, size, and operational profits per year
        """
        cost_profit_dict = self.jsons["cost_profit"]
        growth_rate = self.conf["growth_rate"]
        stations_cost_rev = final_point_scenario[~final_point_scenario.year.isna()][~(final_point_scenario["size"]=="")]
        for i, row in tqdm(stations_cost_rev.iterrows(), total=stations_cost_rev.shape[0]):
            cost_profit = cost_profit_dict[row[1]]
            stations_cost_rev.at[i, 'costs_fix'] = cost_profit['capex']
            stations_cost_rev.at[i, 'costs_operational'] = cost_profit['capex'] * cost_profit['yearly_opex']
            stations_cost_rev.at[i, "revenue"] = min(
                 cost_profit["price_per_kg"]*row[3]/(growth_rate**(2040-int(row["year"]))),
                 cost_profit["price_per_kg"]*cost_profit["capacity"])*365/1000000
            stations_cost_rev.at[i, "profit_op_year"] = stations_cost_rev.at[i, "revenue"] - stations_cost_rev.at[i, 'costs_operational']
        return stations_cost_rev
    
    def max_capacity(self, row):
        if row["size"] == "large":
            return 4000*5/1000000*365 - row["costs_operational"]
        elif row["size"] == "medium":
            return 2000*5/1000000*365 - row["costs_operational"]
        elif row["size"] == "small":
            return 1000*5/1000000*365 - row["costs_operational"]
        
    def min_func(self, row):
        return min(row[0], row[1])

    def get_profitability_by_year(self, stations_cost_rev: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        stations_cost_rev["max_op_profit"] = stations_cost_rev[["size", "costs_operational"]].apply(lambda row: self.max_capacity(row), axis=1)

        cost_yearly = stations_cost_rev.groupby("year")[["costs_fix","profit_op_year", "max_op_profit"]].sum().reset_index()

        growth_rate = 0.196
        cost_yearly = stations_cost_rev.groupby("year")[["costs_fix","profit_op_year", "max_op_profit"]].sum().reset_index()
        col = [cost_yearly.profit_op_year.values[0]]
        for i in range(1, len(cost_yearly.index)):
            col.append(col[i-1]*(1+growth_rate)+cost_yearly.profit_op_year.values[i])
        cost_yearly["cumsum_maxopprofit"] = cost_yearly["max_op_profit"].cumsum()
        cost_yearly['cumsumwithgrowth_profit'] = col
        cost_yearly['cumsumwithgrowth_profit'] = cost_yearly[['cumsumwithgrowth_profit', "cumsum_maxopprofit"]].apply(lambda row: self.min_func(row), axis=1)
        cost_yearly["total_profit"] = cost_yearly['cumsumwithgrowth_profit'] - cost_yearly['costs_fix']
        cost_yearly["total_profit_cumsum"] = cost_yearly["total_profit"].cumsum()
        return cost_yearly