#class that links TorNet samples to their IEM warning polygons, based on spatial and temporal overlap
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta

class Polygon_Linker:

    def __init__(self):
        self.loaded_years = {}

    #links a tornet sample with a polygon
    def link(self, year, sample):

        buffer = timedelta(minutes=10)

        #get relevent year of IEM polygons
        if (not year in self.loaded_years):
            self.loaded_years[year] = gpd.read_file(f"./source/IEM_Polygons/warnings-{year}/warnings-{year}.shp")
        IEM_df = self.loaded_years[year]

        #convert datetime fields to datetime objects for the IEM polygons, and project to correct coordinate systems
        IEM_df["ISSUED"] = pd.to_datetime(IEM_df["ISSUED"])
        IEM_df["EXPIRED"] = pd.to_datetime(IEM_df["EXPIRED"])
        IEM_df = IEM_df.to_crs(epsg=4326)
            
        pt = Point(sample.tn_lon, sample.tn_lat)

        # Temporal overlap
        active = IEM_df[
            (IEM_df["ISSUED"] <= sample.tn_end_time + buffer) &
            (IEM_df["EXPIRED"] >= sample.tn_start_time - buffer)
        ]

        # Spatial overlap
        active = active[active.geometry.intersects(pt.buffer(0.02))]


        if len(active) > 0:
            itm = active.iloc[0].to_dict()
            return {
                "geometry": itm["geometry"],
                "PHENOM": itm["PHENOM"]
            }
            
        return None


