#class that handles the lookup of the nearest valid NEXRAD radar site to a given lat/lon point
import geopandas
import pandas
from shapely.geometry import Point

DATA_PATH = "./source/nexrad_sites.csv"

class Nexrad_Lookup:

    def __init__(self):
        df = pandas.read_csv(DATA_PATH)
        df = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        df = df.to_crs("EPSG:3857")
        self.df = df

    def getNearestRadarId(self, point: Point):
        point = geopandas.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs("EPSG:3857")
        res = geopandas.sjoin_nearest(point, self.df)
        return res.iloc[0].radar_id
    
    def getRadarCoords(self, radar_id):
        itm = self.df.loc[self.df["radar_id"] == radar_id]
        if (itm.empty):
            return None
        lat = float(itm["lat"].iloc[0])
        lon = float(itm["lon"].iloc[0])
        return Point(lon, lat)