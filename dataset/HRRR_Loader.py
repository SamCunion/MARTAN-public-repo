#creates a loader for querying the hrrr archives for meso data
import xarray as xr
import s3fs
import time
import cartopy.crs as ccrs
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.exceptions import ClientError
import datetime
import numcodecs as ncd
import numpy as np
import dataclasses
import geopandas
import asyncio
from botocore import UNSIGNED
from botocore.config import Config

VARIABLE_MAP = {
    "lcl_height": ("level_of_adiabatic_condensation_from_sfc", "HGT"),
    "dewpoint": ("2m_above_ground", "DPT"),
    "mlcape": ("90_0mb_above_ground", "CAPE"),
    "cape_0_3km": ("0_3000m_above_ground", "CAPE"),
    "srh_0_1km": ("1000_0m_above_ground", "HLCY"),
    "shear_0_1km_U": ("0_1000m_above_ground", "VUCSH"),
    "shear_0_1km_V": ("0_1000m_above_ground", "VVCSH"),
    "shear_0_6km_U": ("0_6000m_above_ground", "VUCSH"),
    "shear_0_6km_V": ("0_6000m_above_ground", "VVCSH"),
}


class HRRR_Loader:

    def __init__(self):
        #load chunk index
        self.fs = s3fs.S3FileSystem(anon=True)
        self.chunk_index = xr.open_zarr(s3fs.S3Map("s3://hrrrzarr/grid/HRRR_chunk_index.zarr", s3=self.fs))
        self.projection = ccrs.LambertConformal(central_longitude=262.5, central_latitude=38.5, standard_parallels=(38.5, 38.5), globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229))
        self.s3 = boto3.resource(service_name="s3", region_name="us-west-1", config=Config(signature_version=UNSIGNED))

    #gets the nearest point on the projection from lat/lon to globe segment
    def get_nearest_point(self, lon, lat):
        x, y = self.projection.transform_point(lon, lat, ccrs.PlateCarree())
        return self.chunk_index.sel(x=x, y=y, method="nearest")
    
    #builds the first part of the URI
    def create_s3_group_url(self, zarr_id):
        url = ""
        url += zarr_id.run_hour.strftime(
            f"{zarr_id.level_type}/%Y%m%d/%Y%m%d_%Hz_{zarr_id.model_type}.zarr/"
        )
        url += f"{zarr_id.var_level}/{zarr_id.var_name}"
        return url
    
    #builds the second part of the URI
    def create_s3_subgroup_url(self, zarr_id):
        url = self.create_s3_group_url(zarr_id)
        url += f"/{zarr_id.var_level}"
        return url
    
    #builds the third part of the URI
    def create_s3_chunk_url(self, zarr_id, chunk_id):
        url = self.create_s3_subgroup_url(zarr_id)
        url += f"/{zarr_id.var_name}/{chunk_id}"
        return url
    
    #gets the compressed data from the zarr repo
    def retrieve_object(self, s3_url):
        obj = self.s3.Object("hrrrzarr", s3_url)
        try:
            return obj.get()["Body"].read()
        except ClientError as e:
            if (e.response["Error"]["Code"] == "NoSuchKey"):
                return None
            else:
                print("timeout detected, retrying HRRR get")
                time.sleep(1)
                return self.retrieve_object(s3_url)
                
    
    #decompresses the returned chunk
    def decompress_chunk(self, compressed_data):
        buffer = ncd.blosc.decompress(compressed_data)
        chunk = np.frombuffer(buffer, dtype="<f2") #might need to be <f4 for newer samples
        data_array = np.reshape(chunk, (150, 150))
        return data_array
    
    def getMesoData(self, datetime: datetime.datetime, point: geopandas.GeoSeries):

        #calculate chunk point first, so we dont have to calcualte it for each variable
        nearest_point = self.get_nearest_point(point.x, point.y)
        chunk_id = nearest_point.chunk_id.values
        chunk_y = nearest_point.in_chunk_y.values
        chunk_x = nearest_point.in_chunk_x.values

        #get each variable

        vars = [
            "lcl_height",
            "dewpoint",
            "mlcape",
            "cape_0_3km",
            "srh_0_1km",
            "shear_0_1km_U",
            "shear_0_1km_V",
            "shear_0_6km_U",
            "shear_0_6km_V"
        ]

        with ThreadPoolExecutor(max_workers=9) as executor:
            futures = {
                executor.submit(self.getMesoValueAtPoint, v, datetime, chunk_id, chunk_y, chunk_x): v
                for v in vars
            }

            results = {}
            for future in futures:
                varname = futures[future]
                results[varname] = future.result()
        

        if (results["lcl_height"] == None and results["cape_0_3km"] == None and results["shear_0_6km_U"] == None): #no valid entries exist
            return None
        
        #calculate shear given U, V
        results["shear_0_1km"] = np.sqrt(results["shear_0_1km_U"]**2 + results["shear_0_1km_V"]**2)
        results["shear_0_6km"] = np.sqrt(results["shear_0_6km_U"]**2 + results["shear_0_6km_V"]**2)
        del results["shear_0_1km_U"]
        del results["shear_0_1km_V"]
        del results["shear_0_6km_U"]
        del results["shear_0_6km_V"]

        return results



    def getMesoValueAtPoint(self, varname, datetime: datetime.datetime, chunk_id, chunk_y, chunk_x):
        #map varname to zarr_id
        zarr_vars = VARIABLE_MAP[varname]
        zarr_id = ZarrId(
            run_hour=datetime,
            level_type="sfc",
            var_level=zarr_vars[0],
            var_name=zarr_vars[1],
            model_type="anl"
        )
        compressed_data = self.retrieve_object(self.create_s3_chunk_url(zarr_id, chunk_id))
        if (compressed_data == None): #no values retrieved, probably a sample earlier than the HRRR repo has data for (2014)
            return None
        chunk_data = self.decompress_chunk(compressed_data)

        return chunk_data[chunk_y, chunk_x]
        

@dataclasses.dataclass
class ZarrId:
    run_hour: datetime.datetime
    level_type: str
    var_level: str
    var_name: str
    model_type: str

