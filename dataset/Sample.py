#represents an individual sample within the dataset during compilation
import numpy as np
import asyncio
import os
from datetime import datetime
from shapely import Point
from HRRR_Loader import HRRR_Loader
from Nexrad_Lookup import Nexrad_Lookup
from Polygon_Linker import Polygon_Linker
from TorNet_Handler import TorNet_Handler

SAVE_DIR = os.getenv("DATASET_SAMPLE_SAVE_DIR")

class Sample:
    
    #type - 0: confirmed tornado, 1: warned supercell, 2: severe thunderstorm, 3: negative sample
    #tn_category - TOR/WRN/NUL from tornet dataset
    #tn_lat - tornet segment centroid lat
    #tn_lon - tornet segment centroid lon
    #tn_start_time - tornet event start time
    #tn_end_time - tornet event end time
    #event_id - samples event id
    #radar_id - radar site to view the sample from
    #ef_number - ef number of a confirmed tornado sample. -1 if N/A.
    #tn_az_min - minimum azimuth to define left-most delim of tornet sector
    #tn_az_max - maximum azimuth to define right-most delim of tornet sector
    #tn_rng_min - minimum range to define inner-most delim of tornet sector
    #tn_rng_max - maximum range to define outer-most delim of tornet sector
    #tn_filename - filename of the netcdf file for the linked tornet sample
    #year - the year of the event
    #
    #phenom - tye type of warning issued for the sample (TO/SV)
    #polygon - the geoseries of the warning polygon
    #centroid - the centroid of the polygon
    def __init__(self, tornet_sample):
        self.type = None
        self.radar_data = None
        self.mesoscale_data = None
        self.phenom = None
        self.centroid = None
        self.polygon = None
        self.tn_category = tornet_sample["category"]
        self.tn_lat = tornet_sample["lat"]
        self.tn_lon = tornet_sample["lon"]
        self.tn_start_time = datetime.strptime(tornet_sample["start_time"], "%Y-%m-%d %H:%M:%S")
        self.tn_end_time = datetime.strptime(tornet_sample["end_time"], "%Y-%m-%d %H:%M:%S")
        self.event_id = tornet_sample["event_id"]
        self.radar_id = tornet_sample["site_name"]
        self.ef_number = tornet_sample["ef_number"]
        self.tn_az_min = tornet_sample["az_min"]
        self.tn_az_max = tornet_sample["az_max"]
        self.tn_rng_min = tornet_sample["rng_min"]
        self.tn_rng_max = tornet_sample["rng_max"]
        self.tn_filename = tornet_sample["filename"]
        self.year = self.tn_start_time.year

    #saves the sample to the directory
    def Save(self):
        np.savez_compressed(SAVE_DIR + self.getSampleFilename(), type=self.type, radar_data=self.radar_data, meso_data=self.mesoscale_data, ef=self.ef_number, timestamp=str(self.tn_start_time), lat=self.tn_lat, lon=self.tn_lon)

    #gets the nearest radar to the centroid of the samples polygon
    def getNearestRadarId(self, nexrad_lookup: Nexrad_Lookup):
        return nexrad_lookup.getNearestRadarId(self.centroid)
    
    #attempts to associate a historical warning polygon with the sample. Returns False if none found, Returns True and sets attributes if polygon found
    def tryLinkPolygon(self, polygon_linker: Polygon_Linker):
        linked = polygon_linker.link(self.year, self)
        if (linked == None): #no match found
            return False
        
        self.phenom = linked["PHENOM"]
        self.polygon = linked["geometry"]
        self.centroid = self.polygon.centroid
        return True
    
    #gets the required radar data from a tornet sample, and sets the radar_data attribute with the result
    def getRadarData(self, tornet_handler: TorNet_Handler, tornet_root_dir, nexrad_lookup: Nexrad_Lookup):
        data = tornet_handler.get(tornet_root_dir + self.tn_filename)
        out = {
            "DBZ": [None, None],
            "VEL": [None, None],
            "RHOHV": [None, None],
            "WIDTH": [None, None]
        }

        #for 120*120 samples
        # if (self.centroid):
            # cen_lat = self.centroid.y
            # cen_lon = self.centroid.x
        # else:
            # cen_lat = self.tn_lat
            # cen_lon = self.tn_lon

        # radar_point = nexrad_lookup.getRadarCoords(self.radar_id)
        # radar_lat = radar_point.y
        # radar_lon = radar_point.x

        # rng_min = data["segment_range_min"]
        # rng_max = data["segment_range_max"]
        # az_min = data["segment_azimuth_min"]
        # az_max = data["segment_azimuth_max"]

        
        # out["DBZ"][0] = tornet_handler.extract_matrix_patch(data["DBZ"][0], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)
        # out["DBZ"][1] = tornet_handler.extract_matrix_patch(data["DBZ"][1], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)
        # out["VEL"][0] = tornet_handler.extract_matrix_patch(data["VEL"][0], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)
        # out["VEL"][1] = tornet_handler.extract_matrix_patch(data["VEL"][1], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)
        # out["RHOHV"][0] = tornet_handler.extract_matrix_patch(data["RHOHV"][0], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)
        # out["RHOHV"][1] = tornet_handler.extract_matrix_patch(data["RHOHV"][1], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)
        # out["WIDTH"][0] = tornet_handler.extract_matrix_patch(data["WIDTH"][0], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)
        # out["WIDTH"][1] = tornet_handler.extract_matrix_patch(data["WIDTH"][1], radar_lat, radar_lon, cen_lat, cen_lon, rng_min, rng_max, az_min, az_max)

        #for full tornet sample
        out["DBZ"][0] = data["DBZ"][0]
        out["DBZ"][1] = data["DBZ"][1]
        out["VEL"][0] = data["VEL"][0]
        out["VEL"][1] = data["VEL"][1]
        out["RHOHV"][0] = data["RHOHV"][0]
        out["RHOHV"][1] = data["RHOHV"][1]
        out["WIDTH"][0] = data["WIDTH"][0]
        out["WIDTH"][1] = data["WIDTH"][1]

        self.radar_data = out

        return True
    
    #gets the mesoscale data for the sample
    def getMesoscaleData(self, hrrr_loader: HRRR_Loader):
        #get patch extraction centroid, centre of the polygon if one exists, else centre of the tornet sector
        centre = Point(self.tn_lon, self.tn_lat)
        if (self.centroid):
            centre = self.centroid

        self.mesoscale_data = hrrr_loader.getMesoData(self.tn_start_time, centre)
        

    def getSampleFilename(self):
        return f"{self.radar_id}-{self.event_id}-{self.tn_start_time.strftime('%H%M')}-{self.tn_category}.npz"

    def getSampleFilepath(self):
        return SAVE_DIR + self.getSampleFilename()