#handles interacting and extracting data from the tornet samples
import netCDF4
import numpy as np
from pyproj import Geod

INITIAL_SEGMENT_DIM_RNG = 120 #dataset radar patch size in the range dimension
INITIAL_SEGMENT_DIM_AZ = 120 #dataset radar patch size in the azimuth dimension


class TorNet_Handler:

    def __init__(self):
        pass

    #gets the relevent info from the sample (min/max segment range/az for centroid calculation + DBZ/VEL/RHOHV/WIDTH data for 2 tilts)
    def get(self, path):
        file = netCDF4.Dataset(path)
        return {
            "segment_range_min": file["range_limits"][0],
            "segment_range_max": file["range_limits"][1],
            "segment_azimuth_min": file["azimuth_limits"][0],
            "segment_azimuth_max": file["azimuth_limits"][1],

            "DBZ": [
                file["DBZ"][-1,:,:,0].filled(np.nan), file["DBZ"][-1,:,:,1].filled(np.nan)
            ],
            "VEL": [
                file["VEL"][-1,:,:,0].filled(np.nan), file["VEL"][-1,:,:,1].filled(np.nan)
            ],
            "RHOHV": [
                file["RHOHV"][-1,:,:,0].filled(np.nan), file["RHOHV"][-1,:,:,1].filled(np.nan)
            ],
            "WIDTH": [
                file["WIDTH"][-1,:,:,0].filled(np.nan), file["WIDTH"][-1,:,:,1].filled(np.nan)
            ]
        }
        
    #extracts a patch of a tornet segment around a warning polygon's centroid, based on the scans range and azimuth, as well as location compared to the active radar
    def extract_matrix_patch(self, matrix, radar_lat, radar_lon, centroid_lat, centroid_lon, segment_range_min, segment_range_max, segment_az_min, segment_az_max):
        #convert lat/lon to rng/az
        geod = Geod(ellps="WGS84")
        fwd_az, back_az, dist_m = geod.inv(radar_lon, radar_lat, centroid_lon, centroid_lat)
        centroid_r = dist_m
        centroid_az = (fwd_az + 360) % 360
    
        #map rng/az to matrix indexes
        n_range, n_az = matrix.shape

        #ranges are always assumed to be 60,000m,  with 60 degrees of azimuth.
        dr = (segment_range_max - segment_range_min) / n_range
        daz = (segment_az_max - segment_az_min) / n_az
    
        r_idx = int(round((centroid_r - segment_range_min) / dr))
        az_idx = int(round((centroid_az - segment_az_min) / daz)) % n_az
    
        #extract the patch according to INITIAL_SEGMENT_DIMS
        half_rng = INITIAL_SEGMENT_DIM_RNG // 2
        half_az = INITIAL_SEGMENT_DIM_AZ // 2
    
        #get the range slice. range might be clipped if the patch magnitude exceeds the maximum or falls below the minimum.
        r_start = max(r_idx - half_rng, 0)
        r_end   = min(r_start + INITIAL_SEGMENT_DIM_RNG, n_range)
        r_start = r_end - INITIAL_SEGMENT_DIM_RNG if (r_end - r_start) < INITIAL_SEGMENT_DIM_RNG else r_start
    
        #get the azimuth slice
        az_start = max(az_idx - half_az, 0)
        az_end = min(az_start + INITIAL_SEGMENT_DIM_AZ, n_az)
        az_start = az_end - INITIAL_SEGMENT_DIM_AZ if (az_end - az_start) < INITIAL_SEGMENT_DIM_AZ else az_start
    
        patch = matrix[r_start:r_end,:][:,az_start:az_end]
    
        return patch












