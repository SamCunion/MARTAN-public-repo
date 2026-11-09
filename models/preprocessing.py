#radar preprocessing pipeline based on TorCast. Processes the radar snapshots as tensors, aims to clean up data, correct data, fill in gaps.

import torch
import numpy as np
from kornia.contrib import connected_components

#NORMALISATION VARS (max/min values for each field are calculated across the entire dataset)
NORMALISATION_CONSTS = {
    "DBZ": {
        "MAX": 86.0, "MIN": -32.5
    },
    "VEL": {
        "MAX": 130.0, "MIN": -130.0
    },
    "WIDTH": {
        "MEAN": 2.6579, "STD": 1.7988
    },
    "lcl_height": {
        "MEAN": 868.8040, "STD": 660.1223
    },
    "dewpoint": {
        "MEAN": 291.1823, "STD": 4.9704
    },
    "mlcape": {
        "MEAN": 6.1985, "STD": 1.9383
    },
    "srh_0_1km": {
        "MEAN": 3.6948, "STD": 2.4309
    },
    "shear_0_1km": {
        "MEAN": 10.8266, "STD": 6.911
    },
    "shear_0_6km": {
        "MEAN": 20.5644, "STD": 9.8653
    }
}

#interpolates velocity data to fill in holes where the corresponding DBZ is greater than a value. Requires DBZ and VEL input, sigma influences the size of the smoothing kernel
def interpolate_velocity_noise(DBZ, VEL, SIGMA=3.0):
    device = VEL.device

    #true where VEL is NaN but DBZ is valid
    mask = torch.isnan(VEL) & ~torch.isnan(DBZ)

    #valid-value mask 1: vel is valid, 0: vel is nan
    valid = (~torch.isnan(VEL)).float()

    #replaces nans with 0
    filled_vel = torch.nan_to_num(VEL, nan=0.0)

    #gaussian kernel
    kernel_size = int(6 * SIGMA + 1)
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    gaussian = torch.exp(-0.5 * (coords / SIGMA)**2)
    kernel = (gaussian[:, None] @ gaussian[None, :])
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    #numerator = conv(values * mask)
    #denominator = conv(mask)
    num = torch.nn.functional.conv2d(filled_vel, kernel, padding=kernel_size // 2)
    den = torch.nn.functional.conv2d(valid, kernel, padding=kernel_size // 2)

    #avoid divide by zero
    smoothed = num / torch.clamp(den, min=1e-6)

    #fill only previously invalid pixels that have been smoothed
    out = VEL.clone()
    out[0][mask[0]] = smoothed.squeeze(0)[mask[0]]

    return out

#Normalises the given matrices according to the rules of the given data type (DBZ, VEL, RHOHV, WIDTH)
def normalise_radar(type, matrix):
    device = matrix.device
    if (type == "DBZ"):
        
        #replace NaNs with 0 for DBZ
        nansafe_matrix = torch.nan_to_num(matrix, nan=0.0)

        #determine if given invalid matrix of all the same value
        min_val = torch.min(nansafe_matrix)
        max_val = torch.max(nansafe_matrix)
        if (min_val == max_val): #reject operation if min and max value are the same
            return False
        
        #clamp the DBZ to the dataset-wide max and min DBZ value
        clamped = torch.clamp(nansafe_matrix, 0, NORMALISATION_CONSTS["DBZ"]["MAX"])

        #normalise DBZ between [0, 1]
        return (clamped) / (NORMALISATION_CONSTS["DBZ"]["MAX"])
    
    elif (type == "VEL"):

        #replace NaNs with 0 for VEL
        nansafe_matrix = torch.nan_to_num(matrix, nan=0.0)

        #determine if the matrix is filed with the same value across the board, reject if this is the case (invalid input)
        abs_max = torch.max(torch.abs(nansafe_matrix))
        if (abs_max == 0): #matrix filled with nothing, reject as its invalid
            return False
        
        #clamp the VEL to the dataset-wide max and min VEL value
        clamped = torch.clamp(nansafe_matrix, NORMALISATION_CONSTS["VEL"]["MIN"], NORMALISATION_CONSTS["VEL"]["MAX"])

        #normalise VEL betwenn [-1, 1]
        return (clamped / NORMALISATION_CONSTS["VEL"]["MAX"])
    
    elif (type == "RHOHV"):
        
        #replace NaNs with 1 for RHOHV
        nansafe_matrix = torch.nan_to_num(matrix, nan=1.0)

        #clamp values between [0, 1]
        return torch.clamp(nansafe_matrix, 0.0, 1.0)

    elif (type == "WIDTH"):
        
        #replace NaNs with 0 for WIDTH
        nansafe_matrix = torch.nan_to_num(matrix, nan=0.0)

        #determine if given invalid matrix of all the same value
        min_val = torch.min(nansafe_matrix)
        max_val = torch.max(nansafe_matrix)
        if (min_val == max_val): #reject operation if min and max value are the same
            return False
        
        #clamp the WIDTH to the dataset-wide max and min WIDTH value
        clamped = torch.clamp(nansafe_matrix, min=0.0)
        logged = torch.log1p(clamped)

        #normalise WIDTH between [0, 1]
        return ((logged - NORMALISATION_CONSTS["WIDTH"]["MEAN"]) / NORMALISATION_CONSTS["WIDTH"]["STD"]).to(device)

    else:
        print("INVALID INPUT TYPE PASSED TO NORMALISE INPUT: " + type + ", EXITING PROCESS")
        exit()

#normalises the mesoscale variables
def normalise_variable(varname, x):
    if (x == -99):
        return False
    elif (varname == "lcl_height" or varname == "dewpoint" or varname == "shear_0_1km" or varname == "shear_0_6km"): #standard normalisation
        return np.clip((x - NORMALISATION_CONSTS[varname]["MEAN"]) / NORMALISATION_CONSTS[varname]["STD"], min=0.0, max=1.0)
    elif (varname == "mlcape"):
        return np.clip((np.log1p(x) - NORMALISATION_CONSTS[varname]["MEAN"]) / NORMALISATION_CONSTS[varname]["STD"], min=0.0, max=1.0)
    elif (varname == "srh_0_1km"):
        y = np.sign(x) * np.log1p(np.abs(x))
        return np.clip((y - NORMALISATION_CONSTS[varname]["MEAN"]) / NORMALISATION_CONSTS[varname]["STD"], min=-1.0, max=1.0)
    
#takes the CAM activation map from MARTAN-0 and computes the centroid (x,y) of the most "interesting" area. 
def get_cam_aoi_centroid(cam: torch.Tensor):
    DEVICE = cam.device
    #shape: (1, 120, 120)

    #threshold only where activation values greater or equal to 0.8
    masked = (cam >= 0.8).float()
    
    #get the connected components (discrete "blobs" of attention)
    blobs = connected_components(masked.unsqueeze(0))[0]
    num_blobs = int(blobs.max().item())
    if (num_blobs == 0): #no distinct areas of interest
        return None, None
    
    #score each "blob" by its sum of activations (the blob most "interesting" overall will prevail)
    scores = torch.zeros(num_blobs, device=DEVICE)
    for blob_id in range (1, num_blobs + 1):
        scores[blob_id - 1] = cam[blobs == blob_id].sum()
    best_blob = int(scores.argmax().item() + 1)

    #get the centroid of the best blob
    best_blob_mask = (blobs == best_blob)
    c, y, x = torch.nonzero(best_blob_mask, as_tuple=True)
    weights = cam[0, y, x]
    centroid_x = int((weights * x.float()).sum() / (weights.sum() + 1e-8))
    centroid_y = int((weights * y.float()).sum() / (weights.sum() + 1e-8))

    return centroid_x, centroid_y

#extracts a 50x50 patch around a centroid, works across the channel dimension.
def extract_radar_aoi(tensor: torch.Tensor, cent_x: int, cent_y: int):
    #bounds
    x1 = max(0, cent_x - 25)
    y1 = max(0, cent_y - 25)
    x2 = min(120, x1 + 50)
    y2 = min(120, y1 + 50)

    #clamp
    if ((x2 - x1) < 50):
        x1 = x2 - 50
    if ((y2 - y1) < 50):
        y1 = y2 - 50

    return tensor[:, x1:x2, y1:y2]