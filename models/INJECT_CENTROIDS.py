#uses the production version of MARTAN-0 to inject its calculated centroids into every .npz file.

import torch
import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from preprocessing import normalise_radar, interpolate_velocity_noise, get_cam_aoi_centroid
load_dotenv()

thread_local = threading.local()
counter = 0
counter_lock = threading.Lock()

def increment_counter():
    global counter
    with counter_lock:
        counter += 1
        return counter

def process_file(file_name, model_factory):
    model = get_model(model_factory)
    item = np.load(os.getenv("DATASET_SAMPLE_SAVE_DIR") + "/" + file_name, allow_pickle=True)

    if ("martan_centroid" in item): #already has a centroid, skip it
        increment_counter()
        return

    DBZ_0 = torch.from_numpy(item["radar_data"].item()["DBZ"][0]).unsqueeze(0).unsqueeze(0).to(DEVICE)
    VEL_0 = torch.from_numpy(item["radar_data"].item()["VEL"][0]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
    #do preprocessing
    VEL_0 = interpolate_velocity_noise(DBZ_0, VEL_0)
    VEL_0 = normalise_radar("VEL", VEL_0)
    if (VEL_0 == False):
        np.savez_compressed(os.getenv("DATASET_SAMPLE_SAVE_DIR") + file_name, type=item["type"], radar_data=item["radar_data"], meso_data=item["meso_data"], ef=item["ef"], timestamp=item["timestamp"], lat=item["lat"], lon=item["lon"], martan_centroid=None)
        return

    _, cam = model(VEL_0)
    cenx, ceny = get_cam_aoi_centroid(cam)
    if (cenx is None):
        np.savez_compressed(os.getenv("DATASET_SAMPLE_SAVE_DIR") + file_name, type=item["type"], radar_data=item["radar_data"], meso_data=item["meso_data"], ef=item["ef"], timestamp=item["timestamp"], lat=item["lat"], lon=item["lon"], martan_centroid=None)
    else:
        np.savez_compressed(os.getenv("DATASET_SAMPLE_SAVE_DIR") + file_name, type=item["type"], radar_data=item["radar_data"], meso_data=item["meso_data"], ef=item["ef"], timestamp=item["timestamp"], lat=item["lat"], lon=item["lon"], martan_centroid=(cenx, ceny))
    completed = increment_counter()
    print(f"Processed: {completed}/{SAMPLE_COUNT}")


def model_factory():
    model = torch.load(MARTAN0_PTH, weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    model.with_gradcam()
    return model

def get_model(model_factory):
    if not hasattr(thread_local, "model"):
        thread_local.model = model_factory()
    return thread_local.model

if (__name__ == "__main__"):
    MARTAN0_PTH = os.getenv("MARTAN_0_PROD_PTH")
    THREADS = 8
    START_AT = 0 #start half way through?
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {DEVICE_NAME}")
    DEVICE = torch.device(DEVICE_NAME)
    counter = START_AT

    FILE_LIST = os.listdir(os.getenv("DATASET_SAMPLE_SAVE_DIR"))
    SAMPLE_COUNT = len(FILE_LIST)

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = [
            executor.submit(process_file, fp, model_factory) for fp in FILE_LIST[START_AT:]
        ]
        
