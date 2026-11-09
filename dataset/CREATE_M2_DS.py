#creates a .csv that trains the MARTAN2 random forest model

from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import csv

#iterates through all samples in the directory, and writes them to the newly created csv
def loadSamples(sample_pth, file):
    samples = os.listdir(sample_pth)
    tot_samples = len(samples)
    curr_samples = 0
    writer = csv.writer(file)
    writer.writerow(["lcl_height", "dewpoint", "mlcape", "srh_0_1km", "shear_0_1km", "shear_0_6km", "tor_intensity_class"])
    for filename in samples:
        sample_filepath = os.path.join(sample_pth, filename)
        sample = np.load(sample_filepath, allow_pickle=True)
        line = processSample(sample)
        if (line is None):
            curr_samples += 1
            continue
        writer.writerow(line)
        curr_samples += 1
        print(f"Processed item {curr_samples}/{tot_samples} ({round(curr_samples / tot_samples, 4) * 100}%)")

#processes a sample and returns an array of its indexed values
def processSample(sample):
    #check if sample has mesoscale data
    if (sample["meso_data"].item() is None):
        return None
    
    #format tor_intensity_class
    INTENSITY_CLASSES = ["none", "weak", "strong"]
    if (sample["type"] != 0):
        intensity_class = INTENSITY_CLASSES[0]
    elif (sample["type"] == 0):
        if (sample["ef"] == 0.0 or sample["ef"] == 1.0):
            intensity_class = INTENSITY_CLASSES[1]
        elif (sample["ef"] == 2.0 or sample["ef"] == 3.0 or sample["ef"] == 4.0 or sample["ef"] == 5.0):
            intensity_class = INTENSITY_CLASSES[2]
        else:
            ValueError(f"Invalid EF number {sample['ef']}")
            exit(0)
    
    return [
        sample["meso_data"].item()["lcl_height"], 
        sample["meso_data"].item()["dewpoint"], 
        sample["meso_data"].item()["mlcape"],
        sample["meso_data"].item()["srh_0_1km"],
        sample["meso_data"].item()["shear_0_1km"],
        sample["meso_data"].item()["shear_0_6km"],
        intensity_class
    ]


if (__name__ == "__main__"):
    SAMPLE_PATH = os.getenv("DATASET_SAMPLE_SAVE_DIR")
    OUT_PATH = "./MARTAN2_DS.csv"
    #make the csv
    if (os.path.isfile(OUT_PATH)):
        os.remove(OUT_PATH)

    with open(OUT_PATH, "a+", newline="") as f:
        loadSamples(SAMPLE_PATH, f)
    