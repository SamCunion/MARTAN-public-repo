#Main file that builds the dataset from principles 
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from HRRR_Loader import HRRR_Loader
from Nexrad_Lookup import Nexrad_Lookup
from Polygon_Linker import Polygon_Linker
from TorNet_Handler import TorNet_Handler
from Sample import Sample


if (__name__ == "__main__"):
    #load the TorNet catalogue into a dataframe
    #for each sample:
    # - check if its filename exists in the /dataset directory
    # - if so, skip, else attempt to find a polygon
    # - if no polygon found for TOR/WRN, skip it
    # - if it has a linked polygon, get the radar segment from the centroid of the polygon. else get it from the centroid of the scan.
    # - get the mesoscale info for the sample (asyncio)
    # - save the sample

    TORNET_CATALOGUE_PTH = os.getenv("TORNET_CATALOGUE_PTH") #path to the catalogue.csv file provided by TorNet
    TORNET_DATA_PTH = os.getenv("TORNET_DATA_PTH") #root directory of the actual TorNet dataset (the one that contains test and train)
    START_AT_SAMPLE = 0 #start at this sample, cuts down a lot of checking if previous samples have been done

    #load helpers
    hrrr_loader = HRRR_Loader()
    nexrad_lookup = Nexrad_Lookup()
    polygon_linker = Polygon_Linker()
    tornet_handler = TorNet_Handler()
    
    #load the tornet catalogue
    tornet = pd.read_csv(TORNET_CATALOGUE_PTH)
    tot_tornet_samples = len(tornet)
    tornet_samples_processed = START_AT_SAMPLE

    #loop through the tornet samples
    for i, sample in tornet.iterrows():
        if (i < START_AT_SAMPLE): #quickly skip past the first n samples
            continue

        s = Sample(sample.to_dict())
        
        if (os.path.exists(s.getSampleFilepath())): #sample already processed, skip
            tornet_samples_processed += 1
            continue
        
        if (not s.tryLinkPolygon(polygon_linker)): #no matching polygon from IEM found
            if (s.tn_category == "TOR" or s.tn_category == "WRN"): #if its a tornado warned or tornado confirmed sample, skip it
                tornet_samples_processed += 1
                continue
        
        #sets the samples radar data
        s.getRadarData(tornet_handler, TORNET_DATA_PTH, nexrad_lookup)

        #get the meso data
        s.getMesoscaleData(hrrr_loader)

        #set the type of the sample correctly
        if (s.tn_category == "TOR"):
            s.type = 0
        elif (s.tn_category == "WRN"):
            s.type = 1
        elif (s.phenom == "SV"):
            s.type = 2
        else:
            s.type = 3

        #save the sample
        s.Save()
        del s
        tornet_samples_processed += 1
        print(f"Processed sample {tornet_samples_processed}/{tot_tornet_samples} ({(tornet_samples_processed / tot_tornet_samples) * 100}%)")