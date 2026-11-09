#first dataset needs to take:
# - all type 0
# - all type 1
# - (len(type0) + len(type1)) / 2 amount of type 2
# - (len(type0) + len(type1)) / 2 amount of type 3 
#
#second dataset
# - all type 0
#
#First dataset:
# - take all type 0 samples
# - group by ef number
# - randomly select 1/5 of samples for each category to "test", rest to "train"
# - take all type 1 samples
# - take 1/5 samples, put in "test", rest in "train"
# - calculate total number of type 0 and type 1 samples
# - randomly select total/2 amount of type 2 samples
# - do the 4/5 split, add to their datasets
# - randomly select total/2 amount of type 3 samples
# - do the 4/5 split, add to their datasets
# - build
#
#Second Dataset:
# - take all type 0 samples
# - group by ef number
# - randomly select 1/5 samples for each category to "test", rest to "train"
#
#Functions needed:
# - SplitByEF(df)
# - FourFiveSplit(df)
# - LoadSamples(): df
#
#USE ENVIRONMENT TFDS_BUILDER

import os
import sys
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import random
import json
from dotenv import load_dotenv
load_dotenv()

#loads samples from the given directory, extracts their relevent fields, converts to dataframe for processing and classification
def LoadSamples(dir):
    rows = []
    for filename in os.listdir(dir):
        try:
            sample_filepath = os.path.join(dir, filename)
            data = np.load(sample_filepath, allow_pickle=True)

            sample_type = data["type"].item()
            sample_ef = data["ef"].item()
            sample_meso = data["meso_data"].item()
            sample_r = data["radar_data"].item()
            if (sample_r["DBZ"][0].shape != (120, 240)):
                print(filename)
                os.remove(sample_filepath)
        except:
            print(filename)
            os.remove(sample_filepath)
            continue

        row = {
            "file": sample_filepath,
            "type": sample_type,
            "ef": sample_ef,
            "is_meso_available": sample_meso,
        }
        rows.append(row)
    return pd.DataFrame(rows)

#splits the samples by their ef field
def SplitByEF(samples):
    #ef0, 1, 2, 3, 4 (no 5 in dataset)
    efs = [[], [], [], [], []]
    for i in range(0, 5):
        efs[i] = [s for s in samples if s["ef"] == i]
    return efs

#splits the dataframe into two dataframes, one containing training samples, one the contains testing samples
def FourFiveSplit(samples, seed: int):
    random.seed(seed)
    random.shuffle(samples)
    #splits the dataset at the 4/5 point
    split_i = int(len(samples) * 0.8)
    train_df = samples[:split_i]
    test_df = samples[split_i:]
    return train_df, test_df

class Martan0(tfds.core.BuilderConfig):
    def __init__(self, train_paths, test_paths, dataset_type, **kwargs):
        super().__init__(**kwargs)
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.dataset_type = dataset_type
    
class Martan1(tfds.core.BuilderConfig):
    def __init__(self, train_paths, test_paths, dataset_type, **kwargs):
        super().__init__(**kwargs)
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.dataset_type = dataset_type


class MARTAN_Dataset(tfds.core.GeneratorBasedBuilder):


    VERSION = tfds.core.Version("1.0.0")


    BUILDER_CONFIGS = [
        Martan0(
            name = "Martan0",
            dataset_type = 0,
            train_paths = [],
            test_paths = [],
            description = "Dataset for training MARTAN-0"
        ),
        Martan1(
            name="Martan1",
            dataset_type = 1,
            train_paths = [],
            test_paths = [],
            description = "Dataset for training MARTAN-1"
        )
    ]

    def _info(self):
        ds_type = self.builder_config.dataset_type
        if (ds_type == 0):
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict({
                    "radar": {
                        "DBZ": tfds.features.Tensor(shape=(1, 120, 240), dtype=np.float32),
                        "VEL": tfds.features.Tensor(shape=(1, 120, 240), dtype=np.float32),
                        #"RHOHV": tfds.features.Tensor(shape=(1, 120, 120), dtype=np.float32),
                        #"WIDTH": tfds.features.Tensor(shape=(1, 120, 120), dtype=np.float32),
                    },
                    "type": tfds.features.Scalar(np.int32)
                }
            ))
        elif (ds_type == 1):
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict({
                    "radar": {
                        "DBZ": tfds.features.Tensor(shape=(2, 120, 240), dtype=np.float32),
                        "VEL": tfds.features.Tensor(shape=(2, 120, 240), dtype=np.float32),
                        "RHOHV": tfds.features.Tensor(shape=(2, 120, 240), dtype=np.float32),
                        "WIDTH": tfds.features.Tensor(shape=(2, 120, 240), dtype=np.float32),
                    },
                    "type": tfds.features.Scalar(np.int32),
                }
            ))
    
    def _split_generators(self, dl_manager):
        cfg = self.builder_config
        return {
            "train": self._generate_examples(cfg.train_paths),
            "test": self._generate_examples(cfg.test_paths)
        }
    
    def _generate_examples(self, file_paths):
        ds_type = self.builder_config.dataset_type
        for path in file_paths:
            data = np.load(path, allow_pickle=True)

            radar_data = data["radar_data"].item()
            type = data["type"].item()

            radar_dict = {}
            if (ds_type == 0):
                radar_fields = ["DBZ", "VEL"] #fields needed for martan-0 dataset
            elif (ds_type == 1):
                radar_fields = ["DBZ", "VEL", "RHOHV", "WIDTH"] #fields needed for martan-1 dataset
            for field in radar_fields:
                if (ds_type == 0):
                    radar_dict[field] = np.array([radar_data[field][0]]).astype(np.float32)
                elif (ds_type == 1):
                    radar_dict[field] = np.stack((radar_data[field][0], radar_data[field][1])).astype(np.float32)

            if (ds_type == 0):
                yield path, {
                    "radar": radar_dict,
                    "type": type
                }
            elif (ds_type == 1):
                yield path, {
                    "radar": radar_dict,
                    "type": type
                }


if (__name__ == "__main__"):
    #entry
    SAMPLE_PATH = os.getenv("DATASET_SAMPLE_SAVE_DIR")
    SEED = int(os.getenv("TFDS_SEED"))
    random.seed(SEED)
    DATASET_TYPE = int(sys.argv[1])
    if (not(DATASET_TYPE == 0 or DATASET_TYPE == 1)):
        print("Specify the dataset type to build: 0 for MARTAN-0, 1 for MARTAN-1")
        exit(0)

    samples = LoadSamples(SAMPLE_PATH)
    samples = samples.to_dict("records")
    training_samples = []
    testing_samples = []
    if (DATASET_TYPE == 0): #MARTAN-0
        dataset_name="MARTAN0DS"
        cfg = Martan0
        total_type_0_1 = 0
        #get all type 0 samples
        t0 = [s for s in samples if s["type"] == 0]
        total_type_0_1 += len(t0)
        #split the confirmed tornado samples (type 0) by ef number
        f0, f1, f2, f3, f4 = SplitByEF(t0)
        #split these samples into train/test to ensure even distribution
        f0x, f0y = FourFiveSplit(f0, SEED)
        f1x, f1y = FourFiveSplit(f1, SEED)
        f2x, f2y = FourFiveSplit(f2, SEED)
        f3x, f3y = FourFiveSplit(f3, SEED)
        f4x, f4y = FourFiveSplit(f4, SEED)
        training_samples.extend(f0x+f1x+f2x+f3x+f4x)
        testing_samples.extend(f0y+f1y+f2y+f3y+f4y)
        #get all type 1 samples
        t1 = [s for s in samples if s["type"] == 1]
        total_type_0_1 += len(t1)
        print(f"Total positives: {total_type_0_1}")
        #split type 1 samples into training and test
        t1x, t1y = FourFiveSplit(t1, SEED)
        training_samples.extend(t1x)
        testing_samples.extend(t1y)
        #get total_type_0_1 / 2 samples of type 2
        t2 = [s for s in samples if s["type"] == 2]
        random.seed(SEED**2)
        t2 = random.sample(t2, total_type_0_1//2)
        t2x, t2y = FourFiveSplit(t2, SEED)
        training_samples.extend(t2x)
        testing_samples.extend(t2y)
        print(f"Total type 2 samples: {len(t2x) + len(t2y)}")
        #get total_type_0_1 / 2 samples of type 3
        t3 = [s for s in samples if s["type"] == 3]
        random.seed(SEED**2)
        t3 = random.sample(t3, total_type_0_1//2)
        t3x, t3y = FourFiveSplit(t3, SEED)
        training_samples.extend(t3x)
        testing_samples.extend(t3y)
        print(f"Total type 3 samples: {len(t3x) + len(t3y)}")
    elif (DATASET_TYPE == 1): #MARTAN-1
        dataset_name="MARTAN1DS"
        cfg = Martan1
        #get all type 0 samples
        t0 = [s for s in samples if s["type"] == 0]
        #make sure the centroid exists
        print(t0[0])
        #split the confirmed tornado samples (type 0) by ef number
        f0, f1, f2, f3, f4 = SplitByEF(t0)
        #split these samples into train/test to ensure even distribution
        f0x, f0y = FourFiveSplit(f0, SEED)
        f1x, f1y = FourFiveSplit(f1, SEED)
        f2x, f2y = FourFiveSplit(f2, SEED)
        f3x, f3y = FourFiveSplit(f3, SEED)
        f4x, f4y = FourFiveSplit(f4, SEED)
        training_samples.extend(f0x+f1x+f2x+f3x+f4x)
        testing_samples.extend(f0y+f1y+f2y+f3y+f4y)
        #get all t1 samples where centroid exists
        t1 = [s for s in samples if s["type"] == 1]
        #make sure the centroid exists
        t1x, t1y = FourFiveSplit(t1, SEED)
        training_samples.extend(t1x)
        testing_samples.extend(t1y)
        #get all t2 samples where centroid exists
        t2 = [s for s in samples if s["type"] == 2]
        #make sure the centroid exists
        t2x, t2y = FourFiveSplit(t2, SEED)
        training_samples.extend(t2x)
        testing_samples.extend(t2y)
        #get all t3 samples where centroid exists
        t3 = [s for s in samples if s["type"] == 3]
        #make sure the centroid exists
        t3x, t3y = FourFiveSplit(t3, SEED)
        training_samples.extend(t3x)
        testing_samples.extend(t3y)
        #print ds contents
        print(f"Tornado samples: {len(t0)}, Warning samples: {len(t1)}, Severe Thunderstorm Warning Samples: {len(t2)}, Negative Samples: {len(t3)}")

    #shuffle the training/testing samples 
    random.shuffle(training_samples)
    random.shuffle(testing_samples)

    training_samples = [s["file"] for s in training_samples]
    testing_samples = [s["file"] for s in testing_samples]

    with open("./tfds_catalogue.json", "w+") as f:
        json.dump([training_samples, testing_samples], f)

    builder = MARTAN_Dataset(config=cfg(
        name=dataset_name,
        train_paths=training_samples,
        test_paths=testing_samples,
        dataset_type=DATASET_TYPE
    ))

    print(f"Training samples: {len(training_samples)}, Testing samples: {len(testing_samples)}")
    #builds the dataset
    builder.download_and_prepare()