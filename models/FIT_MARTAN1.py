#Continuously fits the MARTAN-1 model, saves its state each epoch for inspection.
#DS Contents:
#Type0: 12,102
#Type1: 48,785
#Type2: 30,540
#Type3: 86,819
#Train: 142,595
#Test: 35,651

import torch
import tensorflow_datasets as tfds
from TFDS_BUILDER import MARTAN_Dataset as MARTAN_Builder #REQUIRED
from torch.utils.data import DataLoader
from sklearn import metrics
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from model_defs.MARTAN0 import MARTAN_0
from model_defs.MARTAN1 import MARTAN_1
from TrainingVisualiser import TrainingVisualiser
from MARTAN_Dataset import MARTAN_Dataset
from preprocessing import normalise_radar, interpolate_velocity_noise, extract_radar_aoi

#annoying warnings mean nothing
import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable*")

#fits the model
def fit(start_epoch: int, MODEL):
    TORNADO_THRESHOLD = float(os.getenv("TORNADO_THRESHOLD"))
    SMOOTHING = 0.1
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE_NAME)
    print(f"Using torch device: {DEVICE_NAME}")
    #load dataset
    train_ds = MARTAN_Dataset(1, "train")
    test_ds = MARTAN_Dataset(1, "test")
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=2, pin_memory=True)
    train_count = len(train_loader)
    test_count = len(test_loader)
    WEIGHT = torch.tensor([(1.0 - 0.068) / 0.068], device=DEVICE)

    #load model
    if (MODEL is None):
        MODEL = MARTAN_1()
    MODEL = MODEL.to(DEVICE)

    #optimiser/loss
    optimiser = torch.optim.AdamW(MODEL.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=15)
    prob_loss = torch.nn.BCEWithLogitsLoss(pos_weight=WEIGHT, reduction="none") #biases towards positive

    #visualiser
    visualiser = TrainingVisualiser(train_count, test_count)

    #early stopping
    epoch_log_losses = []
    converged = False
    epoch = start_epoch

    #main fitting loop
    while (not converged):
        
        #start training for this epoch
        visualiser.start_phase(epoch, "TRAIN")
        MODEL.train()
        for i, batch in enumerate(train_loader):

            #preprocessing
            DBZ, VEL, RHO, WID, TYPES = preprocess_batch(batch, DEVICE)

            #reset grads
            optimiser.zero_grad()

            #process batch
            prob_logit = MODEL(DBZ, VEL, RHO, WID)

            #construct truth labels from "types" field (0s are confirmed tornados, rest arent)
            labels = torch.isin(TYPES, torch.tensor([0], device=DEVICE)).long()

            #calc loss
            loss = prob_loss(prob_logit.squeeze(dim=1), labels)

            #"lenient" loss when predicting positive on an unconfirmed, but warned sample
            prob_percent = torch.sigmoid(prob_logit.squeeze(dim=1))
            leniency_matches = (prob_percent > TORNADO_THRESHOLD) & (TYPES == 1)
            leniency_scale = torch.ones_like(labels, dtype=torch.float32, device=DEVICE)
            leniency_scale[leniency_matches] = 0.5 #"forgiveness" coefficient
            loss *= leniency_scale
            loss = loss.mean()

            #backwards pass
            loss.backward()
            scheduler.step(epoch)

            #update visual
            visualiser.update_progress(i, "TRAIN")
        
        #start testing for this epoch
        visualiser.start_phase(epoch, "TEST")
        MODEL.eval()
        tor_prob_predictions = []
        tor_prob_truths = []
        tor_prob_percentages = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):

                #preprocessing
                DBZ, VEL, RHO, WID, TYPES = preprocess_batch(batch, DEVICE)

                #construct sample labels from sample types
                labels = torch.isin(TYPES, torch.tensor([0], device=DEVICE)).long()

                #inference
                prob_logit = MODEL(DBZ, VEL, RHO, WID)

                #collects the batch's tornado probabilities, and thresholds them to separate into positive and negative identifications
                batch_tor_probs = torch.sigmoid(prob_logit)
                batch_tor_predictions = (batch_tor_probs > TORNADO_THRESHOLD).int().view(-1).cpu().numpy()
                tor_prob_predictions.extend(batch_tor_predictions)
                tor_prob_truths.extend(labels.int().cpu().numpy())
                tor_prob_percentages.extend(batch_tor_probs.cpu().numpy())

                #update visualiser
                visualiser.update_progress(i, "TEST")

            #deal with class imbalance in evaluation metrics
            class_counts = np.bincount(tor_prob_truths)
            class_weights = 1 / class_counts
            sample_weights = class_weights[tor_prob_truths]

            tn, fp, fn, tp = metrics.confusion_matrix(tor_prob_truths, tor_prob_predictions).ravel()
            ll = metrics.log_loss(tor_prob_truths, tor_prob_percentages, sample_weight=sample_weights)
            #normalise the confusion matrix (out of 1000)
            ntn = round(tn / (tn + fp) * 1000)
            nfp = round(fp / (fp + tn) * 1000)
            nfn = round(fn / (tp + fn) * 1000)
            ntp = round(tp / (tp + fn) * 1000)
            #calculate normalise metrics
            accuracy = (ntn + ntp) / (ntp + nfp + ntn + nfn)
            precision = ntp / (ntp + nfp)
            recall = ntp / (ntp + nfn)
            f1 = 2 * ((precision * recall) / (precision + recall))
        
            epoch_log_losses.append(ll)

            #update visualiser graphs
            visualiser.update_metrics(ll, accuracy, precision, recall, f1)

            #update summary file
            with open(f"{SAVE_DIR}MARTAN1-summary.txt", "a+") as f:
                f.write(f"\nEpoch {epoch} yielded:\nLearning Rate: {round(str(optimiser.param_groups[0]["lr"]), 6)}\nLog Loss: {str(round(ll, 4))}\nAccuracy: {str(round(accuracy, 4))}\nPrecision: {str(round(precision, 4))}\nRecall: {str(round(recall, 4))}\nF1: {str(round(f1, 4))}\nTP: {str(tp)}\nFP: {str(fp)}\nTN: {str(tn)}\nFN: {str(fn)}\n")
    
            torch.save(MODEL, f"{SAVE_DIR}e{str(epoch)}-MARTAN1.pt")

            #check for stop
            if (len(epoch_log_losses) > 5 and (epoch_log_losses[-1] > epoch_log_losses[-4] and epoch_log_losses[-2] > epoch_log_losses[-4] and epoch_log_losses[-3] > epoch_log_losses[-4])):
                #converged = True
                print(f"Optimal LL reached: {epoch_log_losses[-4]}")
    
        epoch += 1

#takes the raw torch batch, performs preprocessing, and returns the post-processed batch ready for training/inference, as well as the indexed sample labels
def preprocess_batch(batch, DEVICE):
    batch_size = len(batch["type"])
    #preprocessing
    #separate batch into items
    BATCH_DBZ = batch["radar"]["DBZ"].to(DEVICE)
    BATCH_VEL = batch["radar"]["VEL"].to(DEVICE)
    BATCH_RHO = batch["radar"]["RHOHV"].to(DEVICE)
    BATCH_WID = batch["radar"]["WIDTH"].to(DEVICE)
    
    BATCH_TYPE = batch["type"].to(DEVICE)
    
    SPLIT_DBZ = []
    SPLIT_VEL = []
    SPLIT_RHO = []
    SPLIT_WID = []
    SPLIT_TYPE = []
    
    for j in range(0, batch_size): #preprocess individual sample within batch
        dbz_0_data = BATCH_DBZ[j][0].unsqueeze(0)
        vel_0_data = BATCH_VEL[j][0].unsqueeze(0)
        rho_0_data = BATCH_RHO[j][0].unsqueeze(0)
        wid_0_data = BATCH_WID[j][0].unsqueeze(0)
        dbz_1_data = BATCH_DBZ[j][1].unsqueeze(0)
        vel_1_data = BATCH_VEL[j][1].unsqueeze(0)
        rho_1_data = BATCH_RHO[j][1].unsqueeze(0)
        wid_1_data = BATCH_WID[j][1].unsqueeze(0)
        t = BATCH_TYPE[j]
    
        vel_0_data = interpolate_velocity_noise(dbz_0_data, vel_0_data)
        vel_1_data = interpolate_velocity_noise(dbz_1_data, vel_1_data)
    
        if (isinstance(vel_0_data, bool) or isinstance(vel_1_data, bool)):
            continue
    
        dbz_0_data = normalise_radar("DBZ", dbz_0_data)
        dbz_1_data = normalise_radar("DBZ", dbz_1_data)
        vel_0_data = normalise_radar("VEL", vel_0_data)
        vel_1_data = normalise_radar("VEL", vel_1_data)
        rho_0_data = normalise_radar("RHOHV", rho_0_data)
        rho_1_data = normalise_radar("RHOHV", rho_1_data)
        wid_0_data = normalise_radar("WIDTH", wid_0_data)
        wid_1_data = normalise_radar("WIDTH", wid_1_data)
    
        if (isinstance(dbz_0_data, bool) or isinstance(dbz_1_data, bool) or 
            isinstance(vel_0_data, bool) or isinstance(vel_1_data, bool) or
            isinstance(rho_0_data, bool) or isinstance(rho_1_data, bool) or
            isinstance(wid_0_data, bool) or isinstance(wid_1_data, bool)
        ): continue
    
        #recombine
        DBZ_STACK = torch.cat([dbz_0_data, dbz_1_data], dim=0)
        VEL_STACK = torch.cat([vel_0_data, vel_1_data], dim=0)
        RHO_STACK = torch.cat([rho_0_data, rho_1_data], dim=0)
        WID_STACK = torch.cat([wid_0_data, wid_1_data], dim=0)
    
        #extract 50x50 patch
        #DBZ_STACK = extract_radar_aoi(DBZ_STACK, cenx, ceny)
        #VEL_STACK = extract_radar_aoi(VEL_STACK, cenx, ceny)
        #RHO_STACK = extract_radar_aoi(RHO_STACK, cenx, ceny)
        #WID_STACK = extract_radar_aoi(WID_STACK, cenx, ceny)
    
        SPLIT_DBZ.append(DBZ_STACK)
        SPLIT_VEL.append(VEL_STACK)
        SPLIT_RHO.append(RHO_STACK)
        SPLIT_WID.append(WID_STACK)
        SPLIT_TYPE.append(t)
    
    #re-merge batch
    DBZ = torch.stack(SPLIT_DBZ).to(DEVICE)
    VEL = torch.stack(SPLIT_VEL).to(DEVICE)
    RHO = torch.stack(SPLIT_RHO).to(DEVICE)
    WID = torch.stack(SPLIT_WID).to(DEVICE)
    TYPES = torch.stack(SPLIT_TYPE).to(DEVICE)

    return DBZ, VEL, RHO, WID, TYPES

if (__name__ == "__main__"):
    SAVE_DIR = os.getenv("MARTAN_1_SAVE_PTH")
    print("MARTAN-1 model fitter")

    print("\nNew model, or load model?")

    start_epoch = 1

    ans = input("New/Load: ")
    if ans.lower() == "new": #uses the default model loaded earlier, with randomised weights. Starts at epoch 0 (1)
        fit(start_epoch, None)
    elif ans.lower() == "load": #reads the directory, creates a simple file selector interface
        saved_model_list = os.listdir(SAVE_DIR)
        print_str = "Select model to load:\n"
        for i in range(0, len(saved_model_list)):
            print_str += "[" + str(i) + "] " + saved_model_list[i] + "\n"
        print(print_str)
        saved_model_id = int(input("Model ID: "))
        model_name = saved_model_list[saved_model_id]
        MODEL = torch.load(SAVE_DIR + model_name, weights_only=False) #load the entire saved model params
        start_epoch = int(input("Last Epoch: ")) + 1
        fit(start_epoch, MODEL) #starts with the loaded model, and defined starting epoch
    else:
        exit()