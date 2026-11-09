#Continuously fits the MARTAN-0 model, saves its state each epoch for inspection.

import torch
import tensorflow_datasets as tfds
from TFDS_BUILDER import MARTAN_Dataset as MARTAN_Builder #REQUIRED
from torch.utils.data import DataLoader
from sklearn import metrics
import os
from dotenv import load_dotenv
load_dotenv()
from model_defs.MARTAN0 import MARTAN_0
from TrainingVisualiser import TrainingVisualiser
from MARTAN_Dataset import MARTAN_Dataset
from preprocessing import normalise_radar, interpolate_velocity_noise

#annoying warnings mean nothing
import warnings
warnings.filterwarnings("ignore", message="The given NumPy array is not writable*")

#fits the model
def fit(start_epoch: int, MODEL):
    TORNADO_THRESHOLD = float(os.getenv("TORNADO_THRESHOLD"))
    DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE_NAME)
    print(f"Using torch device: {DEVICE_NAME}")
    #load dataset
    train_ds = MARTAN_Dataset(0, "train")
    test_ds = MARTAN_Dataset(0, "test")
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=4, pin_memory=True)
    train_count = len(train_loader)
    test_count = len(test_loader)

    #load model
    if (MODEL is None):
        MODEL = MARTAN_0()
    MODEL = MODEL.to(DEVICE)
    optimiser = torch.optim.AdamW(MODEL.parameters())
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean").to(DEVICE)

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
            INPUT_STACK, labels = preprocess_batch(batch, DEVICE)

            #reset grads
            optimiser.zero_grad()

            #process batch
            prob_logit = MODEL(INPUT_STACK)

            #calc loss
            prob_loss = loss(prob_logit.squeeze(dim=1), labels.float())

            #backwards pass
            prob_loss.backward()
            optimiser.step()

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
                INPUT_STACK, labels = preprocess_batch(batch, DEVICE)

                #inference
                prob_logit = MODEL(INPUT_STACK)

                #collects the batch's tornado probabilities, and thresholds them to separate into positive and negative identifications
                batch_tor_probs = torch.sigmoid(prob_logit)
                batch_tor_predictions = (batch_tor_probs > TORNADO_THRESHOLD).int().view(-1).cpu().numpy()
                tor_prob_predictions.extend(batch_tor_predictions)
                tor_prob_truths.extend(labels.int().cpu().numpy())
                tor_prob_percentages.extend(batch_tor_probs.cpu().numpy())

                #update visualiser
                visualiser.update_progress(i, "TEST")

        ll = metrics.log_loss(tor_prob_truths, tor_prob_percentages)
        accuracy = metrics.accuracy_score(tor_prob_truths, tor_prob_predictions)
        precision = metrics.precision_score(tor_prob_truths, tor_prob_predictions)
        recall = metrics.recall_score(tor_prob_truths, tor_prob_predictions)
        f1 = metrics.f1_score(tor_prob_truths, tor_prob_predictions)
        tn, fp, fn, tp = metrics.confusion_matrix(tor_prob_truths, tor_prob_predictions).ravel()
        epoch_log_losses.append(ll)

        #update visualiser graphs
        visualiser.update_metrics(ll, accuracy, precision, recall, f1)

        #update summary file
        with open(f"{SAVE_DIR}MARTAN0-summary.txt", "a+") as f:
            f.write(f"\nEpoch {epoch} yielded:\nLog Loss: {str(round(ll, 4))}\nTP: {str(tp)}\nFP: {str(fp)}\nTN: {str(tn)}\nFN: {str(fn)}\n")
    
        torch.save(MODEL, f"{SAVE_DIR}e{str(epoch)}-MARTAN0.pt")

        #check for stop
        if (len(epoch_log_losses) > 5 and (epoch_log_losses[-1] > epoch_log_losses[-4] and epoch_log_losses[-2] > epoch_log_losses[-4] and epoch_log_losses[-3] > epoch_log_losses[-4])):
            converged = True
            print(f"Optimal LL reached: {epoch_log_losses[-4]}")
    
        epoch += 1

#takes the raw torch batch, performs preprocessing, and returns the post-processed batch ready for training/inference, as well as the indexed sample labels
def preprocess_batch(batch, DEVICE):

    batch_size = len(batch["type"])

    #preprocessing
    #separate batch into items
    BATCH_DBZ = batch["radar"]["DBZ"].to(DEVICE) #DBZ used for preprocessing only (for now)
    BATCH_VEL = batch["radar"]["VEL"].to(DEVICE)
    #BATCH_WID = batch["radar"]["WIDTH"].to(DEVICE)
    
    BATCH_TYPE = batch["type"].squeeze().long().to(DEVICE)
    SPLIT_TYPE = []
    SPLIT_STACK = []
    
    for j in range(0, batch_size): #preprocess individual sample within batch
        dbz_0_data = BATCH_DBZ[j]
        vel_0_data = BATCH_VEL[j]
        #wid_0_data = BATCH_WID[j]
        type_data = BATCH_TYPE[j]
    
        vel_0_interpolated = interpolate_velocity_noise(dbz_0_data, vel_0_data)
    
        if (isinstance(vel_0_interpolated, bool) and vel_0_interpolated == False):
            continue
    
        vel_0_norm = normalise_radar("VEL", vel_0_interpolated)
    
        if (isinstance(vel_0_norm, bool) and vel_0_norm == False):
            continue
    
        dbz_0_norm = normalise_radar("DBZ", dbz_0_data)
    
        if (isinstance(dbz_0_norm, bool) and dbz_0_norm == False):
            continue
    
        #recombine
        STACK = torch.cat([dbz_0_norm, vel_0_norm], dim=0)
        SPLIT_STACK.append(STACK)
        SPLIT_TYPE.append(type_data)
    
    #re-merge batch
    INPUT_STACK = torch.stack(SPLIT_STACK).to(DEVICE)
    
    #stack the radar tilts
    TYPES = torch.stack(SPLIT_TYPE).to(DEVICE)
    
    #construct truth labels from "types" field (0s are confirmed tornados, rest arent)
    labels = torch.isin(TYPES, torch.tensor([0, 1], device=DEVICE)).long().to(DEVICE)

    return INPUT_STACK, labels

if (__name__ == "__main__"):
    SAVE_DIR = os.getenv("MARTAN_0_SAVE_PTH")
    print("MARTAN-0 model fitter")

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