#uses random forests to fit the model to the dataset
# needs to use kfold to split the dataset into 5 splits
# trains the model on each split, averages for metrics
# trains the model on all data and saves

import dotenv
dotenv.load_dotenv()
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd
import joblib

if (__name__ == "__main__"):
    DS_PTH = "./MARTAN2_DS.csv" #path of the MARTAN2 dataset
    SAVE_DIR = "./model/MARTAN2/MARTAN2.joblib" #path to save the final model

    #read dataset
    df = pd.read_csv(DS_PTH)
    feature_cols = ["lcl_height", "dewpoint", "mlcape", "srh_0_1km", "shear_0_1km", "shear_0_6km"]
    label_col = "tor_intensity_class"

    X = df[feature_cols].values
    Y = df[label_col].values

    #split using stratified kfold
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(os.getenv("KFOLD_SEED")))

    #iterate the kfolds
    classes = np.unique(Y)
    n_classes = len(classes)
    fold = 0
    precisions = []
    recalls = []
    f1s = []
    for train_idx, validate_idx in kfold.split(X, Y):
        train_features, train_labels = X[train_idx], Y[train_idx]
        validate_features, validate_labels = X[validate_idx], Y[validate_idx]

        #define model
        classifier = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=int(os.getenv("KFOLD_SEED")), n_jobs=-1)

        #fit the classifier
        classifier.fit(train_features, train_labels)
        predictions = classifier.predict(validate_features)

        p, r, f, _ = precision_recall_fscore_support(validate_labels, predictions, labels=classes, zero_division=0)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        print(f"\n==========Fold {fold}==========")
        print(classification_report(validate_labels, predictions, digits=4))
        fold += 1
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    #final metrics
    print("\n==============================")
    print("Averaged metrics:")
    print("==============================")
    for i, cls in enumerate(classes):
        print(f"Class: {cls}")
        print(f"Precision: {precisions[:, i].mean():.4f}")
        print(f"Recall: {recalls[:, i].mean():.4f}")
        print(f"F1: {f1s[:, i].mean():.4f}")
        print(f"\nMacro F1: ", f1s.mean())
    
    #train on the entire set
    print("\nFitting final model...")
    final_classifier = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=int(os.getenv("KFOLD_SEED")), n_jobs=-1)
    final_classifier.fit(X, Y)

    #save
    joblib.dump(final_classifier, SAVE_DIR)