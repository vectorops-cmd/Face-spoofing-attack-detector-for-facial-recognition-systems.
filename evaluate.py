# backend/evaluate.py
import os, json, numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="backend/saved_model/model.h5")
parser.add_argument("--valdir", default="backend/dataset/val")
parser.add_argument("--batch", type=int, default=16)
args = parser.parse_args()

print("Loading model:", args.model)
model = load_model(args.model, compile=False)

val_gen = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
    args.valdir, target_size=(224,224), batch_size=args.batch, class_mode="binary", shuffle=False
)

# predict all
preds = model.predict(val_gen, verbose=1)
# if preds shape is (N,1) -> sigmoid per-sample
if preds.ndim == 2 and preds.shape[1] == 1:
    probs = preds.reshape(-1)
else:
    probs = np.squeeze(preds.reshape(-1))

# default threshold 0.5; we'll compute for multiple thresholds
y_true = val_gen.classes  # 0/1 per the generator
print("Ground truth class indices mapping (generator):", val_gen.class_indices)
print("Total samples:", len(y_true))

def eval_for_threshold(th):
    y_pred = (probs >= th).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, digits=4, zero_division=0)
    acc = (y_pred == y_true).mean()
    return acc, cm, cr

for th in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
    acc, cm, cr = eval_for_threshold(th)
    print("=== threshold:", th, "accuracy:", acc)
    print("confusion matrix:\n", cm)
    print("classification report:\n", cr)
    print("-"*60)
