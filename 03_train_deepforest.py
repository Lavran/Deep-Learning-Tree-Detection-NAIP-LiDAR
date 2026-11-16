import os
import shutil
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from deepforest import main

import config

# inputs (written by 02_prepare_training_data.py; every path is set in config.py)
data_dir = str(config.TRAIN_DIR)            # NAIP patches (blanked where the CHM has no data) and fold tables

# outputs: deepforest_naip_cv_fold{k}_best.ckpt (and _last.ckpt) for each fold in model_dir, and the loss /
# validation metrics of each fold in log_dir, under deepforest_naip_cv_fold{k}/version_N/metrics.csv
model_dir = str(config.MODEL_DIR)
log_dir = str(config.LOG_DIR)
os.makedirs(model_dir, exist_ok=True)
run_name = "deepforest_naip_cv"

# settings
n_folds = 5
existing = {2: os.path.join(model_dir, "deepforest_naip_run4_best.ckpt")}
                        # fold 2 has exactly run 4's training, validation and test tiles (see 02), and run 4
                        # used these same settings, so its model is reused for fold 2 instead of training it
                        # again (about 4 hours). Set existing = {} to train every fold.
redo = False            # False: skip folds that are already done, so if training stops part-way, running the
                        # script again carries on from the next unfinished fold. True: train every fold again.


for k in range(n_folds):
    best_path = os.path.join(model_dir, f"{run_name}_fold{k}_best.ckpt")
    done_path = os.path.join(model_dir, f"{run_name}_fold{k}_done.txt")   # written when a fold finishes
    if os.path.exists(done_path) and not redo:
        print(f"fold {k}: already trained ({best_path}), skipping (set redo = True to train it again)")
        continue
    if k in existing and os.path.exists(existing[k]) and not redo:
        shutil.copyfile(existing[k], best_path)
        with open(done_path, "w") as f:
            f.write(f"copied from {existing[k]} (same tiles and settings)\n")
        print(f"fold {k}: same tiles as {existing[k]}, copied it to {best_path}")
        continue
    print(f"\n######## fold {k}: training on {os.path.join(data_dir, f'fold{k}_train.csv')} ########")

    # ############################
    # 1. Start from the pretrained tree model
    # ############################
    # nms_thresh: after predicting, DeepForest drops any box that overlaps a higher-scoring box
    # by more than this IoU. The default (0.05) is tuned for open woodland; in this canopy 70% of
    # the watershed boxes overlap a neighbour by more than that, so use 0.3. It has to be set
    # before load_model, which builds the model with it.
    m = main.deepforest(config_args={"nms_thresh": 0.3})
    m.load_model("weecology/deepforest-tree")

    # ############################
    # 2. Point it at this fold's NAIP patches
    # ############################
    m.config.train.csv_file = os.path.join(data_dir, f"fold{k}_train.csv")
    m.config.train.root_dir = os.path.join(data_dir, "patches")
    m.config.validation.csv_file = os.path.join(data_dir, f"fold{k}_val.csv")
    m.config.validation.root_dir = os.path.join(data_dir, "patches")

    m.config.train.epochs = 10
    m.config.batch_size = 2                          # same as run 4 (fold 2); use 1 if CUDA runs out of memory
    m.config.workers = 0                             # keep 0 when running in Positron's console
    m.config.validation.val_accuracy_interval = 1    # box precision/recall on validation every epoch

    # ############################
    # 3. Train, keeping the best epoch
    # ############################
    # keep the epoch with the highest mAP@50 on the validation tiles. (In run 2, val_loss was
    # lowest at epoch 2 while precision, recall and mAP kept improving until epochs 8-9, so
    # val_loss picked the wrong epoch.) save_last also keeps the final epoch. An unfinished
    # earlier attempt at this fold is overwritten rather than saved alongside as "-v1".
    best_epoch = ModelCheckpoint(dirpath=model_dir, filename=f"{run_name}_fold{k}_best",
                                 monitor="map_50", mode="max", save_top_k=1, save_last=True,
                                 enable_version_counter=False)
    best_epoch.CHECKPOINT_NAME_LAST = f"{run_name}_fold{k}_last"

    # gradient_clip_val caps how much one batch can change the weights, so a single bad batch
    # can't push the loss to nan (what happened in run 1, partway through epoch 8)
    m.create_trainer(gradient_clip_val=1.0, callbacks=[best_epoch],
                     logger=CSVLogger(log_dir, name=f"{run_name}_fold{k}"))
    m.trainer.fit(m)

    with open(done_path, "w") as f:
        f.write(f"best epoch mAP@50 {float(best_epoch.best_model_score):.3f}\n")
    print(f"fold {k}: best epoch mAP@50 {float(best_epoch.best_model_score):.3f} -> {best_epoch.best_model_path}")

    del m                       # free the GPU before the next fold
    torch.cuda.empty_cache()
