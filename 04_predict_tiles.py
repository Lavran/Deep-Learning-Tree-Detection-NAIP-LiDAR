import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box
from deepforest import main

import config

# inputs (every path is set in config.py)
model_paths = os.path.join(config.MODEL_DIR, "deepforest_naip_cv_fold{}_best.ckpt")   # trained model of each fold (from 03)
naip_path = str(config.NAIP_PATH)
tiles_path = str(config.TILES_PATH)        # tiles and folds from 02
crowns_path = str(config.CROWNS_PATH)
tops_path = str(config.TOPS_PATH)
chm_path = str(config.CHM_PATH)

# outputs (all inside this folder)
out_dir = str(config.PRED_DIR)
# trained/fold{k}: each fold's model on its own test tiles
# pretrained/fold{k}: DeepForest as downloaded on the same tiles (the baseline)
# pretrained_patch_choice/fold{k}_patch{size}: the baseline on that fold's validation tiles, to pick its patch size
# naip_tiles: the masked NAIP tiles that were predicted
# cv_scores.csv, cv_scores_by_fold.csv, cv_tile_scores.csv, patch_size_choice.csv: the cross-validated results

# settings
n_folds = 5
patch_size = 200        # trained models: same as training
baseline_patch_sizes = [100, 200, 400]  # baseline: each tried on the fold's validation tiles; the best is then
                                        # scored once on the fold's test tiles. The pretrained model learned on
                                        # ~10 cm imagery, so a smaller patch (enlarged more inside the model) may
                                        # suit it better.
choose_by = "treetop_F1"    # tile_scores.csv column used to pick the baseline patch size (treetop = 05's main criterion)
min_height = 5.0        # same crowns as the training labels
patch_overlap = 0.1
merge_iou = 0.3         # predictions from overlapping patches are merged above this IoU (DeepForest's
                        # default 0.15 is too strict for this canopy, same issue as nms_thresh)
min_score = 0.1         # ignore predictions below this confidence (raise it to trade recall for precision)
edge = 10               # m; only score trees and predictions at least this far inside the tile edge
iou_levels = [0.5, 0.4] # box-overlap thresholds for a match. 0.5 is the standard in object detection (Everingham
                        # et al. 2010) and is widely used for tree crowns; 0.4 is DeepForest's own default.
with_treetop = True     # also score each threshold with the extra requirement that the watershed treetop lies
                        # inside the predicted box ("both50", "both40"): the box must fit the crown AND contain its top
redo = False            # False: a run is skipped, and its scores read back, if its folder already has a
                        # tile_scores.csv holding every criterion above (so this can be run again as more folds
                        # finish training). True: predict everything again (do this after changing any other setting).


# ############################
# 1. Matching criteria (each one-to-one: a prediction is matched to at most one watershed tree, and a
#    watershed tree to at most one prediction)
# ############################
def iou_match(pred, ref_boxes, thr, tops=None):
    """Box overlap (Everingham et al. 2010): predictions, most confident first, each take the unmatched
    watershed box they overlap most, if IoU >= thr. If `tops` is given (the treetops, in the same order as
    ref_boxes), the treetop must also lie inside the predicted box."""
    p, r = ref_boxes.sindex.query(pred.geometry, predicate="intersects")
    a, b = np.asarray(pred.geometry.values)[p], np.asarray(ref_boxes.geometry.values)[r]
    inter = shapely.area(shapely.intersection(a, b))
    iou = inter / (shapely.area(a) + shapely.area(b) - inter)
    ok = iou >= thr
    if tops is not None:
        ok &= shapely.contains(a, np.asarray(tops.geometry.values)[r])
    pairs = pd.DataFrame({"p": p, "r": r, "iou": iou, "score": pred.score.values[p]})
    pairs = pairs[ok].sort_values(["score", "iou"], ascending=False)
    return greedy(pairs)

def treetop_match(pred, tops):
    """Treetop-in-box, the LiDAR version of the field-stem recall of Weinstein et al. (2021): a prediction
    is a hit if a watershed treetop falls inside its box (the treetop closest to the box centre, if several)."""
    p, r = tops.sindex.query(pred.geometry, predicate="contains")
    d = shapely.distance(np.asarray(pred.geometry.centroid.values)[p], np.asarray(tops.geometry.values)[r])
    pairs = pd.DataFrame({"p": p, "r": r, "d": d, "score": pred.score.values[p]})
    pairs = pairs.sort_values(["score", "d"], ascending=[False, True])
    return greedy(pairs)

def greedy(pairs):
    used_p, used_r, keep = set(), set(), []
    for p, r in zip(pairs.p, pairs.r):
        if p in used_p or r in used_r:
            continue
        used_p.add(p); used_r.add(r); keep.append((p, r))
    return np.array(keep, dtype=int).reshape(-1, 2)


# ############################
# 2. Watershed reference trees and tiles
# ############################
crowns_all = gpd.read_file(crowns_path)
tops_all = gpd.read_file(tops_path)
crowns_all = crowns_all.merge(tops_all[["crown_id", "height"]], on="crown_id")
crowns_all = crowns_all[crowns_all.height >= min_height]
tops_all = tops_all[tops_all.crown_id.isin(crowns_all.crown_id)]

all_tiles = gpd.read_file(tiles_path)
methods = ([f"iou{int(t * 100)}" for t in iou_levels]                              # box overlap
           + ([f"both{int(t * 100)}" for t in iou_levels] if with_treetop else [])  # box overlap and treetop in box
           + ["treetop"])                                                           # treetop in box
shift_from = f"iou{int(min(iou_levels) * 100)}"     # matches used for the shift-by-height table


# ############################
# 3. Predict and score: one run = one model, some tiles, one patch size, one output folder
# ############################
def predict_and_score(m, label, tiles, patch_size, out_dir, tile_dir):
    """Predict `tiles` (a table of tiles with a `split` column: their role in this fold) with model `m`,
    score against the watershed trees, and write tile_scores.csv, the recall tables and the shapefiles to
    out_dir. Returns the tile scores."""
    scores_path = os.path.join(out_dir, "tile_scores.csv")
    if os.path.exists(scores_path) and not redo:
        old = pd.read_csv(scores_path)
        if all(f"{n}_F1" in old.columns for n in methods):
            print(f"\n{out_dir} already has tile_scores.csv: skipping it (set redo = True to predict again)")
            return old
        print(f"\n{out_dir} was scored with different matching criteria: predicting it again")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tile_dir, exist_ok=True)

    tiles = tiles.reset_index(drop=True)
    splits = list(tiles.split.unique())
    print(f"\nmodel: {label}  (overlap filter nms_thresh = {m.config.nms_thresh}, patch size {patch_size} px)")
    print(f"scoring {len(tiles)} tiles: {', '.join(splits)} -> {out_dir}")

    # 3a. Predict each tile and score it
    tile_rows, pred_tables, crown_tables, offsets = [], [], [], []
    with rasterio.open(naip_path) as src, rasterio.open(chm_path) as chm_src:
        res = src.res[0]
        for t in tiles.itertuples():
            # NAIP for this tile (same window as 02), saved so it can be opened in QGIS
            w = from_bounds(*t.geometry.bounds, transform=src.transform).round_offsets().round_lengths()
            tile_transform = src.window_transform(w)
            tile_tif = os.path.join(tile_dir, f"{t.tile}.tif")
            profile = src.profile.copy()
            profile.update(width=w.width, height=w.height, count=3, transform=tile_transform)
            # NAIP blanked where the CHM has no data, exactly as in 02
            img = src.read([1, 2, 3], window=w)
            chm = chm_src.read(1, window=from_bounds(*window_bounds(w, src.transform), transform=chm_src.transform),
                               out_shape=(w.height, w.width), resampling=Resampling.nearest)
            no_lidar = ~(np.isfinite(chm) & (chm > -100))
            img[:, no_lidar] = 0
            with rasterio.open(tile_tif, "w", **profile) as dst:
                dst.write(img)

            # predict, then convert boxes from tile pixels to map coordinates
            p = m.predict_tile(path=tile_tif, patch_size=patch_size, patch_overlap=patch_overlap,
                               iou_threshold=merge_iou)
            left, top = tile_transform.c, tile_transform.f
            if p is None:
                p = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score"])
            p = p[p.score >= min_score]
            pred = gpd.GeoDataFrame(
                {"tile": t.tile, "score": p.score.values},
                geometry=[box(left + x0 * res, top - y1 * res, left + x1 * res, top - y0 * res)
                          for x0, y0, x1, y1 in zip(p.xmin, p.ymin, p.xmax, p.ymax)],
                crs=crowns_all.crs).reset_index(drop=True)

            # watershed trees in this tile (by treetop), as crowns, boxes and treetops
            tile_tops = tops_all[tops_all.within(t.geometry)]
            crowns = crowns_all[crowns_all.crown_id.isin(tile_tops.crown_id)].set_index("crown_id")
            crowns = crowns.loc[tile_tops.crown_id].reset_index()          # same order as tile_tops
            tops = tile_tops.reset_index(drop=True)
            ref_boxes = crowns.copy()
            ref_boxes["geometry"] = crowns.envelope

            # score only what lies at least `edge` m inside the tile, and on LiDAR data
            # (where the CHM has no data there are no watershed trees to match against)
            def on_lidar(points):
                col = ((shapely.get_x(points) - left) / res).astype(int).clip(0, w.width - 1)
                row = ((top - shapely.get_y(points)) / res).astype(int).clip(0, w.height - 1)
                return ~no_lidar[row, col]
            inner = t.geometry.buffer(-edge)
            pred_centres = np.asarray(pred.geometry.centroid.values)
            pred_in = pred.geometry.centroid.within(inner).values & on_lidar(pred_centres)
            ref_in = tops.within(inner).values & on_lidar(np.asarray(tops.geometry.values))

            row = {"tile": t.tile, "split": t.split, "rumple": t.rumple,
                   "n_trees": int(ref_in.sum()), "n_pred": int(pred_in.sum())}
            for name in methods:
                if name == "treetop":
                    mt = treetop_match(pred, tops)
                elif name.startswith("both"):
                    mt = iou_match(pred, ref_boxes, int(name[4:]) / 100, tops)
                else:
                    mt = iou_match(pred, ref_boxes, int(name[3:]) / 100)
                p_hit = np.zeros(len(pred), bool); p_hit[mt[:, 0]] = True
                r_hit = np.zeros(len(tops), bool); r_hit[mt[:, 1]] = True
                row[f"{name}_tp"] = int((p_hit & pred_in).sum())
                row[f"{name}_found"] = int((r_hit & ref_in).sum())
                pred[f"tp_{name}"] = p_hit.astype(int)
                crowns[f"f_{name}"] = r_hit.astype(int)
                if name == shift_from:
                    # offset between matched prediction and watershed box centres (lean / systematic shift)
                    pc = np.asarray(pred.geometry.centroid.values)[mt[:, 0]]; rc = np.asarray(ref_boxes.geometry.centroid.values)[mt[:, 1]]
                    offsets.append(pd.DataFrame({"split": t.split, "tile": t.tile, "height": crowns.height.values[mt[:, 1]],
                                                 "dx": shapely.get_x(pc) - shapely.get_x(rc),
                                                 "dy": shapely.get_y(pc) - shapely.get_y(rc)}))
            tile_rows.append(row)
            pred_tables.append(pred)
            crown_tables.append(crowns.assign(tile=t.tile, split=t.split, scored=ref_in.astype(int)))
            print(f"{t.split:5s} {t.tile}: {row['n_trees']:,} watershed trees, {row['n_pred']:,} predictions")

    # 3b. Precision / recall / F1 per tile, and pooled per split (train / val / test kept apart)
    scores = pd.DataFrame(tile_rows).sort_values(["split", "rumple"])
    totals = scores.drop(columns=["tile", "rumple"]).groupby("split").sum().reset_index()
    totals["tile"], totals["rumple"] = "ALL " + totals.split, np.nan
    scores = pd.concat([scores, totals], ignore_index=True)
    for name in methods:
        prec = scores[f"{name}_tp"] / scores.n_pred
        rec = scores[f"{name}_found"] / scores.n_trees
        scores[f"{name}_P"], scores[f"{name}_R"] = prec, rec
        scores[f"{name}_F1"] = 2 * prec * rec / (prec + rec)
    scores.to_csv(scores_path, index=False)

    show = ["split", "tile", "rumple", "n_trees", "n_pred"] + [f"{n}_{s}" for n in methods for s in ("P", "R")]
    print("\nprecision (P) and recall (R) by matching criterion")
    print(scores[show].round(2).to_string(index=False))

    # recall by tree height, per split
    crowns_scored = pd.concat(crown_tables, ignore_index=True)
    cs = crowns_scored[crowns_scored.scored == 1].copy()
    cs["height_class"] = pd.cut(cs.height, [5, 10, 20, 30, 40, 100], right=False)
    by_height = cs.groupby(["split", "height_class"], observed=True)[[f"f_{n}" for n in methods]].mean()
    by_height.insert(0, "n_trees", cs.groupby(["split", "height_class"], observed=True).size())
    by_height.to_csv(os.path.join(out_dir, "recall_by_height.csv"))
    print("\nrecall by tree height")
    print(by_height.round(2).to_string())

    # recall by crown area (watershed crown, m2; one NAIP pixel is 0.36 m2)
    cs["area_class"] = pd.cut(cs.area_m2, [0, 10, 25, 50, 100, 200, 100000], right=False)
    by_area = cs.groupby(["split", "area_class"], observed=True)[[f"f_{n}" for n in methods]].mean()
    by_area.insert(0, "n_trees", cs.groupby(["split", "area_class"], observed=True).size())
    by_area.to_csv(os.path.join(out_dir, "recall_by_crown_area.csv"))
    print("\nrecall by crown area (m2)")
    print(by_area.round(2).to_string())

    # height and crown area together: tall trees have big crowns, so this shows which one matters
    both = cs.groupby(["split", "height_class", "area_class"], observed=True)[[f"f_{n}" for n in methods]].mean()
    both.insert(0, "n_trees", cs.groupby(["split", "height_class", "area_class"], observed=True).size())
    both.to_csv(os.path.join(out_dir, "recall_by_height_and_area.csv"))
    print(f"\nheight vs crown area of scored trees: Spearman {cs[['height', 'area_m2']].corr(method='spearman').iloc[0, 1]:.2f}")
    for split in [s for s in ["test", "val", "train"] if s in splits]:
        tab = cs[cs.split == split].pivot_table(index="height_class", columns="area_class", values="f_treetop",
                                                aggfunc="mean", observed=True)
        n = cs[cs.split == split].pivot_table(index="height_class", columns="area_class", values="f_treetop",
                                              aggfunc="size", observed=True)
        print(f"\ntreetop-in-box recall, {split} tiles, by height (rows) and crown area in m2 (columns); "
              f"cells with < 30 trees blank")
        print(tab.where(n >= 30).round(2).to_string())

    # systematic shift of matched predictions, by height (a shift that grows with height = lean)
    off = pd.concat(offsets, ignore_index=True)
    off["height_class"] = pd.cut(off.height, [5, 10, 20, 30, 40, 100], right=False)
    shift = off.groupby(["split", "tile", "height_class"], observed=True)[["dx", "dy"]].mean()
    shift["shift_m"] = np.hypot(shift.dx, shift.dy)
    shift["n"] = off.groupby(["split", "tile", "height_class"], observed=True).size()
    shift.to_csv(os.path.join(out_dir, "shift_by_height.csv"))
    print("\nmean shift of matched predictions from watershed boxes (m; dx = east, dy = north)")
    print(shift.round(2).to_string())

    # 3c. Shapefiles for QGIS
    pd.concat(pred_tables, ignore_index=True).pipe(gpd.GeoDataFrame, crs=crowns_all.crs).to_file(
        os.path.join(out_dir, "predictions.shp"))
    crowns_scored.drop(columns=["height_class", "area_class"], errors="ignore").pipe(gpd.GeoDataFrame, crs=crowns_all.crs).to_file(
        os.path.join(out_dir, "crowns_scored.shp"))
    print(f"\nwrote predictions.shp, crowns_scored.shp, tile_scores.csv, recall_by_height.csv, "
          f"recall_by_crown_area.csv, recall_by_height_and_area.csv, shift_by_height.csv -> {out_dir}")
    return scores


# ############################
# 4. Each fold: its trained model and the baseline on its test tiles
# ############################
# Baseline = DeepForest as downloaded (none of our training), with the same overlap filter as the trained
# models so the comparison is fair. Its patch size is chosen on each fold's validation tiles, then that
# fold's test tiles are scored once with it (choosing it on the test tiles would flatter the baseline).
tile_dir = os.path.join(out_dir, "naip_tiles")
baseline = main.deepforest(config_args={"nms_thresh": 0.3})
baseline.load_model("weecology/deepforest-tree")

choice = []
for k in range(n_folds):
    fold_tiles = all_tiles.assign(split=all_tiles[f"fold{k}"])          # train / val / test in this fold
    test_tiles = fold_tiles[fold_tiles.split == "test"]
    val_tiles = fold_tiles[fold_tiles.split == "val"]
    print(f"\n\n######## fold {k}: {len(test_tiles)} test tiles ########")

    # this fold's trained model (skipped until 03 has trained it)
    if os.path.exists(model_paths.format(k)):
        m = main.deepforest.load_from_checkpoint(model_paths.format(k))
        predict_and_score(m, model_paths.format(k), test_tiles, patch_size,
                          os.path.join(out_dir, "trained", f"fold{k}"), tile_dir)
        del m
    else:
        print(f"fold {k}: {model_paths.format(k)} doesn't exist yet (run 03), skipping its trained model")

    # baseline: patch size chosen on this fold's validation tiles...
    for ps in baseline_patch_sizes:
        s = predict_and_score(baseline, "pretrained (DeepForest as downloaded)", val_tiles, ps,
                              os.path.join(out_dir, "pretrained_patch_choice", f"fold{k}_patch{ps}"), tile_dir)
        choice.append({"fold": k, "patch_size": ps, choose_by: s.loc[s.tile == "ALL val", choose_by].item()})
    c = pd.DataFrame([x for x in choice if x["fold"] == k])
    best = int(c.loc[c[choose_by].fillna(0).idxmax(), "patch_size"])    # no predictions -> F1 counts as 0
    print(f"\nfold {k} baseline patch size: {best} px ({choose_by} on the validation tiles: "
          f"{dict(zip(c.patch_size, c[choose_by].round(3)))})")
    # ...then this fold's test tiles, once
    predict_and_score(baseline, "pretrained (DeepForest as downloaded)", test_tiles, best,
                      os.path.join(out_dir, "pretrained", f"fold{k}"), tile_dir)

choice = pd.DataFrame(choice)
choice["chosen"] = choice.groupby("fold")[choose_by].transform(lambda f: f.fillna(0) == f.fillna(0).max())
choice.to_csv(os.path.join(out_dir, "patch_size_choice.csv"), index=False)


# ############################
# 5. Cross-validated scores: every tile scored once, by the model that never saw it
# ############################
rows = []
for name in ["trained", "pretrained"]:
    for k in range(n_folds):
        f = os.path.join(out_dir, name, f"fold{k}", "tile_scores.csv")
        if os.path.exists(f):
            s = pd.read_csv(f)
            rows.append(s[~s.tile.str.startswith("ALL")].assign(model=name, fold=k))
tile_scores = pd.concat(rows, ignore_index=True)
# compare the models only on folds that both have
both = tile_scores.groupby("fold").model.nunique() == 2
tile_scores = tile_scores[tile_scores.fold.isin(both[both].index)]
print(f"\n\nfolds with both models scored: {sorted(int(f) for f in tile_scores.fold.unique())} of {list(range(n_folds))}")

counts = ["n_trees", "n_pred"] + [f"{n}_{c}" for n in methods for c in ("tp", "found")]
def add_prf(df):
    for name in methods:
        df[f"{name}_P"] = df[f"{name}_tp"] / df.n_pred
        df[f"{name}_R"] = df[f"{name}_found"] / df.n_trees
        df[f"{name}_F1"] = 2 * df[f"{name}_P"] * df[f"{name}_R"] / (df[f"{name}_P"] + df[f"{name}_R"])
    return df

# per fold, and all test tiles pooled (every crown counted once), plus the mean and SD across folds
by_fold = add_prf(tile_scores.groupby(["model", "fold"])[counts].sum()).reset_index()
pooled = add_prf(tile_scores.groupby("model")[counts].sum())
prf = [f"{n}_{s}" for n in methods for s in ("P", "R", "F1")]
summary = pd.concat({"pooled": pooled[prf].T, "fold mean": by_fold.groupby("model")[prf].mean().T,
                     "fold SD": by_fold.groupby("model")[prf].std().T}, axis=1)
summary = summary.swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)
by_fold.to_csv(os.path.join(out_dir, "cv_scores_by_fold.csv"), index=False)
summary.to_csv(os.path.join(out_dir, "cv_scores.csv"))
print(f"\ncross-validated precision (P), recall (R) and F1: all test tiles pooled, and mean / SD across folds")
print(summary.round(3).to_string())

# every tile, both models side by side, sorted by rumple
tile_scores = add_prf(tile_scores)
wide = tile_scores.pivot_table(index=["tile", "fold", "rumple", "n_trees"], columns="model",
                               values=[f"treetop_{s}" for s in ("P", "R", "F1")]).sort_index(level="rumple")
wide.columns = [f"{m}_{v}" for v, m in wide.columns]
wide = wide[[f"{m}_treetop_{v}" for m in ["pretrained", "trained"] for v in ("P", "R", "F1")]]
wide.to_csv(os.path.join(out_dir, "cv_tile_scores.csv"))
print("\neach tile (treetop matching), sorted by rumple")
print(wide.round(2).to_string())
print(f"\nwrote cv_scores.csv, cv_scores_by_fold.csv, cv_tile_scores.csv, patch_size_choice.csv -> {out_dir}")
