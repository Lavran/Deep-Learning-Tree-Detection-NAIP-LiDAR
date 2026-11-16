import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config

# inputs: the 04_predict_tiles.py output folder of each model, holding one subfolder per cross-validation fold
# (the first model is the reference in section 6). Every path is set in config.py.
models = {
    "trained": os.path.join(config.PRED_DIR, "trained"),        # fine-tuned on the watershed crowns, one model per fold
    "pretrained": os.path.join(config.PRED_DIR, "pretrained"),  # DeepForest as downloaded (the baseline)
}
fig_names = {"trained": "Fine-tuned model", "pretrained": "Released model"}   # what each model is called in the figures
tiles_path = str(config.TILES_PATH)
tops_path = str(config.TOPS_PATH)
rumple_path = str(config.RUMPLE_PATH)
chm_path = str(config.CHM_PATH)

# outputs (all inside this folder): one subfolder per model, plus the model comparison
out_dir = str(config.ANALYSIS_DIR)
os.makedirs(out_dir, exist_ok=True)

# settings
# Every tile is analysed: with cross-validation, each tile's predictions come from the fold whose model never
# saw it (04), so all of them are honest test results.
match = "treetop"       # main matching criterion. Treetop-in-box is the LiDAR version of the field-stem recall of
                        # Weinstein et al. (2021): a stem counts as found when it falls inside a predicted box.
                        # Section 4 also repeats the models with every other criterion scored by 04
                        # (e.g. "iou50": box overlap with IoU >= 0.5; "both50": IoU >= 0.5 and treetop in box).
edge = 10               # m, same as 04: predictions are scored only this far inside the tile edge
n_bins = 5              # rumple bins, each with the same number of trees
area_bins = [0, 10, 25, 50, 100, 200, 100000]    # crown area classes (m2), same as 04
fig_bins = [0, 25, 50, 100, 100000]              # fewer, wider classes for the figures and the within-tile check
much_shorter = 0.75     # "next to a much taller tree": shorter than 75% of its tallest neighbour. This cut-off
                        # is my choice; Kaartinen et al. (2012) classed neighbourhoods by eye.
cv_folds = 10           # Hastings et al. (2020): 10-fold cross-validation
min_n = 30              # blank table cells / figure points with fewer trees than this
n_boot = 1000           # tile bootstrap resamples for the figures' 95% intervals
small_crown = 25        # m2; "small" crowns for the visual check sample
n_check = 50            # missed small crowns per group in the visual check sample
seed = 0
hit, correct = f"f_{match}", f"tp_{match}"     # found (watershed tree) and correct (prediction) columns from 04


# ############################
# 1. Watershed trees: size, rumple, neighbours (the same trees for every model)
# ############################
# the folds 04 has scored so far for each model
fold_dirs = {n: sorted(glob.glob(os.path.join(d, "fold*", "crowns_scored.shp"))) for n, d in models.items()}
for n in [n for n in models if not fold_dirs[n]]:
    print(f"skipping {n}: {models[n]} has no fold results yet (run 04 first)")
if not any(fold_dirs.values()):
    raise SystemExit(f"Nothing to analyse yet: no {os.path.join('fold*', 'crowns_scored.shp')} in {' or '.join(models.values())}. "
                     f"Run 04_predict_tiles.py first.")
models = {n: d for n, d in models.items() if fold_dirs[n]}
names = list(models)
for n in names:
    print(f"{n}: {len(fold_dirs[n])} folds ({', '.join(os.path.basename(os.path.dirname(f)) for f in fold_dirs[n])})")

def read_folds(n, file, **kwargs):
    """One model's crowns_scored.shp or predictions.shp, all folds together."""
    return pd.concat([gpd.read_file(os.path.join(os.path.dirname(f), file), **kwargs) for f in fold_dirs[n]],
                     ignore_index=True)

crowns = read_folds(names[0], "crowns_scored.shp")
tiles = gpd.read_file(tiles_path)
tiles = tiles[tiles.tile.isin(crowns.tile)].reset_index(drop=True)
criteria = [c[2:] for c in crowns.columns if c.startswith("f_")]     # every matching criterion 04 scored
crowns = crowns.drop(columns=[f"f_{r}" for r in criteria])     # found / missed is added for each model below

# Canopy position, after Kaartinen et al. (2012)'s neighbourhood classes ("tallest tree in the
# neighbourhood or isolated", "in a group of similar trees", "next to a bigger tree"). Neighbours are
# the watershed crowns that share a boundary with this one; every crown in the tile counts, scored or not.
i, j = crowns.sindex.query(crowns.geometry, predicate="intersects")
nb = pd.DataFrame({"i": i, "h": crowns.height.values[j]})[i != j]
crowns["tallest_nb"] = nb.groupby("i").h.max().reindex(crowns.index).values
crowns["rel_height"] = crowns.height / crowns.tallest_nb         # below 1: a neighbour is taller
crowns["position"] = np.select(
    [crowns.rel_height.isna() | (crowns.rel_height >= 1), crowns.rel_height >= much_shorter],
    ["tallest or isolated", "similar to neighbours"], default="next to a much taller tree")
positions = ["tallest or isolated", "similar to neighbours", "next to a much taller tree"]

# rumple of the 30 m cell each treetop falls in. (A tree's own crown is part of its cell's canopy
# surface, so a big crown raises its own cell's rumple a little.)
tops = gpd.read_file(tops_path).set_index("crown_id").loc[crowns.crown_id]
x, y = tops.geometry.x.values, tops.geometry.y.values
with rasterio.open(rumple_path) as src:
    crowns["rumple"] = [v[0] for v in src.sample(zip(x, y))]
    rows, cols = rasterio.transform.rowcol(src.transform, x, y)
crowns["cell"] = np.asarray(rows) * 100000 + np.asarray(cols)   # one id per 30 m cell

# the trees 04 scored (treetop at least `edge` m inside the tile, on LiDAR data), with a rumple value
trees = crowns[(crowns.scored == 1) & (crowns.rumple > 0)].copy()
trees["log_area"] = np.log(trees.area_m2)

# which trees each model found (04 scores the same trees for every model)
found = {}
for n in names:
    f = read_folds(n, "crowns_scored.shp", ignore_geometry=True)
    found[n] = f[f.scored == 1].set_index("crown_id")[[f"f_{r}" for r in criteria]]
in_all = trees.crown_id.isin(set.intersection(*[set(f.index) for f in found.values()]))
if not in_all.all():
    print(f"note: {(~in_all).sum():,} trees weren't scored for every model; using the {in_all.sum():,} that were")
trees = trees[in_all].copy()

# rumple bins with equal numbers of trees (the same edges for every model and for the predictions)
edges = np.quantile(trees.rumple, np.linspace(0, 1, n_bins + 1))
bin_labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]
cut_edges = np.r_[-np.inf, edges[1:-1], np.inf]
trees["rumple_bin"] = pd.cut(trees.rumple, cut_edges, labels=bin_labels)
trees["area_class"] = pd.cut(trees.area_m2, area_bins, right=False)
trees["fig_class"] = pd.cut(trees.area_m2, fig_bins, right=False)
area_order = [str(c) for c in trees.area_class.cat.categories]
trees["half"] = np.where(trees.rumple > trees.groupby("tile").rumple.transform("median"), "rougher", "smoother")

# standardized predictors (mean 0, SD 1), as in Hastings et al. (2020); one scale for every model
for v in ["height", "log_area", "rumple"]:
    trees[f"z_{v}"] = (trees[v] - trees[v].mean()) / trees[v].std()
rumple_mean, rumple_sd = trees.rumple.mean(), trees.rumple.std()
# canopy position as 0/1 columns, compared with "tallest or isolated"
trees["pos_similar"] = (trees.position == "similar to neighbours").astype(int)
trees["pos_shorter"] = (trees.position == "next to a much taller tree").astype(int)
print(f"{len(trees):,} watershed trees in {trees.tile.nunique()} test tiles "
      f"({trees.cell.nunique():,} 30 m rumple cells); models: {', '.join(names)}")


# ############################
# 2. Predictions of each model: correct or not, box size, rumple
# ############################
# 04 doesn't save which predictions it scored, so redo its rule: box centre at least `edge` m inside
# the tile and on LiDAR data
inner = tiles.set_index("tile").geometry.buffer(-edge)
preds = {}
for n in names:
    pred = read_folds(n, "predictions.shp")
    pred = pred[pred.tile.isin(tiles.tile)].reset_index(drop=True)
    centres = pred.geometry.centroid
    inside = centres.within(gpd.GeoSeries(inner.loc[pred.tile].values, index=pred.index, crs=pred.crs))
    with rasterio.open(chm_path) as src:
        chm = np.array([v[0] for v in src.sample(zip(centres.x, centres.y))])
    with rasterio.open(rumple_path) as src:
        pred["rumple"] = [v[0] for v in src.sample(zip(centres.x, centres.y))]
        rows, cols = rasterio.transform.rowcol(src.transform, centres.x.values, centres.y.values)
    pred["cell"] = np.asarray(rows) * 100000 + np.asarray(cols)
    pred["box_area"] = pred.geometry.area
    preds[n] = pred[inside.values & np.isfinite(chm) & (chm > -100) & (pred.rumple > 0)].copy()

# box size standardized on all models' predictions together, rumple on the trees' scale (as above)
log_box = np.log(pd.concat([p.box_area for p in preds.values()]))
for p in preds.values():
    p["log_box_area"] = np.log(p.box_area)
    p["z_log_box_area"] = (p.log_box_area - log_box.mean()) / log_box.std()
    p["z_rumple"] = (p.rumple - rumple_mean) / rumple_sd
    p["rumple_bin"] = pd.cut(p.rumple, cut_edges, labels=bin_labels)
    p["area_class"] = pd.cut(p.box_area, area_bins, right=False)
    p["fig_class"] = pd.cut(p.box_area, fig_bins, right=False)


# ############################
# 3. Tools for the tables, models and figures
# ############################
def by_rumple(df, group, value, order):
    """Rate (mean of a 0/1 column) and count for each group x rumple bin, plus an 'all' row."""
    d = pd.concat([df[[group, "rumple_bin", value]].assign(**{group: df[group].astype(str)}),
                   df[["rumple_bin", value]].assign(**{group: "all"})])
    d[group] = pd.Categorical(d[group], categories=order + ["all"])
    return d.groupby([group, "rumple_bin"], observed=True)[value].agg(n="size", rate="mean")

def show(t, title):
    rate = t.rate.unstack("rumple_bin").where(t.n.unstack("rumple_bin") >= min_n)
    print(f"\n{title} (columns: rumple bins; cells with < {min_n} trees blank)")
    print(rate.round(2).to_string())

# Tree-level logistic regression, as in Hastings et al. (2020):
#  - predictors standardized (section 1), so each odds ratio is "per 1 SD"
#  - multicollinearity checked with VIF (all should be < 10)
#  - backward elimination from the global model, keeping the lowest AIC
#  - 10-fold cross-validated accuracy
# Differences from Hastings, and why:
#  - crown area is logged: detection rises steeply for small crowns and levels off for big ones
#    (recall_by_crown_area.csv from 04), which a straight line on raw area can't follow
#  - Hastings used plot as a random effect (lme4 in R). Here tile is a fixed effect (each tile gets its
#    own baseline), which does the same job with 8 tiles and thousands of trees per tile. Trees in the
#    same 30 m cell aren't independent, so standard errors are clustered by cell.
#  - no DBH (no field data); height and crown area come from the LiDAR
# Reading an odds ratio: 1 = no effect; below 1 = less likely to be found as that predictor goes up 1 SD,
# with the other predictors held fixed.
def fit(formula, df):
    return smf.logit(formula, df).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df.cell.values})

def backward_aic(y, terms, df, label):
    """Start from the global model, drop the term whose removal lowers AIC most, and repeat until no
    removal lowers it. Tile always stays in (it stands in for Hastings et al.'s random effect)."""
    terms = list(terms)
    formula = lambda ts: f"{y} ~ " + " + ".join(ts + ["C(tile)"])
    best = fit(formula(terms), df)
    steps = [{"model": label, "step": "global", "terms": " + ".join(terms), "aic": best.aic}]
    while terms:
        trials = {t: fit(formula([s for s in terms if s != t]), df) for t in terms}
        drop = min(trials, key=lambda t: trials[t].aic)
        if trials[drop].aic >= best.aic:
            break
        terms.remove(drop)
        best = trials[drop]
        steps.append({"model": label, "step": f"drop {drop}", "terms": " + ".join(terms), "aic": best.aic})
    return best, terms, steps

def cv_scores(formula, df, y):
    """10-fold cross-validation: fit on 9/10 of the rows, predict the other 1/10, ten times. Accuracy uses
    a 0.5 cut-off; 'no_skill' is the accuracy of always guessing the commoner outcome; AUC is the chance
    that a found tree gets a higher predicted probability than a missed one (0.5 = no better than chance)."""
    fold = np.random.default_rng(seed).permutation(len(df)) % cv_folds
    p = np.zeros(len(df))
    for k in range(cv_folds):
        m = smf.logit(formula, df[fold != k]).fit(disp=0)
        p[fold == k] = m.predict(df[fold == k]).values
    obs = df[y].values.astype(float)           # float: with ~200,000 trees the counts below overflow integers
    n1, n0 = obs.sum(), len(obs) - obs.sum()
    auc = (stats.rankdata(p)[obs == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"cv_accuracy": ((p >= 0.5) == obs).mean(), "no_skill": max(obs.mean(), 1 - obs.mean()), "cv_auc": auc}

def odds_ratios(m, label):
    ci = np.exp(m.conf_int())
    t = pd.DataFrame({"model": label, "odds_ratio": np.exp(m.params), "or_low_95": ci[0], "or_high_95": ci[1],
                      "p": m.pvalues})
    t = t[~(t.index.str.startswith("C(tile)") | (t.index == "Intercept"))]     # just the predictors
    return t.rename_axis("term").reset_index()

def boot_rates(df, group, value):
    """Rate per group x rumple bin with a 95% interval from resampling whole tiles (trees in a tile aren't
    independent, so resampling single trees would make the intervals too narrow)."""
    c = df.pivot_table(index=[group, "rumple_bin"], columns="tile", values=value,
                       aggfunc=["sum", "size"], observed=True, fill_value=0)
    hits, n = c["sum"].values, c["size"].values
    pick = np.random.default_rng(seed).integers(0, n.shape[1], (n_boot, n.shape[1]))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        boot = hits[:, pick].sum(-1) / n[:, pick].sum(-1)
        low, high = np.nanpercentile(boot, [2.5, 97.5], axis=1)
    return pd.DataFrame({"n": n.sum(1), "rate": hits.sum(1) / n.sum(1), "low": low, "high": high}, index=c.index)

fig_labels = ["< 25 m²", "25–50 m²", "50–100 m²", "≥ 100 m²"]
colours = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]     # one blue, light = small -> dark = big
ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e6e6e6"
xs = np.arange(n_bins)

def draw_panel(ax, df, value, title):
    """One panel: rate vs rumple bin, one line per size class plus all sizes together (dashed)."""
    r = boot_rates(df, "fig_class", value)
    present = set(map(str, r.index.get_level_values(0)))      # size classes this model has at all
    ends = []       # [x, y, text] for a label at the right end of each line
    for cls, lab, col in zip(df.fig_class.cat.categories, fig_labels, colours):
        if str(cls) in present:
            s = r.xs(cls, level=0).reindex(bin_labels)
        else:
            s = pd.DataFrame(np.nan, index=bin_labels, columns=r.columns)
        s = s.where(s.n >= min_n)
        ax.errorbar(xs, s.rate, yerr=[s.rate - s.low, s.high - s.rate], color=col, lw=2, marker="o", ms=7,
                    elinewidth=1, capsize=0, label=lab)
        if s.rate.notna().any():
            last = bin_labels.index(s.rate.last_valid_index())
            ends.append([xs[last], s.rate.iloc[last], lab])
    a = boot_rates(df.assign(everything="all"), "everything", value).xs("all", level=0).reindex(bin_labels)
    ax.errorbar(xs, a.rate, yerr=[a.rate - a.low, a.high - a.rate], color=ink, lw=2, ls="--", marker="o", ms=7,
                elinewidth=1, capsize=0, label="all sizes together")
    ends.append([xs[-1], a.rate.iloc[-1], "all sizes"])
    # nudge labels that end at the same x apart so they don't overlap
    for xe in set(e[0] for e in ends):
        same_x = sorted([e for e in ends if e[0] == xe], key=lambda e: e[1])
        for k in range(1, len(same_x)):
            same_x[k][1] = max(same_x[k][1], same_x[k - 1][1] + 0.05)
    for xe, ye, lab in ends:
        ax.annotate(lab, (xe, ye), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=ink)
    ax.set_title(title, loc="left", fontsize=11, color=ink)
    ax.set_xticks(xs, bin_labels, fontsize=8, color=muted)
    ax.set_xlim(-0.3, n_bins - 0.3 + 0.6)
    ax.set_xlabel("rumple of the 30 m cell (5 bins, equal numbers of trees)", fontsize=9, color=muted)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="y", labelsize=8, labelcolor=muted)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

def finish_figure(fig, axes, note, path):
    for ax in axes[:, 0]:
        ax.set_ylabel(f"share ({match} matching)", fontsize=9, color=muted)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"{note} Bars: 95% interval from resampling whole tiles; points with < {min_n} trees not shown.",
                 fontsize=9, color=muted, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07 / axes.shape[0], 1, 1 - 0.05 / axes.shape[0]))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ############################
# 4. Each model: recall and precision vs rumple, by size; Hastings-style models; matching criteria
# ############################
for name in names:
    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    tm = trees.join(found[name], on="crown_id")          # the watershed trees, found or missed by this model
    pm = preds[name]
    print(f"\n\n################ {name} ({models[name]}) -> {model_dir} ################")
    print(f"{tm[hit].mean():.0%} of {len(tm):,} watershed trees found; {pm[correct].mean():.0%} of {len(pm):,} "
          f"predictions correct ({match} matching)")

    # 4a. Overall accuracy by rumple (all trees together; this is what rumple looks like before
    # accounting for crown size), by rumple bin and by tile
    acc = pd.DataFrame({"n_trees": tm.groupby("rumple_bin", observed=True).size(),
                        "recall": tm.groupby("rumple_bin", observed=True)[hit].mean(),
                        "n_predictions": pm.groupby("rumple_bin", observed=True).size(),
                        "precision": pm.groupby("rumple_bin", observed=True)[correct].mean()})
    acc.index = acc.index.astype(str)
    acc.loc["all"] =[len(tm), tm[hit].mean(), len(pm), pm[correct].mean()]
    acc["F1"] = 2 * acc.precision * acc.recall / (acc.precision + acc.recall)
    acc.to_csv(os.path.join(model_dir, "accuracy_by_rumple.csv"))
    print("\noverall accuracy by rumple bin (all crown sizes together)")
    print(acc.round(3).to_string())
    acc_tile = pd.DataFrame({"n_trees": tm.groupby("tile").size(), "recall": tm.groupby("tile")[hit].mean(),
                             "n_predictions": pm.groupby("tile").size(), "precision": pm.groupby("tile")[correct].mean()})
    acc_tile["F1"] = 2 * acc_tile.precision * acc_tile.recall / (acc_tile.precision + acc_tile.recall)
    acc_tile = tiles.set_index("tile")[["fold", "rumple"]].join(acc_tile).sort_values("rumple")
    acc_tile.to_csv(os.path.join(model_dir, "accuracy_by_tile.csv"))
    print("\noverall accuracy by tile, sorted by the tile's mean rumple")
    print(acc_tile.round(3).to_string())

    # 4b. Recall and precision by rumple, within size and position classes. Accuracy reported by stratum
    # (Yin & Wang 2016): without the size classes, rumple looks helpful only because rough canopy holds
    # more big trees, and big trees are easy to find.
    recall_area = by_rumple(tm, "area_class", hit, area_order)
    recall_position = by_rumple(tm, "position", hit, positions)
    precision_area = by_rumple(pm, "area_class", correct, area_order)
    recall_area.to_csv(os.path.join(model_dir, "recall_by_area_and_rumple.csv"))
    recall_position.to_csv(os.path.join(model_dir, "recall_by_position_and_rumple.csv"))
    precision_area.to_csv(os.path.join(model_dir, "precision_by_box_area_and_rumple.csv"))

    show(recall_area, "recall by crown area (m2)")
    print("\nshare of each rumple bin's trees in each crown area class (%)")
    print((pd.crosstab(tm.area_class, tm.rumple_bin, normalize="columns") * 100).round(0).to_string())
    show(recall_position, "recall by canopy position")
    show(precision_area, "precision by predicted box area (m2)")

    # Same comparison inside each tile: recall in the rougher half of the tile's trees minus the smoother
    # half, so differences between tiles (image date, sun angle, stand type) can't drive it
    w = tm.pivot_table(index="fig_class", columns=["tile", "half"], values=hit, aggfunc="mean", observed=True)
    within = w.xs("rougher", axis=1, level="half") - w.xs("smoother", axis=1, level="half")
    n_lower = (within < 0).sum(axis=1)
    n_tiles = within.notna().sum(axis=1)
    within_summary = pd.DataFrame({"tiles_lower_when_rougher": n_lower.astype(str) + " of " + n_tiles.astype(str),
                                   "median_difference": within.median(axis=1)})
    within.join(within_summary).to_csv(os.path.join(model_dir, "within_tile_check.csv"))
    print("\nwithin each tile: recall in the rougher half minus the smoother half, by crown area (m2) "
          "(every tile in within_tile_check.csv)")
    print(within_summary.round(3).to_string())

    # 4c. The aggregated view: Hastings et al. (2020) first regressed plot-level accuracy on plot-level
    # metrics such as rumple. Same here for tiles and 30 m cells. Averages over many
    # trees mix crown sizes, which is why this view can mislead. (Cells in the same tile aren't
    # independent, so the cell p-values are optimistic.)
    tile_acc = tm.groupby("tile").agg(recall=(hit, "mean"), n=(hit, "size")).join(tiles.set_index("tile").rumple)
    cell_acc = tm.groupby("cell").agg(recall=(hit, "mean"), n=(hit, "size"), rumple=("rumple", "first"),
                                      log_area=("log_area", "mean"))
    cell_acc = cell_acc[cell_acc.n >= 5]
    agg_rows = []
    for unit, df, formula in [("tiles", tile_acc, "recall ~ rumple"),
                              ("30 m cells (>= 5 trees)", cell_acc, "recall ~ rumple"),
                              ("30 m cells (>= 5 trees)", cell_acc, "recall ~ rumple + log_area")]:
        m = smf.ols(formula, df).fit()
        agg_rows.append({"unit": unit, "n": len(df), "model": formula, "rumple_slope": m.params["rumple"],
                         "p": m.pvalues["rumple"], "r2": m.rsquared})
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(os.path.join(model_dir, "aggregated_regressions.csv"), index=False)
    print("\naggregated view: linear regressions of recall on rumple (log_area = mean log crown area of the cell)")
    print(agg.round(4).to_string(index=False))

    # 4d. Tree-level logistic regression (Hastings et al. 2020; see section 3)
    X = sm.add_constant(tm[["z_height", "z_log_area", "z_rumple"]])
    vif = pd.Series([variance_inflation_factor(X.values, k) for k in range(1, X.shape[1])], index=X.columns[1:])
    print(f"\nVIF (Hastings et al. kept all < 10): {vif.round(2).to_dict()}")

    # recall models (y = found)
    recall_models = {
        # rumple alone: the aggregated view's question, asked tree by tree
        "R1: rumple only": f"{hit} ~ z_rumple + C(tile)",
        # Hastings et al.'s global model: tree size + rumple (backward elimination below)
        "R2: global": f"{hit} ~ z_height + z_log_area + z_rumple + C(tile)",
        # beyond Hastings: does rumple matter more for small crowns? and does canopy position explain it?
        "R3: + rumple x crown area": f"{hit} ~ z_height + z_log_area * z_rumple + C(tile)",
        "R4: + canopy position": f"{hit} ~ z_height + z_log_area * z_rumple + pos_similar + pos_shorter + C(tile)",
    }
    selected, kept, steps = backward_aic(hit, ["z_height", "z_log_area", "z_rumple"], tm, f"{name} recall")
    recall_models["R2s: selected by AIC"] = f"{hit} ~ " + " + ".join(kept + ["C(tile)"])

    # precision models (y = prediction correct); predicted box area stands in for crown size
    precision_models = {
        "P1: rumple only": f"{correct} ~ z_rumple + C(tile)",
        "P2: global": f"{correct} ~ z_log_box_area + z_rumple + C(tile)",
        "P3: + rumple x box area": f"{correct} ~ z_log_box_area * z_rumple + C(tile)",
    }
    p_selected, p_kept, p_steps = backward_aic(correct, ["z_log_box_area", "z_rumple"], pm, f"{name} precision")
    precision_models["P2s: selected by AIC"] = f"{correct} ~ " + " + ".join(p_kept + ["C(tile)"])

    ors, summary = [], []
    for model_set, df, yv in [(recall_models, tm, hit), (precision_models, pm, correct)]:
        for label, formula in model_set.items():
            m = fit(formula, df)
            ors.append(odds_ratios(m, label))
            summary.append({"model": label, "formula": formula, "n": int(m.nobs), "aic": m.aic,
                            **cv_scores(formula, df, yv)})
    ors = pd.concat(ors, ignore_index=True)
    summary = pd.DataFrame(summary)
    summary["delta_aic"] = summary.aic - summary.groupby(summary.model.str[0]).aic.transform("min")
    ors.to_csv(os.path.join(model_dir, "model_odds_ratios.csv"), index=False)
    summary.to_csv(os.path.join(model_dir, "model_summary.csv"), index=False)
    pd.DataFrame(steps + p_steps).to_csv(os.path.join(model_dir, "model_selection_steps.csv"), index=False)

    print("\nbackward elimination by AIC (Hastings et al. 2020)")
    print(pd.DataFrame(steps + p_steps).round(1).to_string(index=False))
    print("\nodds ratios per 1 SD (95% CI; standard errors clustered by 30 m cell)")
    print(ors.round(3).to_string(index=False))
    print("\nmodel comparison (lowest AIC = best within recall / precision; delta_aic > 2 is a real difference)")
    print(summary.drop(columns="formula").round(3).to_string(index=False))

    # 4e. Does the result depend on the matching criterion? Yin & Wang (2016) name the matching method as one
    # of the main choices behind any accuracy number, so repeat the rumple odds ratios with every criterion
    rule_rows = []
    for r in criteria:
        alone = fit(f"f_{r} ~ z_rumple + C(tile)", tm)
        sized = fit(f"f_{r} ~ z_height + z_log_area + z_rumple + C(tile)", tm)
        inter = fit(f"f_{r} ~ z_height + z_log_area * z_rumple + C(tile)", tm)
        rule_rows.append({"criterion": r, "recall": tm[f"f_{r}"].mean(),
                          "rumple_OR_alone": np.exp(alone.params["z_rumple"]), "p_alone": alone.pvalues["z_rumple"],
                          "rumple_OR_with_size": np.exp(sized.params["z_rumple"]), "p_with_size": sized.pvalues["z_rumple"],
                          "rumple_x_area_OR": np.exp(inter.params["z_log_area:z_rumple"]),
                          "p_interaction": inter.pvalues["z_log_area:z_rumple"]})
    rule_check = pd.DataFrame(rule_rows)
    rule_check.to_csv(os.path.join(model_dir, "matching_criteria_check.csv"), index=False)
    print("\nrumple odds ratio per 1 SD under each matching criterion (recall)")
    print(rule_check.round(3).to_string(index=False))

    # 4f. Figure, and the tree-level table (e.g. to refit the models in R with lme4 as Hastings et al. did)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, squeeze=False)
    draw_panel(axes[0, 0], tm, hit, "Recall, by watershed crown area")
    draw_panel(axes[0, 1], pm, correct, "Precision, by predicted box area")
    finish_figure(fig, axes, f"{fig_names.get(name, name)}: detection vs canopy rumple, {trees.tile.nunique()} test tiles "
                             f"({len(tm):,} trees, {len(pm):,} predictions).",
                  os.path.join(model_dir, "rumple_recall_precision.png"))
    tm.drop(columns="geometry").to_csv(os.path.join(model_dir, "trees_table.csv"), index=False)
    print(f"\nwrote accuracy_by_rumple.csv, accuracy_by_tile.csv, recall_by_area_and_rumple.csv, recall_by_position_and_rumple.csv, "
          f"precision_by_box_area_and_rumple.csv, within_tile_check.csv, aggregated_regressions.csv, "
          f"model_odds_ratios.csv, model_summary.csv, model_selection_steps.csv, matching_criteria_check.csv, "
          f"rumple_recall_precision.png, trees_table.csv -> {model_dir}")


# ############################
# 5. Sample for a visual check in QGIS (the first model's misses)
# ############################
# Are the missed small crowns real trees, or watershed errors (e.g. a branch of a big crown split off as
# its own crown)? 50 missed small crowns from the smoothest rumple bin and 50 from the roughest, shuffled.
# Open with the NAIP tiles (deepforest_predictions_cv\naip_tiles) and the CHM, hide the `group` column
# so you don't know which bin each came from, and fill in `verdict`: tree / not tree / unsure.
tm = trees.join(found[names[0]], on="crown_id")
missed = tm[(tm.area_m2 < small_crown) & (tm[hit] == 0)]
parts = []
for lab, b in [("smoothest", bin_labels[0]), ("roughest", bin_labels[-1])]:
    g = missed[missed.rumple_bin == b]
    parts.append(g.sample(min(n_check, len(g)), random_state=seed).assign(group=lab))
check = pd.concat(parts).sample(frac=1, random_state=seed)
check["check_id"] = np.arange(1, len(check) + 1)
check["verdict"] = "-"
check[["check_id", "crown_id", "tile", "group", "rumple", "area_m2", "height", "verdict", "geometry"]].to_file(
    os.path.join(out_dir, names[0], "check_missed_small_crowns.shp"))
print(f"\nwrote check_missed_small_crowns.shp ({names[0]} model's misses) -> {os.path.join(out_dir, names[0])}")


# ############################
# 6. Does fine-tuning change how rumple affects detection? (first model vs second)
# ############################
# Both models are scored on the same watershed trees, so stack their results and fit one model with
# separate size effects and tile baselines per model (so differences in size or tile can't be mistaken for
# a rumple difference). The rumple x model term then tests whether rumple's effect differs between them:
# a ratio below 1 means rumple hurts the second model more than the first. Standard errors are clustered
# by 30 m cell, which also keeps each tree's two results together.
if len(names) >= 2:
    a, b = names[0], names[1]

    # side-by-side figure: one row per model
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4.6 * len(names)), sharey=True, squeeze=False)
    for k, n in enumerate(names):
        draw_panel(axes[k, 0], trees.join(found[n], on="crown_id"), hit, f"{fig_names.get(n, n)}: recall, by watershed crown area")
        draw_panel(axes[k, 1], preds[n], correct, f"{fig_names.get(n, n)}: precision, by predicted box area")
    finish_figure(fig, axes, f"Detection vs canopy rumple by model, {trees.tile.nunique()} test tiles ({len(trees):,} trees).",
                  os.path.join(out_dir, "models_rumple_recall_precision.png"))

    # overall accuracy by rumple bin, both models side by side
    acc_both = pd.concat({n: pd.read_csv(os.path.join(out_dir, n, "accuracy_by_rumple.csv"), index_col=0)
                          [["recall", "precision", "F1"]] for n in names}, axis=1)
    acc_both.to_csv(os.path.join(out_dir, "models_accuracy_by_rumple.csv"))
    print(f"\n\n################ {a} vs {b} ################")
    print("\noverall accuracy by rumple bin, both models")
    print(acc_both.round(3).to_string())

    # recall by crown area x rumple for both models, side by side
    side = pd.concat({n: by_rumple(trees.join(found[n], on="crown_id"), "area_class", hit, area_order).rate
                      for n in names}, axis=1)
    side[f"{b} - {a}"] = side[b] - side[a]
    side.insert(0, "n_trees", by_rumple(trees.join(found[a], on="crown_id"), "area_class", hit, area_order).n)
    side.to_csv(os.path.join(out_dir, "models_recall_by_area_and_rumple.csv"))
    print(f"\nrecall by crown area (m2) and rumple, both models")
    print(side.round(2).to_string())

    # stacked models: recall (trees) and precision (predictions)
    tl = pd.concat([trees.join(found[n], on="crown_id").assign(model=n) for n in (a, b)], ignore_index=True)
    pl = pd.concat([preds[n].assign(model=n) for n in (a, b)], ignore_index=True)
    tl["m2"], pl["m2"] = (tl.model == b).astype(int), (pl.model == b).astype(int)
    comp_rows = []
    for what, df, yv, size_terms in [("recall", tl, hit, "z_height + z_log_area"),
                                     ("precision", pl, correct, "z_log_box_area")]:
        m = fit(f"{yv} ~ ({size_terms} + z_rumple) * m2 + C(tile) * m2", df)
        V = m.cov_params()
        b1, b12 = m.params["z_rumple"], m.params["z_rumple:m2"]
        se1, se12 = m.bse["z_rumple"], m.bse["z_rumple:m2"]
        se2 = np.sqrt(V.loc["z_rumple", "z_rumple"] + V.loc["z_rumple:m2", "z_rumple:m2"]
                      + 2 * V.loc["z_rumple", "z_rumple:m2"])
        comp_rows.append({
            "measure": what, f"{a}_rate": df.loc[df.m2 == 0, yv].mean(), f"{b}_rate": df.loc[df.m2 == 1, yv].mean(),
            f"rumple_OR_{a}": np.exp(b1), f"{a}_low_95": np.exp(b1 - 1.96 * se1), f"{a}_high_95": np.exp(b1 + 1.96 * se1),
            f"rumple_OR_{b}": np.exp(b1 + b12), f"{b}_low_95": np.exp(b1 + b12 - 1.96 * se2),
            f"{b}_high_95": np.exp(b1 + b12 + 1.96 * se2),
            f"ratio_{b}_vs_{a}": np.exp(b12), "ratio_low_95": np.exp(b12 - 1.96 * se12),
            "ratio_high_95": np.exp(b12 + 1.96 * se12), "p_difference": m.pvalues["z_rumple:m2"]})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(out_dir, "models_rumple_effect.csv"), index=False)
    print(f"\nrumple odds ratio per 1 SD with crown size held fixed, each model, and the ratio between them")
    print(comp.round(3).T.to_string(header=False))
    print(f"\nwrote models_accuracy_by_rumple.csv, models_rumple_recall_precision.png, models_recall_by_area_and_rumple.csv, "
          f"models_rumple_effect.csv -> {out_dir}")
