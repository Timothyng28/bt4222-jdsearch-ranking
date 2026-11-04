# Impression-Aware Re-ranking of Search Results on JD.com

BT4222 Group 5. We re-rank the candidates shown for each JD.com search query so that the items a user clicks, carts or buys move to the top, using the [JDsearch dataset](https://github.com/rucliujn/JDsearch). The previous team's hybrid NCF model ([bt4222-team3-jdsearch](https://github.com/myathetchai/bt4222-team3-jdsearch)) is our reference baseline.

## Files

| File | What it does |
|---|---|
| `step1_data_preprocessed.ipynb` | Parses the raw JDsearch files, removes the label-ordering leak, turns history time gaps into days before the query, writes parquet files and records how long each interaction type is kept |
| `step2_feature_engineering.ipynb` | Builds one feature row per impression, with 35 model features in 8 hypothesis groups and the interim features kept for comparison, and freezes the model set in `model_features.json` |
| `step3_baselines.ipynb` | Fixes the evaluation protocol, scores random, popularity, the reference model (as published and with each fault fixed) and the interim features with LightGBM, which sets the bar |
| `step4_advanced_models.ipynb` | Trains LightGBM LambdaRank with a feature-group ablation and a short/long-term attention ranker, blends them and scores the test users once |
| `step5_diagnostics.ipynb` | Explains where the final model wins and fails by user and item, what drives it (SHAP), two leak checks and three business ideas tested on the test slates |
| `jd_eval.py` | The shared evaluation code (NDCG@5/10, HR@10, purchase recall@10, bootstrap intervals). Step 3 writes it to Drive and steps 4 and 5 import it from there, so this copy is for reading only |

Every notebook is saved with its outputs, and key results sit under a red RESULT banner.

## Data

The data is too large for GitHub, so it stays in Google Drive in the [`JDsearch_rawData`](https://drive.google.com/drive/folders/1Tozbsm8w9EbTO36YVxWPTulW4qQNB3oe) folder.

* The raw JDsearch files sit at the top of the folder and are only read.
* `processed/parquet/` holds the cleaned parquet files from step 1.
* `processed/artifacts/` holds the features, split, folds and feature dictionary from step 2, `jd_eval.py`, and the saved step-4 models and scores.
* `processed/results/` holds every results table (`step3_results.csv`, `step4_results.csv`, `step5_*.csv`), the per-slate test metrics and the figures.

## How to run

1. In Google Drive, open the `JDsearch_rawData` folder link above and choose **Add shortcut to Drive** → My Drive. Every notebook reads from `/content/drive/MyDrive/JDsearch_rawData`.
2. Open each notebook in Google Colab and run them in order, step 1 to step 5, with **Runtime → Run all**. Each one mounts Drive in its first cell and reads what the earlier steps wrote.
3. Colab's default packages cover everything (pandas, pyarrow, LightGBM, PyTorch, scikit-learn, matplotlib), so nothing needs installing.

| Notebook | Runtime type | Time for Run all |
|---|---|---|
| Step 1 | CPU, standard RAM | About 18 minutes |
| Step 2 | CPU, standard RAM | About 7 minutes. Set `CFG["use_names"] = False` if Colab runs out of memory |
| Step 3 | T4 GPU | About 3 h 40 min, most of it the three reference-model trainings. Set `CFG["quick"] = True` for a 5% smoke test |
| Step 4 | T4 GPU | About 22 minutes from the saved models. `CFG["rerun"] = True` retrains from scratch and takes longer |
| Step 5 | CPU, standard RAM | About 30 minutes, 24 of them for the SHAP contributions |

If Colab disconnects, step 4 resumes from its saved results on the next Run all. Step 3 saves each section's rows as it finishes, so rerun sections 0 to 2 and then only the section that did not finish.
