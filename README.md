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

Every notebook is saved with its outputs, and key results sit under a red RESULT banner.

## Data

The raw data is the JDsearch dataset, which is too large for GitHub. Download `JDsearch.tar.gz` from the link in the [JDsearch repository](https://github.com/rucliujn/JDsearch#obtain-the-data) and extract it with `tar -zxvf JDsearch.tar.gz`, which gives `user_behavior_data.txt` and `product_meta_data.txt`.

Everything else is produced by the notebooks. They write to a `processed/` folder next to the raw files.

* `processed/parquet/` holds the cleaned parquet files from step 1.
* `processed/artifacts/` holds the features, split, folds and feature dictionary from step 2, the shared evaluation code `jd_eval.py` from step 3, and the saved step-4 models and scores.
* `processed/results/` holds every results table (`step3_results.csv`, `step4_results.csv`, `step5_*.csv`), the per-slate test metrics and the figures.

## How to run

1. Create a folder called `JDsearch_rawData` at the top of My Drive and upload the two raw files into it. Every notebook reads from `/content/drive/MyDrive/JDsearch_rawData`. To use another location, set the `JD_BASE` environment variable to it.
2. Open each notebook in Google Colab and run them in order, step 1 to step 5, with **Runtime → Run all**. Each one mounts Drive in its first cell and reads what the earlier steps wrote.
3. Colab's default packages cover everything (pandas, pyarrow, LightGBM, PyTorch, scikit-learn, matplotlib), so nothing needs installing.

| Notebook | Runtime type | Time for Run all |
|---|---|---|
| Step 1 | CPU, standard RAM | About 18 minutes |
| Step 2 | CPU, standard RAM | About 7 minutes. Set `CFG["use_names"] = False` if Colab runs out of memory |
| Step 3 | T4 GPU | About 3 h 40 min, most of it the three reference-model trainings. Set `CFG["quick"] = True` for a 5% smoke test |
| Step 4 | T4 GPU | Longer on a first run, which trains every model. Our saved run took about 22 minutes because it loaded the ablation fits and attention models from an earlier run |
| Step 5 | CPU, standard RAM | About 30 minutes, 24 of them for the SHAP contributions |

If Colab disconnects, step 4 resumes from its saved results on the next Run all. Step 3 saves each section's rows as it finishes, so rerun sections 0 to 2 and then only the section that did not finish.
