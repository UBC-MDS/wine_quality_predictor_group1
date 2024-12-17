## CHANGELOG for Wine Quality Predictor

Feedback was noted from this thread:
https://github.com/UBC-MDS/data-analysis-review-2024/issues/22


### Changes

- Added deepchecks to validation script:
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/915bc17489d5549cf7b96b0d7e0ee0744354cfcd


- Updated eda_script:
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/f1e4125960ae3d7f0af12753d4057313893b2004


- Saved raw `.zip` file in `data/raw` (based on feedback from @nvarabioff):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/a1f6ab1d401856a9ae41fb8dad6f30cc076eeb07


- Updated analysis process and added a link to the final report in `README.md` (based on feedback from @ClaireJ2100):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/df3a8925fa80638c798acdb543ae89e041ab5120


- Updated `README.md` to include updated instructions and `make` commands (based on feedback from @nvarabioff):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/a635c4741cda38f65dc9f3fe0331b7a115f749ce


- Updated `download_data.py` in scripts folder to add more try/except statements (based on feedback from @nvarabioff):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/fada2d50fa03ae49106107ca77000d56c9affe56


- Updated `clean_data.py` in scripts folder to add more try/except statements (based on feedback from @nvarabioff):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/46ac702b01af45c18191f9aa963d3df10e5d339d


- Removed old `conda-lock.yml` file that still existed on repository (based on feedback from @nvarabioff):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/7e885b0a7349fe5196d0eff0c478e0ad73464ea6


- Added tests for `cross_val_score.py`, `multiconfusion_matrix.py` and `summarize_conf_matrix.py` scripts (based on feedback from @nvarabioff):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/cc36531e4854e923ed87e70b21f2d9948b5aa1e8


- Creative Commons license (for project report) was added to `LICENSE` file (based on feedback from Milestone 1):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/73816dafd1cdce885bb93c759cec5cb2386f1301


- Creative Commons license (for project report) was added to `README.md` file (based on feedback from Milestone 1):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/0b28d693f8d5e7a918657e6993bb8ca7175c4156


- Added additional in-line references and addressed ordinal nature of multi-class targets in report. Changed email in `CODE_OF_CONDUCT.md` (based on feedback from Milestone 1):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/ae147978bf9f447e9630486378ca96e0cdbb37e4


- Added version numbers to `environment.yml` (based on feedback from Milestone 1):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/84f00f0717a5977c35a334c1c0c39232f6a58759


- Generated `wine_predictor_analysis_report.html` along with `.pdf` (based on feedback from Milestone 1):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/74ee76a0c0723678bebf9da5c706889eb1466be3


- Added high-level interpretation of analysis findings in `README.md` (based on feedback from Milestone 1):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/c28ea1209072ea3971551269e57307d0f13f7824

- Added reference for $109.5 billion valuation for the red wine industry. (based on feedback from @ClaireJ2100):
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/b689113378b3525ed66f9ca4e1bcca171c108925

- Split up the EDA charts and Train test split functions to two separate files: eda_charts.py and train_test_split.py both which are located in src folder.
https://github.com/UBC-MDS/wine_quality_predictor_group1/commit/cd1bccc6f275764a5e815a07867cc1f6a76ee2ba

- 