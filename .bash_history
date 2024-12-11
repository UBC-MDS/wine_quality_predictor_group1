quarto render scripts/notebooks/wine_predictor_analysis.qmd --to html
quarto render scripts/notebooks/wine_predictor_analysis.qmd --to html
ls
quarto render notebooks/wine_predictor_analysis.qmd --to html
python scripts/download_data.py --url=https://archive.ics.uci.edu/static/public/186/wine+quality.zip --write_to=data/raw/
python scripts/clean_data.py --input_path=data/raw/raw_data.csv --output_path=data/processed/cleaned_data.csv --log_path=results/tables/
python scripts/data_validation_script.py data/processed/cleaned_data.csv
python scripts/split_eda.py --clean_data_path=data/processed/cleaned_data.csv --train_test_path=data/processed/ --figures_path=results/figures/ --tables_path=results/tables/
python scripts/preprocess_model_selection.py --train_data_path=data/processed/ --scores_path=results/tables/ --preprocessor_path=results/models/ --model_path=results/models/
python scripts/tuning_script.py results/models/base_model.pickle results/models/best_model.pickle data/processed/X_train.csv data/processed/y_train.csv data/processed/X_test.csv data/processed/y_test.csv results/tables/best_params.csv
python scripts/model_evaluation.py --tuned_model_path=results/models/best_model.pickle --test_split_path=data/processed/ --test_accuracy_path=results/tables/ --figures_path=results/figures/
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
python scripts/preprocess_model_selection.py --train_data_path=data/processed/ --scores_path=results/tables/ --preprocessor_path=results/models/ --model_path=results/models/
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to html
quarto render report/wine_predictor_analysis.qmd --to pdf
quarto render report/wine_predictor_analysis_report.qmd --to html
quarto render report/wine_predictor_analysis_report.qmd --to pdf
