.PHONY: clean

all: report/wine_predictor_analysis_report_files \
	report/wine_predictor_analysis_report.html \
	report/wine_predictor_analysis_report.pdf


# Downloads the data from path
data/raw/raw_data.csv: scripts/download_data.py
	python scripts/download_data.py \
		--url=http://archive.ics.uci.edu/static/public/186/wine+quality.zip \
		--write_to=data/raw/ 


# Cleans the data
cleaning_outputs = data/processed/cleaned_data.csv \
	results/tables/dataset_overview.csv \
	results/tables/missing_values.csv \
	results/tables/duplicates.csv 

$(cleaning_outputs): data/raw/raw_data.csv scripts/clean_data.py
	python scripts/clean_data.py \
		--input_path=data/raw/raw_data.csv \
		--output_path=data/processed/cleaned_data.csv \
		--log_path=results/tables/


# Splits and performs EDA
split_outputs = data/processed/X_train.csv \
	data/processed/y_train.csv \
	data/processed/X_test.csv \
	data/processed/y_test.csv

eda_outputs = results/figures/target_distribution_plot.png \
	results/figures/correlation_heatmap.png \
	results/figures/feature_distributions.png \
	results/figures/feature_pairplots.png	

$(split_outputs) $(eda_outputs): data/processed/cleaned_data.csv
	python scripts/split_eda.py \
		--clean_data_path=data/processed/cleaned_data.csv \
		--train_test_path=data/processed/ \
		--figures_path=results/figures/ \
		--tables_path=results/tables/


# Performs model selection and saves model
saved_models = results/models/base_model.pickle \
	results/models/preprocessor.pickle

 results/tables/initial_model_scores.csv $(saved_models): data/processed/X_train.csv
	python scripts/preprocess_model_selection.py \
		--train_data_path=data/processed/ \
		--scores_path=results/tables/ \
		--preprocessor_path=results/models/ \
		--model_path=results/models/


# Performs hyperparameter tuning on model
results/models/best_model.pickle results/tables/best_params.csv: results/models/base_model.pickle $(split_outputs)
	python scripts/tuning_script.py \
		results/models/base_model.pickle \
		results/models/best_model.pickle \
		data/processed/X_train.csv \
		data/processed/y_train.csv \
		data/processed/X_test.csv \
		data/processed/y_test.csv \
		results/tables/best_params.csv


# Perform model evaluation on test set
evaluation_outputs = results/figures/confusion_matrix_class_3.png \
	results/figures/confusion_matrix_class_4.png \
	results/figures/confusion_matrix_class_5.png \
	results/figures/confusion_matrix_class_6.png \
	results/figures/confusion_matrix_class_7.png \
	results/figures/confusion_matrix_class_8.png \
	results/tables/test_accuracy.csv

evaluation_inputs = data/processed/X_test.csv \
	data/processed/y_test.csv \
	results/models/best_model.pickle

$(evaluation_outputs): $(evaluation_inputs)
	python scripts/model_evaluation.py \
        --tuned_model_path=results/models/best_model.pickle \
        --test_split_path=data/processed/ \
        --test_accuracy_path=results/tables/ \
        --figures_path=results/figures/


# Renders the report
report_dependencies = $(evaluation_outputs) $(eda_outputs)
report_outputs = report/wine_predictor_analysis_report_files \
	report/wine_predictor_analysis_report.html \
	report/wine_predictor_analysis_report.pdf

$(report_outputs): $(report_dependencies) report/wine_predictor_analysis_report.qmd report/references.bib
	quarto render report/wine_predictor_analysis_report.qmd --to html
	quarto render report/wine_predictor_analysis_report.qmd --to pdf

clean:
	rm -rf data/processed \
		data/raw
	rm -rf results/figures \
		results/models \
		results/tables
	rm -rf report/wine_predictor_analysis_report.html \
		report/wine_predictor_analysis_report.pdf \
		report/wine_predictor_analysis_report_files