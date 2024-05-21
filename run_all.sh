
source env/bin/activate
python src/preprocess.py
python src/timeseries2gaf.py
python src/fit_individual_cnn.py
python src/fit_joint_cnn.py
python src/finetune_joint.py
python src/eval_models.py
python src/plot_results.py