# DataSci_CNN_EEG
This repository holds the code for the final project for Data Science (S2024). This includes preprocessing of EEG data, converting the timeseries from the sensors to gramian angular & Markow transitional fields, and the training and testing 3D convolutional neural networks on the data.



The data is not publicly available, and therefore the full pipeline cannot be run without providing the data. However, the code is provided for reference and to show the steps taken to preprocess the data and train the models.

## Pipeline
All commands should be run from the root directory of the repository.

Create a virtual environment and install the required packages:
```
setup_env.sh
```

Preprocess the data:
```
python src/preprocess.py
```

Convert the timeseries to GAFs and MTFs:
```
python src/timeseries2gaf.py
```

Fit the individual CNNs to training participants:
```
python src/fit_individual_cnn.py
```

Fit the joint CNN on training participants:
```
python src/fit_joint_cnn.py
```

Finetune the joint CNN to test participants:
```
python src/finetune_joint.py
```

Evaluate the models:
```
python src/eval_models.py
```

Plot the results:
```
python src/plot_results.py
```


Alternatively, the pipeline can be run with the following command:
```
run_all.sh
```