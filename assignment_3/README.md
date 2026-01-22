# Assignment 3

## Attribute Inference from ML Models

by _Vishwas Bheda_, _Mitzalo Reyes_, and _Manuel Velarde_

This project explores attribute inference attacks by conducting experiments on decision tree, random forest, and XGBoost machine learning models. It systematically trains these models to the optimal point, as well as deliberately underfits and overfits them. Then, it attempts to infer sensitive input attributes from their predictions over all possible inputs for categorical features.

### Project Structure

```
./
├── README.md
├── setup.sh
│
├── images/
│
├── data/
│   ├── output/
│   │
│   ├── all_datasets.json
│   ├── insurance.json
│   ├── occupation.json
│   ├── personality.json
│   ├── stackoverflow.json
│   │
│   ├── insurance.csv
│   ├── occupation.csv
│   ├── personality.csv
│   └── stackoverflow.csv
│
├── notebooks/
│   ├── data_preparation.ipynb
│   └── proof_of_concept.ipynb
│
├── run_experiment.py
├── helper_functions.py
├── attribute_inference.py
├── attribute_inference_parallel.py
├── process_results.py
│
└── requirements.txt
```

- The `./setup.sh` bash script prepares the experiment to run.
- The `./images/` directory will store all figures generated from the experiment.
- The `./data/output/` directory will store all the experiment results after it runs.
- The JSON files in the `./data/` directory define the target variable, separate categorical versus numerical features, and describe the dataset.
- The CSV files in the `./data/` directory contain the various datasets.
- The `./notebooks/` directory contain the proof of concept for the experiment and the preprocessing done to the datasets.
- The `./run_experiment.py` script executes the experiment and generates figures from the results.
- The `./helper_functions.py` script contains various methods to assist the experiment.
- The `./attribute_inference.py` and `./attribute_inference_parallel.py` contains the method to execute the inference attack.
- The `./process_results.py` script contains the methods to generate figures from the experiment results.
- The `./requirements.txt` lists the required modules to execute the experiment.

### Project Setup

#### Linux

1. Change directory into the project location.

```{bash}
cd /path/to/project
```

2. Then **source** `setup.sh` script to create the virtual environment in `./.venv`, install all required Python modules, and activate the environment.

```{bash}
source setup.sh
```

3. To run the experiment with all the available datasets execute the following command from the activated environment.

```{bash}
python run_experiment.txt
```

To run a specific dataset use the `-d`  flag and specify the JSON file for a the dataset.

```{bash}
python run_experiment.txt -d ./data/insurance.json
```

#### Other Operating Systems

1. Create a Python virtual environment in the project directory using the `venv` module.
2. Activate the python environment.
3. Install requirements listed in `requirements.txt`.
4. Run experiment file `run_experiment.py` whithin the virtual environment.

### Experiment Configuration

#### Data Configuration

To run the experiment on a new dataset a JSON configuration file is required:

```{json}
{
  "dataset_identifier": {
    "filename": "dataset_filename.csv",
    "target": "target_column_name",
    "categorical_features": [
      "categorical_column_1_name",
      "categorical_column_2_name"
    ],
    "numerical_features": [
      "numerical_column_1_name",
      "numerical_column_2_name"
    ],
    "description": "Short description of the dataset"
  }
}
```

- `dataset_identifier` can be any string value used to identify the dataset in the results.
- `filename` should have the name of a CSV file in the `./data/` directory.
- `target` should have the column name for the feature the tree-based models will predict.
- `categorical_features` should list all the column names of categorical features in the dataset included in the experiment.
- `numerical_features` should list all numerical columns to be included in the tree-based models.
- `description` can have a short description of the dataset. It is not necessary.

### Results

After running the experiment, results data can be found in the `./data/output/` directory. When running the experiment with the `-i` or `--images`, report images can be found in the `./images/` directory.