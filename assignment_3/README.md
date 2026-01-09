# Assignment 3

## Attribute Inference from ML Models

by Vishwas Bheda, Mitzalo Reyes, and Manuel Velarde

### Project Structure

```
./
│
├── data/
│   ├── data_config.json
│   │
│   ├── raw/
│   │   ├── insurance.csv
│   │   └── personality.csv
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
├── model_config.json
│
├── run_experiment.py
├── helper_functions.py
├── attribute_inference.py
├── attribute_inference_parallel.py
│
└── requirements.txt
```

### Project Setup

1. Create a virtual environment:

```{bash}
python -m venv .venv
```

2. Activate the python environment:

```{bash}
./.venv/bin/activate
```

3. Install requirements:

```{bash}
pip install -r requirements.txt
```

3. Run experiment:

```{bash}
python ./run_experiment.py -d ./data/data_settings.json -m ./model_settings.json
```

### Experiment Configuration

#### Data Configuration

#### Model Configuration

