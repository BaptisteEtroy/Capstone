# Interpretability of LLMS

## Installation

```bash
cd Capstone
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Full Pipeline

```bash
python main.py

python steer.py

python label_features.py
```

config.py has all the training and model configurations as well as the data classes for the SAE

main.py trains the model and does a input/ouput centric feature analysis.

label_features.py does exactly what you think it does


## Output Files

```
outputs/
├── sae.pt                  # Trained SAE model
├── activations.pt          # Raw layer 6 activations
├── decoder_vectors.pt      # Feature directions [768, 6144]
├── feature_activations.pt  # Encoded features [n_tokens, 6144]
├── features.json           # Feature analysis (top 100)
├── training_history.json   # Loss curves
└── summary.json            # Run summary
```

## Configuration

All settings are hardcoded at the top of `main.py`: