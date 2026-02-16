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
```

This will:
1. Load GPT-2 via TransformerLens
2. Collect ~10,000 text samples from OpenWebText
3. Extract residual stream activations from layer 6
4. Train an SAE (768 → 6144 → 768 with 8x expansion)
5. Analyse features using VocabProj method
6. Save all results to `outputs/`

### Quick Test

```bash
python main.py --quick
```

Runs a minimal test (500 samples, 1 epoch) to verify the pipeline works.

### Resume from Cached Activations

```bash
python main.py --skip-collection
```

Skips activation collection and uses cached `outputs/activations.pt`.

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