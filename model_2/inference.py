import torch
import torch.nn.functional as F
import pickle
from models import TransformerLM

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

device = 'cpu'

# Load trained model
model = TransformerLM(len(vocab), 100, 4, 256, 2, 32)
model.load_state_dict(torch.load("transformer_lm.pth", map_location=device))
model.to(device)
model.eval()

print(f"Loaded Model: TransformerLM with vocab size {len(vocab)}")
print(f"Embedding Layer Shape: {model.embedding.weight.shape}")

# Function to generate text with improved decoding
def generate_text(prompt, max_len=20, top_k=5, temperature=0.7):
    words = prompt.lower().split()
    input_ids = torch.tensor([[vocab.get(w, vocab["<UNK>"]) for w in words]], dtype=torch.long).to(device)

    for _ in range(max_len):
        logits = model(input_ids)[:, -1, :] / temperature  # Apply temperature
        probs = F.softmax(logits, dim=-1)

        # Top-k sampling
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        next_token = top_k_indices[0, torch.multinomial(top_k_probs[0], 1).item()].item()

        if next_token not in vocab.values():
            print(f"⚠️ Warning: Token ID {next_token} not found in vocab!")
            break

        next_word = [k for k, v in vocab.items() if v == next_token][0]

        words.append(next_word)
        input_ids = torch.cat((input_ids, torch.tensor([[next_token]]).to(device)), dim=1)

        # Stop if it generates padding (unlikely with top-k sampling)
        if next_token == vocab["<PAD>"]:
            break

    return " ".join(words)

# Run inference with better decoding
print("\nGenerated Text:")
print(generate_text("hello how are you", max_len=10, top_k=10, temperature=1.0))