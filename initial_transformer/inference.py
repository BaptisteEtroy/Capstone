import torch
import torch.nn.functional as F
import pickle
from initial_transformer.models import TransformerLM  # Import the model

# Function to build reverse vocab for decoding
def build_reverse_vocab(vocab):
    return {idx: token for token, idx in vocab.items()}

# Text generation function with temperature control
def generate_text(model, vocab, prompt, max_length=20, temperature=0.8, device='cpu'):
    model.eval()
    reverse_vocab = build_reverse_vocab(vocab)

    # Tokenize the prompt
    tokens = prompt.lower().split()
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate tokens
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_tensor)
        last_logits = logits[0, -1, :] / temperature
        probs = F.softmax(last_logits, dim=0)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=device)], dim=1)

        # Stop if an end-of-sequence token (if defined) is generated
        if next_token_id == vocab.get("<EOS>", None):
            break

    # Decode and return the generated text
    generated_text = " ".join([reverse_vocab.get(token, "<UNK>") for token in input_tensor.squeeze().tolist()])
    return generated_text[len(prompt):].strip()

if __name__ == "__main__":
    # Load model and vocab
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    model = TransformerLM(
        vocab_size=len(vocab), embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, seq_length=32
    )
    model.load_state_dict(torch.load("transformer_lm.pth", map_location=device))
    model.to(device)

    # Start inference loop
    print("=== Interactive Inference Mode ===")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input("User: ")
        if prompt.strip().lower() == "exit":
            break
        response = generate_text(model, vocab, prompt, max_length=20, temperature=0.8, device=device)
        print("Bot:", response)