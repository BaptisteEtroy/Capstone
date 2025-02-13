import os
import re
import time
import pickle
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from initial_transformer.models import TransformerLM  # Import the model from models.py

#############################################
# 1. Load and Preprocess Movie Conversations
#############################################
def load_movie_conversations(file_path):
    conversations = []
    
    # ISO-8859-1 encoding
    with open(file_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                conversations.append(parts[4])
    
    all_text = " ".join(conversations)
    all_text = re.sub(r"[^a-zA-Z0-9.,!?'\"]+", " ", all_text)  # Clean noisy characters
    return all_text

#############################################
# 2. Text Preprocessing and Tokenization
#############################################
def build_vocab(text, min_freq=1):
    tokens = text.lower().split()
    freq = Counter(tokens)
    vocab = {token: idx + 2 for idx, (token, count) in enumerate(freq.items()) if count >= min_freq}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def tokenize(text, vocab):
    tokens = text.lower().split()
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

#############################################
# 3. Dataset for Language Modeling
#############################################
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_length=32):
        self.token_ids = token_ids
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.token_ids) - self.seq_length
    
    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y

#############################################
# 4. Training Loop
#############################################
def train_model(model, dataloader, NUM_EPOCHS, LR, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # Reduce LR every epoch by 30%
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f} sec")

        scheduler.step()

        # Overwrite model weights after each epoch
        torch.save(model.state_dict(), "transformer_lm.pth")
        print(f"Model saved after epoch {epoch+1}.")

#############################################
# 5. Main Training Routine
#############################################
if __name__ == '__main__':
    # Load the movie conversations dataset
    file_path = "movie_lines.txt"
    print(f"Loading movie conversations from {file_path}...")
    all_text = load_movie_conversations(file_path)

    if not all_text.strip():
        raise RuntimeError("Failed to load conversation data.")
    
    # Build vocabulary and tokenize text
    vocab = build_vocab(all_text, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")
    token_ids = tokenize(all_text, vocab)
    print(f"Number of tokens in the dataset: {len(token_ids)}")

    # Create dataset and dataloader
    seq_length = 32
    dataset = TextDataset(token_ids, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize the model
    embed_dim = 512 
    num_heads = 4
    hidden_dim = 512
    num_layers = 6
    model = TransformerLM(len(vocab), embed_dim, num_heads, hidden_dim, num_layers, seq_length)

    # Load existing weights if available
    model_path = "transformer_lm.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model weights.")

    # Select device: MPS (Apple GPU) or CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple GPU).")
    else:
        device = torch.device('cpu')
        print("Using CPU.")
    
    # Train the model
    NUM_EPOCHS = 5
    LR = 1e-4
    train_model(model, dataloader, NUM_EPOCHS, LR, device=device)
    
    # Final save after training completion
    torch.save(model.state_dict(), "transformer_lm.pth")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Training complete. Model and vocabulary saved.")
