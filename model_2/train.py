import time
import torch
import re
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from models import TransformerLM
import pickle

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
# 1. Load GloVe Pre-trained Embeddings
#############################################
def load_glove_embeddings(glove_path, vocab, embed_dim):
    embeddings_index = {}
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim))
    for word, i in vocab.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    
    return torch.tensor(embedding_matrix, dtype=torch.float32)

#############################################
# 2. Load and Preprocess Dataset (Curriculum Learning)
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

def build_vocab(text):
    tokens = text.lower().split()
    vocab = {token: idx + 2 for idx, token in enumerate(set(tokens))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def tokenize(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in text.lower().split()]

#############################################
# 3. Training Function with Knowledge Distillation
#############################################
def train_model(student, teacher, dataloader, optimizer, criterion, num_epochs=5, device='cpu', accumulation_steps=4):
    student.to(device)
    teacher.to(device)
    teacher.eval()  # Teacher does not update

    for epoch in range(num_epochs):
        total_loss = 0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)

            # Knowledge Distillation Loss
            loss = criterion(student_logits.view(-1, student_logits.size(-1)), y.view(-1))
            kd_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_logits, dim=-1), 
                                     nn.functional.softmax(teacher_logits, dim=-1))
            total_loss = 0.7 * loss + 0.3 * kd_loss

            # Gradient Accumulation
            total_loss = total_loss / accumulation_steps
            total_loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss.item():.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Time: {time.time() - start_time:.2f} sec")
        torch.save(student.state_dict(), "transformer_lm.pth")

#############################################
# 4. Main Training Routine
#############################################
if __name__ == '__main__':
    file_path = "movie_lines.txt"
    glove_path = "glove.6B.100d.txt"

    file_path = "movie_lines.txt"
    print(f"Loading movie conversations from {file_path}...")
    all_text = load_movie_conversations(file_path)

    vocab = build_vocab(all_text)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary size: {len(vocab)}")
    
    token_ids = tokenize(all_text, vocab)
    print(f"Number of tokens: {len(token_ids)}")

    dataset = TextDataset(token_ids, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    pretrained_embeddings = load_glove_embeddings(glove_path, vocab, 100)
    print(f"Pretrained embeddings loaded: {pretrained_embeddings.shape}")

    student_model = TransformerLM(len(vocab), 100, 4, 256, 2, 32, pretrained_embeddings)
    teacher_model = TransformerLM(len(vocab), 256, 8, 512, 4, 32)

    optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print('Training model...')
    train_model(student_model, teacher_model, dataloader, optimizer, criterion, num_epochs=5, device='mps')