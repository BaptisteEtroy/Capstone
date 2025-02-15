import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, seq_length):
        super(TransformerLM, self).__init__()
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_length, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)  # (batch, seq_len, embed_dim)
        logits = self.transformer(x)  # (batch, seq_len, embed_dim)
        logits = self.fc(logits)  # (batch, seq_len, vocab_size)
        return logits
