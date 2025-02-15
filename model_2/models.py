import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, seq_length, pretrained_embeddings=None):
        super(TransformerLM, self).__init__()
        self.seq_length = seq_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_length, embed_dim)  # Ensure pos_embedding has `seq_length` size

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = False  # Freeze embeddings for stability

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Ensure positions don't exceed embedding size
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        positions = torch.clamp(positions, max=self.seq_length - 1)  # ✅ Fix: Prevent out-of-range indexing

        x = self.embedding(x) + self.pos_embedding(positions)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)  # (batch, seq_len, embed_dim)
        logits = self.fc(x)  # (batch, seq_len, vocab_size)
        return logits