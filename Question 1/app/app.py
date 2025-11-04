import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re 


class NextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=128, n_layers=1, activation='relu', context_window=3, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_window = context_window
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_prob)

        input_dim = embedding_dim * context_window
        for _ in range(n_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.view(batch_size, -1)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        logits = self.output(x)
        return logits



@st.cache_resource
def load_model(checkpoint_path, vocab_size, embedding_dim, hidden_dim, n_layers, activation, context_len):
    model = NextTokenPredictor(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_dim=128,
        n_layers=1,
        activation='relu',
        context_window=context_len
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def encode(tokens, stoi):
    unk_id = stoi.get("<unk>", 0)
    return torch.tensor([stoi.get(t, unk_id) for t in tokens]).unsqueeze(0)


def decode(indices, itos):
    return " ".join(itos[i] for i in indices)


def generate_text(model, stoi, itos, input_text, max_new_tokens, context_len, temperature=1.0, model_name="text", top_k=None):
    
    tokens = input_text.strip().split()
    tokens = tokens[-context_len:]
    x = encode(tokens, stoi)
    generated = tokens.copy()

    stop_token = "." if "text" in model_name.lower() else ";"

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(x).float()  
            logits = logits / max(temperature, 1e-5)  

            if top_k is not None and top_k < logits.size(-1):
                values, indices = torch.topk(logits, top_k, dim=-1)
                probs = torch.zeros_like(logits)
                probs.scatter_(-1, indices, F.softmax(values, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, 1).item()

        next_token = itos[idx_next]
        generated.append(next_token)

        if next_token == stop_token:
            break

        x = torch.cat([x, torch.tensor([[idx_next]])], dim=1)
        x = x[:, -context_len:]  

    return " ".join(generated)




st.title("🧠 Next-Token Predictor")

MODEL_PATHS = {
    "Text Model cxt = 3": "shakespeare_cw3.pth",
    "Text Model cxt = 5": "shakespeare_cw5.pth",
    "Code Model cxt = 3": "linux_cw3.pth",
    "Code Model cxt = 5": "linux_cw5.pth"
}

model_name = st.selectbox("Choose Model", MODEL_PATHS.keys())
model_path = MODEL_PATHS[model_name]

input_text = st.text_area("Input text/code")

col1, col2, col3 = st.columns(3)
max_tokens = col1.number_input("Tokens to generate", 1, 200, 25)
context_len = col2.slider("Context window", 3, 5)
temperature = col3.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

seed = st.number_input("Random Seed", 0, 9999, 42)


def load_vocab(model_name):
    if "text" in model_name.lower():
        with open("Shakespeare.txt", "r") as f:
            text = f.read()
        text = text.lower()
        text = text.replace('\n', ' ')
        text = re.sub(r'[-–—]+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9 \. ]', '', text)
        text = re.sub(r'\.', ' . ', text)
        words = text.split()
        vocab = set(words)
        vocab.add('<PAD>')
        stoi = {w: i for i, w in enumerate(vocab)}
        itos = {i: w for w, i in stoi.items()}
        
    else:
        with open("Linux.txt", "r") as f:
            text = f.read()
        words = text.split()  
        words = text.split()
        vocab = set(words)
        vocab.add('<PAD>')
        stoi = {w: i for i, w in enumerate(vocab)}
        itos = {i: w for w, i in stoi.items()}
    return stoi, itos


stoi, itos = load_vocab(model_name)
vocab_size = len(stoi)

if st.button("Generate"):
    if not input_text.strip():
        st.error("Enter text first!")
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = load_model(model_path, vocab_size, 32, 128, 1, 'relu', context_len)

        with st.spinner("Generating..."):
            result = generate_text(
                model, stoi, itos,
                input_text=input_text,
                max_new_tokens=max_tokens,
                context_len=context_len,
                temperature=temperature,
                model_name=model_name
            )

        st.success("✅ Done")
        st.subheader("Generated Output:")
        st.write(result)
