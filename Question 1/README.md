# Question 1 — Results: Next-Word Prediction with an MLP

**Datasets**

* **Category I (Natural language):** Shakespeare
* **Category II (Structured/domain):** Linux kernel/C code

**Model family**
MLP language model (shared config across runs): embedding {32/64}, 1–2 hidden layers (1024), ReLU/Tanh, softmax over vocab. Trained with next-word objective and context window (**CW**) ∈ {3, 5}.

---

## 1) Training & Validation Loss (final outcomes)

### Linux (code)

**CW=3 (36 epochs)**

* Train ↓ from ~7.4 → **~4.9**
* Val ↓ from ~6.5 → **~5.7**
* Gap small and steady ⇒ good generalization; no late-epoch blow-up.

**CW=5 (35 epochs)**

* Train ↓ to **~4.85–4.9**
* Val plateaus **~5.65–5.7**
* Net: **marginal** improvement vs CW3; diminishing returns beyond CW=3.

**Takeaway (code):** Increasing CW from 3→5 yields *very small* validation gains.

---

### Shakespeare (text)

**CW=3 (147 epochs)**

* Train converges near **~5.62–5.66**
* Val stabilizes near **~5.74–5.76**
* Curves flatten ~30 epochs; long tail of fine-tuning gives tiny gains.

**CW=5 (139 epochs)**

* Train ~**5.60–5.64**, Val ~**5.72–5.75**
* Practically indistinguishable from CW=3.

**Takeaway (text):** Larger CW does **not** materially reduce validation loss; useful context is mostly local.

<img src="Comparative%20Analysis/L_vs_V.png" alt="Screenshot" width="900" />
---

## 2) Embeddings: t-SNE Snapshots & Interpretation

### Linux code embeddings

**CW=3** 
* Clear groupings: control tokens (`if/for/return`) vs function/identifier tokens (`main/sizeof/init`), and symbols.
* Clusters are **tight** and role-consistent.
<img src="Dataset%202/WE_L_CW3.png" alt="Screenshot" width="400" />

**CW=5**
* Broader spread; more mixing between control/identifier regions.
* **Interpretation:** Wider context injects variability; CW=3 gives crisper, role-aligned structure for code.
<img src="Dataset%202/WE_L_CW5.png" alt="Screenshot" width="400" />

### Shakespeare embeddings

**CW=3**
* Semantics emerge: `king–queen` (royalty), `he–she` (pronouns), `happy–sad` (emotion), `run–walk–fast/slow` (action/speed).
* Nearest-neighbour relations are intuitive.
<img src="Dataset%201/WE_S_CW3.png" alt="Screenshot" width="400" />

**CW=5**
* Relations persist but clusters **spread**; neighbourhoods still meaningful.
<img src="Dataset%201/WE_S_CW5.png" alt="Screenshot" width="400" />

**Embedding conclusion:** CW=3 yields **tighter, more interpretable** clusters in both domains; CW=5 preserves relations but adds variance.

---

## 3) Example Generations (qualitative)

* **Code (CW=3/5):** Produces locally coherent C fragments (balanced control flow tokens, proper use of `return`, common idioms). Long-range variable naming consistency is limited (expected for MLP).
* **Text (CW=3/5):** Shakespeare-like phrasing with plausible local grammar; occasional semantic drift beyond ~8–12 tokens without punctuation anchors.
* **Temperature:** Lower T (≈0.7) → safer, repetitive completions; higher T (≈1.1) → diverse but error-prone. Best perceived quality around **T≈0.9–1.0**.

---

## 4) Streamlit App (deployment outcomes)

* Supports **next-k word** continuation.
* Controls exposed: **context length, embedding dim, activation, seed, temperature**, and model variant selector (CW=3 or CW=5).
* **OOV handling:** User input tokens not in vocab are mapped to `<unk>`; decoding avoids sampling `<unk>` to keep outputs readable.
* UX note: CW=3 variant is faster and subjectively as good as CW=5 for both datasets.

* ##Running the Pretrained Models

## 1️⃣ Download Model Checkpoints

All pretrained model checkpoints are hosted on Google Drive:
🔗 [Download from Google Drive](https://drive.google.com/drive/u/0/folders/1FEKwOQ3dG40Gdk4j_b3Ug5fk3R5PV1b6)

You can either:

* **Manually download** the `.pth` files and place them into the `checkpoints/` folder of this repository,
  **or**
* Use the command line (requires `gdown`):

  ```bash
  pip install gdown
  gdown --folder https://drive.google.com/drive/folders/1FEKwOQ3dG40Gdk4j_b3Ug5fk3R5PV1b6
  ```

This will automatically download all checkpoint files into your current working directory.

---

## 2️⃣ Place Checkpoints

Move all downloaded `.pth` files to:

```
project_root/
└── apps/
    ├── model_text.pth
    ├── model_code.pth
    └── ...
```

If your code expects checkpoints in a different folder, update the paths in your `load_model()` function accordingly.

---

## 3️⃣ Install Dependencies

Since project uses PyTorch:

```bash
pip install torch torchvision torchaudio
```

And for Streamlit (if running a web app):

```bash
pip install streamlit
```

---

## 4️⃣ Run the Application or Model Inference

To launch the Streamlit web app:

```bash
streamlit run app/app.py
```
---

## 5️⃣ (Optional) Modify Configuration

If your model expects configuration data within the checkpoint file, load it as:

```python
checkpoint = torch.load("checkpoints/model_text.pth", map_location="cpu")
model = YourModelClass(**checkpoint["config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```
---

## 5) Comparative Analysis (Category I vs II)

| Aspect           | Shakespeare (Natural)                              | Linux C (Structured)                                          | Observation                                                                   |
| ---------------- | -------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Val loss (final) | ~5.73–5.76                                         | ~5.65–5.70                                                    | Both plateau; code slightly lower, likely due to stronger local regularities. |
| CW effect (3→5)  | Minimal                                            | Minimal                                                       | Diminishing returns; local context dominates.                                 |
| Embeddings       | Clear semantic clusters; antonyms/pronouns/actions | Role-based clusters (control, funcs, symbols)                 | CW=3 produces tighter structure in both.                                      |
| Generation       | Fluent locally; occasional semantic drift          | Syntactically plausible lines; identifier consistency limited | MLP handles short-range dependencies well.                                    |

**Summary insight:**

* **Local context (3–5 tokens) is sufficient** for strong learning signals in both natural and structured text with an MLP.
* Code exhibits **higher predictability** at short range (syntax templates), while natural language relies more on stylistic variation.
* Larger CW slightly **blurs embedding geometry** without clear validation gains.

---

## 6) Preprocessing/Vocabulary (results summary)

### Vocabulary Statistics
#### 📝 Shakespeare Dataset
| Word | Frequency |
|------|-----------:|
| . | 33,850 |
| the | 26,221 |
| and | 23,534 |
| i | 20,049 |
| to | 18,644 |
| of | 16,358 |
| a | 13,601 |
| you | 13,546 |
| my | 12,019 |
| that | 10,481 |

#### 💻 Linux Code Dataset
| Token | Frequency |
|--------|-----------:|
| * | 33,504 |
| = | 28,003 |
| { | 18,915 |
| if | 17,702 |
| } | 16,965 |
| the | 16,080 |
| */ | 13,445 |
| /* | 12,190 |
| struct | 10,997 |
| return | 10,130 |

* Case-folded; punctuation kept per-domain rule (keep `.` for text; keep programming symbols for code).
* Long-tail frequency observed in both corpora (Zipf-like); top-10 tokens dominate counts.
* **Detailed tables (vocab size, top-10, bottom-10)** are reported in the notebook accompanying this README.

---

### One-line conclusion

> **CW=3** gives nearly the **same validation performance** as **CW=5** while producing **cleaner embeddings** and **faster inference** — for both Shakespeare (natural) and Linux C (structured) text.
