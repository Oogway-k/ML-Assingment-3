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
![Linux CW3](L_CW3.png)

* Train ↓ from ~7.4 → **~4.9**
* Val ↓ from ~6.5 → **~5.7**
* Gap small and steady ⇒ good generalization; no late-epoch blow-up.

**CW=5 (35 epochs)**
![Linux CW5](L_CW5.png)

* Train ↓ to **~4.85–4.9**
* Val plateaus **~5.65–5.7**
* Net: **marginal** improvement vs CW3; diminishing returns beyond CW=3.

**Takeaway (code):** Increasing CW from 3→5 yields *very small* validation gains.

---

### Shakespeare (text)

**CW=3 (147 epochs)**
![Shakespeare CW3](S_cw3.png)

* Train converges near **~5.62–5.66**
* Val stabilizes near **~5.74–5.76**
* Curves flatten ~30 epochs; long tail of fine-tuning gives tiny gains.

**CW=5 (139 epochs)**
![Shakespeare CW5](S_cw5.png)

* Train ~**5.60–5.64**, Val ~**5.72–5.75**
* Practically indistinguishable from CW=3.

**Takeaway (text):** Larger CW does **not** materially reduce validation loss; useful context is mostly local.

---

## 2) Embeddings: t-SNE Snapshots & Interpretation

### Linux code embeddings

**CW=3**
![WE\_L\_CW3](WE_L_CW3.png)

* Clear groupings: control tokens (`if/for/return`) vs function/identifier tokens (`main/sizeof/init`), and symbols.
* Clusters are **tight** and role-consistent.

**CW=5**
![WE\_L\_CW5](WE_L_CW5.png)

* Broader spread; more mixing between control/identifier regions.
* **Interpretation:** Wider context injects variability; CW=3 gives crisper, role-aligned structure for code.

### Shakespeare embeddings

**CW=3**
![WE\_S\_CW3](WE_S_CW3.png)

* Semantics emerge: `king–queen` (royalty), `he–she` (pronouns), `happy–sad` (emotion), `run–walk–fast/slow` (action/speed).
* Nearest-neighbour relations are intuitive.

**CW=5**
![WE\_S\_CW5](WE_S_CW5.png)

* Relations persist but clusters **spread**; neighbourhoods still meaningful.

**Embedding conclusion:** CW=3 yields **tighter, more interpretable** clusters in both domains; CW=5 preserves relations but adds variance.

---

## 3) Example Generations (qualitative)

* **Code (CW=3/5):** Produces locally coherent C fragments (balanced control flow tokens, proper use of `return`, common idioms). Long-range variable naming consistency is limited (expected for MLP).
* **Text (CW=3/5):** Shakespeare-like phrasing with plausible local grammar; occasional semantic drift beyond ~8–12 tokens without punctuation anchors.
* **Temperature:** Lower T (≈0.7) → safer, repetitive completions; higher T (≈1.1) → diverse but error-prone. Best perceived quality around **T≈0.9–1.0**.

---

## 4) Streamlit App (deployment outcomes)

* Supports **next-k word** continuation until stop token: `'.'` for text, `';'` for code.
* Controls exposed: **context length, embedding dim, activation, seed, temperature**, and model variant selector (CW=3 or CW=5).
* **OOV handling:** User input tokens not in vocab are mapped to `<unk>`; decoding avoids sampling `<unk>` to keep outputs readable.
* UX note: CW=3 variant is faster and subjectively as good as CW=5 for both datasets.

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

* Case-folded; punctuation kept per-domain rule (keep `.` for text; keep programming symbols for code).
* Long-tail frequency observed in both corpora (Zipf-like); top-10 tokens dominate counts.
* **Detailed tables (vocab size, top-10, bottom-10)** are reported in the notebook accompanying this README.

---

### One-line conclusion

> **CW=3** gives nearly the **same validation performance** as **CW=5** while producing **cleaner embeddings** and **faster inference** — for both Shakespeare (natural) and Linux C (structured) text.
