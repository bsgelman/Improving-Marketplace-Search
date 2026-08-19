# Improving Marketplace Search

Semantic product search for e-commerce, built for
[The Data Science Union at UCLA](https://github.com/the-data-science-union),
Winter 2025.

Keyword search fails when a shopper and a product listing use different words for
the same thing. The fix here is a **dual encoder**: one transformer embeds the
search query, a second embeds the product description, and both land in the same
vector space. Retrieval is then just a nearest-neighbour lookup, so a query finds
products that *mean* what it asked for rather than products that happen to repeat
its words.

Everything is Jupyter notebooks, one set per Amazon product category.

## The pipeline

Each category runs through five stages, one notebook per stage.

**1. Filter** — `*_filtering.ipynb`
Load raw [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) product
metadata (`.jsonl`), keep `title`, `description` and `features`, and drop products
with too few ratings or too little description text to be worth indexing.
Thresholds are picked per category by eye from the distributions.

**2. Cluster** — `cluster_*.ipynb`
Group products into topics with BERTopic — MiniLM sentence embeddings, UMAP
dimensionality reduction, HDBSCAN clustering — plus TF-IDF key terms per cluster.
These labels are not the end product; they exist to make stage 4 possible.

**3. Generate queries** — `generate_queries_*.ipynb`
Real query/product click data isn't public, so GPT writes five realistic customer
search queries per product. That synthetic set is the training data.

**4. Train the dual encoder** — `dual_encoder_*.ipynb`
Two separate transformer encoders — a shallow one for short queries, a deeper one
for long product text — trained with InfoNCE loss over in-batch negatives *plus*
hard negatives sampled from different clusters. Mining negatives from other
clusters is what forces the model to learn real distinctions instead of the easy
ones. Tracked with hits@10 and MRR on a held-out split.

**5. Search** — `search_*.ipynb`
Index the learned product embeddings in FAISS and run live queries, rendering
results with thumbnails, average rating and store name.

## Results

Retrieval quality on the held-out split, at the best epoch of each run. Every model
starts near chance and climbs, so the encoders are learning something real about
query/product alignment:

| Category | Val products | hits@10 (start to best) | best MRR | Epochs |
|---|:-:|:-:|:-:|:-:|
| Beauty Products | 68 | 0.22 → **0.85** | 0.444 | 44 |
| Pet Supplies | 65 | 0.15 → **0.74** | 0.354 | 47 |
| Home Products | 59 | 0.17 → **0.73** | 0.330 | 50 |
| CDs & Vinyl | 56 | 0.20 → **0.50** | 0.224 | 66 |
| Appliances | 50 | 0.20 → **0.50** | 0.214 | 24 |

Read these as directional, not as benchmark numbers. Three caveats worth stating:

- The validation sets are 50 to 68 products. Small enough that a few products
  moving in or out of the top 10 swings hits@10 by several points.
- The same split drives early stopping, so best-epoch figures are optimistic.
- Queries are LLM-generated, so this measures retrieval against *plausible*
  customer phrasing, not against how people actually search.

Appliances and CDs/vinyl lag the rest. Appliances stopped early at 24 epochs and
was still improving; CDs and vinyl ran the longest and plateaued at 0.50, which
fits a catalogue where titles are artist and album names rather than descriptive
text the encoder can latch onto.

## Category coverage

Not every category made it through every stage.

| Category | Filter | Cluster | Queries | Encoder | Search |
|---|:-:|:-:|:-:|:-:|:-:|
| Appliances | ✅ | ✅ | ✅ | ✅ | ✅ |
| Beauty Products | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pet Supplies | ✅ | ✅ | ✅ | ✅ | ✅ |
| Home Products | — | ✅ | ✅ | ✅ | ✅ |
| CDs & Vinyl | ✅ | ✅ | ✅ | ✅ | — |
| Groceries | ✅ | ✅ | ✅ | — | — |
| Health & Personal Care | ✅ | ✅ | — | — | — |
| Baby Products | ✅ | — | — | — | — |
| Musical Instruments | ✅ | — | — | — | — |
| Tools & Home Improvement | ✅ | — | — | — | — |

## Running the notebooks

- Notebooks read and write a local `data/` directory that is not committed. Start by
  downloading the category metadata you want from the Amazon Reviews 2023 dataset.
- Stage 3 needs `OPENAI_API_KEY` in your environment.
- Stage 4 expects a GPU.
- `product_filtering_template.ipynb` is the blank starting point for adding a new
  category.
