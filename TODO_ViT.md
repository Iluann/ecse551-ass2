# ViT : prochaines étapes

> **Règle d'or : une seule modification à la fois**, puis on relance et on compare la **val accuracy**.
> La train loss seule ne suffit pas : l'augmentation et le dropout la font *monter*, et c'est normal.

---

## ⭐ Priorités choisies (dans cet ordre)

### 0. ⭐ Validation courte
- [ ] Mettre 10 % du train de côté (`randperm` avec une seed fixe) → `train_loader` / `val_loader`
- [ ] Afficher la val accuracy à la fin de chaque epoch (`model.eval()` + `torch.no_grad()`)

**Pourquoi :** le test n'a pas de labels et Kaggle limite les soumissions. La val donne une **courbe par epoch**, illimitée, pour régler tous les hyperparamètres.
**Où :** cellule `AugmentedDataset` (le split) + cellule d'entraînement (l'évaluation).
**À la fin :** réentraîner sur 100 % du train pour la soumission finale.

### 1. ⭐ LR scheduler : warmup + cosine decay
- [ ] `LambdaLR` : warmup linéaire sur ~5 epochs, puis descente en cosinus jusqu'à 0
- [ ] `sched.step()` **à chaque batch**, après `opt.step()`
- [ ] Passer à ~50 epochs (le decay n'a d'effet qu'à la fin)
- [ ] 1er run avec `lr=3e-4` (pour comparer), puis essayer `1e-3`

**Pourquoi (Adam ne suffit pas) :** Adam fait un pas ≈ `lr × m/√v`, où `m/√v` est de l'ordre de 1. Il égalise l'échelle *entre paramètres*, mais la **taille absolue du pas reste `lr`**.
- *Début :* les statistiques `m`, `v` sont bruitées et les gradients chaotiques → le **warmup** évite de saturer les softmax de l'attention.
- *Fin :* les pas restent de taille ~`lr` près du minimum → sans **decay**, le modèle tourne autour sans converger.

**Où :** cellule d'entraînement.

### 2. ⭐ Normalisation des entrées
- [ ] `T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` (stats ImageNet : RIVAL10 est un sous-ensemble d'ImageNet)
- [ ] Appliquée au **train, à la val et au test**
- [ ] Monitor : dé-normaliser l'image avant l'affichage (`img * STD + MEAN`)

**Pourquoi :** des pixels tous positifs font que tous les poids d'un neurone de la 1re couche reçoivent un gradient de même signe, d'où un apprentissage en zigzag. Les inits supposent aussi des entrées centrées.
**Où :** cellule `ImageDataset` (+ `plot_attention` dans le monitor).

### 3. ⭐ Plus d'augmentations
- [ ] `RandomResizedCrop(224, scale=(0.5, 1.0))` (au lieu de 0.7)
- [ ] `RandomHorizontalFlip()`
- [ ] `TrivialAugmentWide()` (remplace `ColorJitter`), **sur uint8, avant** la conversion en float
- [ ] `RandomErasing(p=0.25)`, **après** `Normalize`
- [ ] Aucune augmentation sur la val et le test

**Pourquoi :** un ViT n'a pas les biais d'un CNN (localité, invariance à la translation). Il doit les apprendre des données, sinon il mémorise le train.
**Ordre :** `crop → flip → TrivialAugment (uint8) → ToDtype float → Normalize → RandomErasing`
**Où :** cellule `ImageDataset` / `AugmentedDataset`.

### 4. ⭐ Remettre le dropout
- [ ] `config` : ajouter `"attention_probs_dropout_prob": 0.0` (garder `hidden_dropout_prob: 0.1`)
- [ ] `Embeddings` : dropout après l'ajout du PE
- [ ] `AttentionHead` : paramètre `dropout` + dropout sur `attention_probs`
- [ ] `MultiHeadAttention` : passer le dropout aux têtes + `output_dropout` après la projection

**Pourquoi :** il éteint aléatoirement des activations, ce qui empêche la co-adaptation et revient à un ensemble de sous-réseaux. Il est désactivé automatiquement en `model.eval()`.
**Attention :** avec une augmentation forte, trop de dropout fait sous-apprendre (train acc et val acc basses toutes les deux) : il faut alors le baisser.

### 5. ⭐ Positional embedding : base
- [x] **Garder `base = 1000`** : c'est mieux adapté que 10000 pour 197 tokens (≈70 % des dimensions utiles, contre ≈50 %)
- [ ] *(Optionnel, plus tard)* tester un PE **appris** (`nn.Parameter` + `trunc_normal_(std=0.02)`), et ne l'adopter que si la val le confirme

**Pourquoi :** la vraie limite est la numérotation 1D ligne par ligne (les patchs 13 et 14 sont « voisins » dans le PE mais à deux bords opposés de l'image), pas la base.

---

## Ensuite (si le temps le permet)

| Priorité | À faire | Gain attendu |
|---|---|---|
| Haute | Mixup / CutMix (`T.MixUp`, `T.CutMix` sur le batch) | ++ (gros gain pour un ViT sur peu de données) |
| Haute | `CrossEntropyLoss(label_smoothing=0.1)` | + (une ligne) |
| Haute | Init `trunc_normal_(std=0.02)` pour le `cls_token` et les `Linear` (actuellement `randn`, std = 1) | + |
| Moyenne | Stochastic depth / DropPath (~0.1) dans chaque `Block` | + |
| Moyenne | Gradient clipping `clip_grad_norm_(…, 1.0)` | stabilité |
| Moyenne | Pas de weight decay sur les LayerNorm, les biais, le `cls_token` et le PE | + |
| Moyenne | Stem convolutionnel, ou images plus petites + patch 8 | ++ |
| Moyenne | Global average pooling au lieu du token CLS | + |
| Vitesse | `F.scaled_dot_product_attention` + QKV fusionnés | ×2 à ×4 plus rapide |
| Vitesse | Mixed precision (`torch.autocast`, bf16) | ×1.5 à ×2 plus rapide |
| Inférence | TTA : moyenne des logits image + image retournée | +0.5 à 1 % |
| ? | Poids pré-entraînés (`timm` `deit_small`), **si l'énoncé l'autorise** | +++ (probablement plus de 90 %) |

---

## Suivi des runs

| Run | Changement | Val acc | Kaggle | Notes |
|---|---|---|---|---|
| 0 | Baseline + validation | | | |
| 1 | + LR scheduler | | | |
| 2 | + Normalisation | | | |
| 3 | + Augmentations | | | |
| 4 | + Dropout | | | |
