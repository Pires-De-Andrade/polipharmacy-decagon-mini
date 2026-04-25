# Decagon Mini

Reprodução do modelo **Decagon** (Zitnik et al., 2018) para predição de efeitos adversos de polifarmácia, otimizado para CPU.

> **Paper:** *Modeling polypharmacy side effects with graph convolutional networks*
> M. Zitnik, M. Agrawal, J. Leskovec — Bioinformatics, 2018

## 📋 Visão Geral

O Decagon é um modelo baseado em GCN relacional (R-GCN) que opera sobre um grafo multimodal heterogêneo para predizer efeitos adversos causados por combinações de medicamentos (polifarmácia). O grafo integra:

- **Rede PPI** (proteína ↔ proteína)
- **Alvos moleculares** (droga → proteína)
- **Efeitos adversos** (droga ↔ droga, um tipo de aresta por efeito)

### Subconjunto Reduzido

Como o dataset completo é inviável em CPU (645 drogas, 19K proteínas, 5.4M arestas), trabalhamos com um subconjunto:

| Parâmetro | Valor |
|-----------|-------|
| Top drogas (cobertura TWOSIDES) | 100 |
| Top efeitos adversos | 50 |
| Proteínas | apenas as conectadas via drug-target |

## 🗂️ Estrutura do Projeto

```
polipharmacy-decagon-mini/
├── dataset/
│   ├── raw/               ← CSVs originais (SNAP Stanford)
│   └── processed/         ← dados filtrados + grafo PyG
├── src/
│   ├── data/
│   │   ├── loader.py      ← leitura + filtragem dos CSVs
│   │   └── graph_builder.py ← construção do HeteroData PyG
│   ├── model/             ← (futuro) R-GCN + DEDICOM
│   └── training/          ← (futuro) loop de treino
├── scripts/
│   └── check_data.py      ← verificação e estatísticas
├── app/                   ← (futuro) interface Streamlit
├── notebooks/
├── results/
├── saved_models/
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Processar dados brutos → subconjunto filtrado
python -m src.data.loader

# 3. Construir grafo heterogêneo PyG
python -m src.data.graph_builder

# 4. Verificar estatísticas do grafo
python scripts/check_data.py
```

## ⚙️ Configuração

Os filtros do subconjunto podem ser ajustados em `src/data/loader.py`:

```python
N_DRUGS = 100          # Top drogas por cobertura
N_SIDE_EFFECTS = 50    # Top efeitos adversos
MIN_COMBO_PER_SE = 10  # Mín. pares por efeito
```

## 📊 Dados

Os dados originais são do [SNAP Stanford - Decagon](http://snap.stanford.edu/decagon/):

| Arquivo | Descrição |
|---------|-----------|
| `bio-decagon-combo.csv` | Pares de drogas + efeito adverso (TWOSIDES) |
| `bio-decagon-ppi.csv` | Interações proteína-proteína |
| `bio-decagon-targets.csv` | Alvos moleculares (droga → proteína) |
| `bio-decagon-mono.csv` | Efeitos adversos individuais |
| `bio-decagon-effectcategories.csv` | Categorias dos efeitos |

## 🔧 Stack

- Python 3.10+
- PyTorch 2.2 (CPU)
- PyTorch Geometric 2.5
- pandas, scikit-learn
- Streamlit (interface — futuro)
