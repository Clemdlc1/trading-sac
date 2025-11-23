# Entraînement SAC sur Kaggle - Script Standalone

Ce script autonome permet d'entraîner les agents SAC sur Kaggle avec vos données pré-processées.

## 🎯 Fichiers Nécessaires

Sur Kaggle, vous avez besoin de **2 fichiers seulement** :

1. **train_sac_standalone.py** (ce script - standalone, aucune dépendance backend)
2. **Votre fichier .h5** avec les données et features

## 📊 Préparation du Fichier H5

⚠️ **Important** : Votre système génère 2 fichiers h5 séparés :
- `processed_data.h5` (données OHLC)
- `features_normalized.h5` (features calculées)

Vous devez les **combiner en un seul fichier** avant d'uploader sur Kaggle.

### Étape de Fusion (EN LOCAL)

```bash
# Sur votre machine locale
cd "tarding sac"
python merge_h5_files.py
```

Cela crée `combined_data.h5` avec cette structure :

```
/train/
  /EURUSD/
    timestamp, open, high, low, close
  /features/
    feature_1, feature_2, ..., feature_30

/val/
  /EURUSD/
    (même structure)
  /features/
    (même structure)

/test/
  /EURUSD/
    (même structure)
  /features/
    (même structure)
```

**Uploadez UNIQUEMENT `combined_data.h5` sur Kaggle** (pas les fichiers séparés).

## 🚀 Utilisation sur Kaggle

### Étape 1 : Upload des Données

1. Créez un dataset sur Kaggle avec votre fichier .h5
2. Nommez-le par exemple `trading-eurusd-data`

### Étape 2 : Créer un Notebook

1. Créez un nouveau notebook Kaggle
2. Ajoutez votre dataset `trading-eurusd-data`
3. Activez le GPU (Settings → Accelerator → GPU T4 x2)

### Étape 3 : Installer les Dépendances

```python
# Dans une cellule Kaggle
!pip install -q gymnasium
```

### Étape 4 : Exécuter le Script

**Option A : Dans un Notebook Kaggle/Colab (Recommandé)**

Copiez tout le contenu de `train_sac_standalone.py` dans une cellule, puis exécutez :

```python
# Le script détecte automatiquement qu'il est dans un notebook
# et affiche les instructions

# Lancez l'entraînement avec la fonction train_sac()
trainer = train_sac(
    h5_path='/kaggle/input/trading-eurusd-data/data.h5',
    output_dir='/kaggle/working',
    num_episodes=100,
    eval_frequency=10,
    checkpoint_frequency=5,
    device='auto'
)
```

**Option B : Via Ligne de Commande**

Si vous uploadez le script via "Add Data" :

```bash
!python /kaggle/input/your-script/train_sac_standalone.py \
    --h5-path /kaggle/input/trading-eurusd-data/data.h5 \
    --output-dir /kaggle/working \
    --num-episodes 100 \
    --eval-frequency 10 \
    --checkpoint-frequency 5 \
    --device auto
```

## ⚙️ Paramètres

| Paramètre | Description | Défaut | Recommandation |
|-----------|-------------|--------|----------------|
| `--h5-path` | Chemin vers le fichier h5 | (requis) | `/kaggle/input/your-data/data.h5` |
| `--output-dir` | Dossier de sortie | `/kaggle/working` | Laisser par défaut |
| `--num-episodes` | Nombre d'épisodes | 100 | 100-500 |
| `--eval-frequency` | Fréquence d'évaluation | 10 | 5-10 |
| `--checkpoint-frequency` | Fréquence sauvegarde | 5 | 5-10 |
| `--device` | Device (cuda/cpu/auto) | auto | auto |

## 📦 Fichiers Générés

Après l'entraînement, vous trouverez dans `/kaggle/working/` :

```
/kaggle/working/
├── agent_best.pt              # ⭐ Meilleur modèle (selon validation)
├── agent_final.pt             # Modèle final
├── checkpoints/
│   ├── agent_ep5.pt
│   ├── agent_ep10.pt
│   └── ...
└── logs/
    ├── training_stats.json
    └── training_stats.npz
```

**Téléchargez** `agent_best.pt` pour l'utiliser en local !

## 💻 Utilisation en Local

Une fois le modèle téléchargé depuis Kaggle, vous pouvez le charger localement :

```python
from backend.sac_agent import SACAgent, SACConfig

# Créer un agent avec la même configuration
config = SACConfig(state_dim=30, action_dim=1)
agent = SACAgent(config=config, agent_id=1)

# Charger le modèle entraîné sur Kaggle
agent.load("agent_best.pt")

# Utiliser pour l'inférence
action = agent.select_action(state, evaluate=True)
```

## ✅ 100% Compatible

Le modèle entraîné avec ce script est **100% compatible** avec votre code local car :

- ✅ Même architecture (Actor, Critic avec Spectral Normalization)
- ✅ Même configuration (SACConfig, TradingEnvConfig)
- ✅ Même environnement (TradingEnvironment)
- ✅ Même format de features (30 dimensions)
- ✅ Même format de sauvegarde (.pt)

**Résultat** : Vous pouvez entraîner sur Kaggle et utiliser directement en local !

## 🔧 Configuration Avancée

Pour modifier les hyperparamètres, éditez directement dans le script :

```python
# Dans train_sac_standalone.py

@dataclass
class SACConfig:
    state_dim: int = 30
    action_dim: int = 1
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    actor_lr: float = 3e-4        # ← Modifier ici
    critic_lr: float = 3e-4       # ← Ou ici
    gamma: float = 0.95           # ← Ou ici
    # ...

@dataclass
class TradingEnvConfig:
    initial_capital: float = 500000.0
    risk_per_trade: float = 0.0005
    max_leverage: float = 2.0
    episode_length: int = 3000     # ← Ou ici
    # ...
```

## 📊 Monitoring

Le script affiche en temps réel :

```
📈 Episode 15/100
   Reward: 125.43
   Return: 2.35%
   Sharpe: 1.87
   MaxDD: 3.21%
   Actor Loss: 0.0234
   Alpha: 0.0156
   Steps: 45000 | Time: 183.5s

🔍 Évaluation...
   Eval Reward: 132.18 ± 12.45
   Eval Return: 2.51%
   Eval Sharpe: 1.92
   ⭐ Nouveau meilleur modèle!
```

## ⏱️ Durée Estimée

Avec GPU T4 x2 sur Kaggle :

- 1 épisode ≈ 2-5 minutes
- 100 épisodes ≈ 3-8 heures
- 500 épisodes ≈ 16-40 heures

⚠️ **Limite Kaggle** : 12h/semaine de GPU gratuit

## 🐛 Troubleshooting

### "No module named 'gymnasium'"

```python
!pip install gymnasium
```

### "FileNotFoundError: h5 file not found"

Vérifiez le chemin exact :

```python
!ls /kaggle/input/
!ls /kaggle/input/your-dataset/
```

### "CUDA out of memory"

Réduisez le batch size dans le script :

```python
@dataclass
class SACConfig:
    batch_size: int = 512  # Au lieu de 1024
```

### "Notebook timeout"

Sauvegardez un checkpoint et relancez depuis là :

```python
# Charger le dernier checkpoint
agent.load("/kaggle/working/checkpoints/agent_ep50.pt")

# Continuer l'entraînement
trainer.run()
```

## 📝 Checklist

- [ ] Fichier h5 avec données + features uploadé sur Kaggle
- [ ] Script train_sac_standalone.py uploadé (ou copié dans notebook)
- [ ] GPU activé
- [ ] Gymnasium installé (`!pip install gymnasium`)
- [ ] Chemin h5 vérifié
- [ ] Paramètres configurés

## 🎉 Prêt !

Vous avez tout ce qu'il faut pour entraîner votre agent SAC sur Kaggle !

Le modèle sera 100% compatible avec votre code local et prêt à l'emploi.

---

**Version** : 1.0 Standalone
**Date** : 2025-11-23
**Auteur** : Trading SAC System
