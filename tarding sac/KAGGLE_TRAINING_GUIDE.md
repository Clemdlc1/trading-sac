# Guide d'Entraînement SAC sur Kaggle 🚀

Ce guide explique comment entraîner vos agents SAC sur Kaggle avec vos données pré-processées (fichier h5).

## 📋 Table des Matières

1. [Préparation des Données](#1-préparation-des-données)
2. [Configuration de Kaggle](#2-configuration-de-kaggle)
3. [Exécution de l'Entraînement](#3-exécution-de-lentraînement)
4. [Récupération du Modèle](#4-récupération-du-modèle)
5. [Utilisation avec le Code Local](#5-utilisation-avec-le-code-local)
6. [Paramètres et Optimisation](#6-paramètres-et-optimisation)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Préparation des Données

### Vérifier votre fichier h5

Assurez-vous que votre fichier `processed_data.h5` contient bien :

- `/train/EURUSD/` : Données d'entraînement (timestamp, open, high, low, close)
- `/val/EURUSD/` : Données de validation
- `/test/EURUSD/` : Données de test
- `/metadata/` : Métadonnées (pairs, dates, etc.)

### Générer le fichier h5 (si nécessaire)

Si vous n'avez pas encore généré le fichier h5 :

```bash
# Sur votre machine locale
cd "tarding sac"
python -c "
from backend.data_pipeline import DataPipeline
dp = DataPipeline()
dp.run_full_pipeline(force_download=True)
"
```

Le fichier sera créé dans : `data/processed/processed_data.h5`

---

## 2. Configuration de Kaggle

### Étape 1 : Créer un nouveau Notebook Kaggle

1. Allez sur [kaggle.com/code](https://www.kaggle.com/code)
2. Cliquez sur "New Notebook"
3. Choisissez "Python" comme langage
4. Activez GPU si disponible (Settings → Accelerator → GPU T4 x2)

### Étape 2 : Upload des Données

1. Créez un nouveau Dataset :
   - Allez sur [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Cliquez sur "New Dataset"
   - Uploadez votre fichier `processed_data.h5`
   - Nommez-le par exemple : `trading-data`
   - Faites-le en "Private" pour sécurité

2. Ajoutez le Dataset à votre Notebook :
   - Dans votre notebook, cliquez sur "Add data" (panneau de droite)
   - Cherchez votre dataset `trading-data`
   - Cliquez sur "Add"

### Étape 3 : Upload du Code Backend

Dans une cellule de votre notebook Kaggle :

```python
# Créer le répertoire backend
!mkdir -p backend

# Vous devez uploader tous les fichiers backend un par un
# Option 1: Via l'interface Kaggle (Add Data → Upload → New Dataset)
# Option 2: Via code cell (copier-coller le contenu des fichiers)
```

**Fichiers backend à uploader :**

- `backend/sac_agent.py` ⭐ (obligatoire)
- `backend/trading_env.py` ⭐ (obligatoire)
- `backend/data_pipeline.py` ⭐ (obligatoire)
- `backend/feature_engineering.py` ⭐ (obligatoire)
- `backend/hmm_detector.py` (si vous utilisez Agent 3)
- `backend/auxiliary_task.py` (si vous utilisez tâches auxiliaires)
- `backend/risk_manager.py` (optionnel)
- `backend/validation.py` (optionnel)

**Méthode recommandée** : Créez un dataset "trading-sac-backend" sur Kaggle contenant tous les fichiers backend, puis ajoutez-le à votre notebook.

### Étape 4 : Upload du Script d'Entraînement

Uploadez également le fichier `train_sac_kaggle.py` dans votre notebook Kaggle.

---

## 3. Exécution de l'Entraînement

### Installation des Dépendances

Dans une cellule Kaggle :

```python
# Installer les dépendances (la plupart sont déjà installées)
!pip install -q gymnasium h5py
```

### Méthode 1 : Utilisation Simple (via Python)

```python
# Importer le trainer
import sys
sys.path.insert(0, '/kaggle/working')

from train_sac_kaggle import KaggleTrainer

# Créer le trainer
trainer = KaggleTrainer(
    h5_path="/kaggle/input/trading-data/processed_data.h5",
    output_dir="/kaggle/working",
    num_episodes=100,           # Ajustez selon vos besoins
    eval_frequency=10,          # Évaluer tous les 10 épisodes
    checkpoint_frequency=5,     # Sauvegarder tous les 5 épisodes
    agent_id=1,                 # Agent 1, 2, ou 3
    device="auto"               # Utilisera GPU si disponible
)

# Lancer l'entraînement
agent, stats = trainer.run_training()
```

### Méthode 2 : Via Ligne de Commande

```python
# Dans une cellule Kaggle
!python train_sac_kaggle.py \
    --h5-path /kaggle/input/trading-data/processed_data.h5 \
    --output-dir /kaggle/working \
    --num-episodes 100 \
    --eval-frequency 10 \
    --checkpoint-frequency 5 \
    --agent-id 1 \
    --device auto
```

### Paramètres Disponibles

| Paramètre | Description | Défaut | Recommandation |
|-----------|-------------|--------|----------------|
| `--h5-path` | Chemin vers processed_data.h5 | `/kaggle/input/...` | Vérifier le chemin exact |
| `--output-dir` | Dossier de sortie | `/kaggle/working` | Laisser par défaut |
| `--num-episodes` | Nombre d'épisodes | 100 | 100-500 selon temps |
| `--eval-frequency` | Fréquence d'éval | 10 | 5-10 |
| `--checkpoint-frequency` | Fréquence sauvegarde | 5 | 5-10 |
| `--agent-id` | ID agent (1/2/3) | 1 | 1 pour commencer |
| `--device` | Device (cuda/cpu/auto) | auto | auto |

---

## 4. Récupération du Modèle

### Fichiers Générés

Après l'entraînement, vous trouverez dans `/kaggle/working/` :

```
/kaggle/working/
├── agent_1_best.pt              # ⭐ Meilleur modèle (selon validation)
├── agent_1_final.pt             # ⭐ Modèle final (dernier épisode)
├── checkpoints/
│   ├── agent_1_ep5.pt           # Checkpoint épisode 5
│   ├── agent_1_ep10.pt          # Checkpoint épisode 10
│   ├── ...
│   ├── metrics_ep5.json         # Métriques épisode 5
│   └── metrics_ep10.json        # Métriques épisode 10
└── logs/
    ├── training_stats.json      # Stats complètes (JSON)
    └── training_stats.npz       # Stats complètes (NumPy)
```

### Télécharger les Modèles

**Option 1 : Via l'interface Kaggle**

1. Cliquez sur l'icône "Save Version" en haut à droite
2. Choisissez "Save & Run All"
3. Une fois terminé, allez dans "Output" (panneau de droite)
4. Téléchargez `agent_1_best.pt` et `agent_1_final.pt`

**Option 2 : Via code**

```python
# Compresser les fichiers importants
!zip -r models.zip agent_*.pt checkpoints/ logs/

# Le fichier models.zip sera disponible dans l'Output du notebook
```

---

## 5. Utilisation avec le Code Local

### Charger le Modèle Entraîné sur Kaggle

```python
# Sur votre machine locale
from backend.sac_agent import SACAgent, SACConfig

# 1. Créer un agent avec la même configuration
config = SACConfig(
    state_dim=30,
    action_dim=1,
    hidden_dims=[256, 256],
    # ... autres paramètres (doivent correspondre)
)

agent = SACAgent(config=config, agent_id=1)

# 2. Charger le modèle entraîné sur Kaggle
agent.load("/path/to/downloaded/agent_1_best.pt")

# 3. Utiliser pour l'inférence
action = agent.select_action(state, evaluate=True)
```

### Vérification de Compatibilité

Le modèle entraîné sur Kaggle est **100% compatible** car :

- ✅ Même architecture (SACAgent)
- ✅ Même configuration (SACConfig)
- ✅ Mêmes features (FeaturePipeline)
- ✅ Même environnement (TradingEnvironment)
- ✅ Format de sauvegarde identique (.pt)

### Test de Validation

```python
# Tester le modèle chargé
from backend.data_pipeline import DataPipeline
from backend.feature_engineering import FeaturePipeline
from backend.trading_env import TradingEnvironment, TradingEnvConfig

# Charger les données de test
data_pipeline = DataPipeline()
_, _, test_data = data_pipeline.get_processed_data()

# Calculer les features
feature_pipeline = FeaturePipeline()
_, _, test_features = feature_pipeline.run_full_pipeline(
    train_data, val_data, test_data
)

# Créer l'environnement de test
env = TradingEnvironment(
    data=test_data['EURUSD'],
    features=test_features,
    config=TradingEnvConfig(),
    eval_mode=True
)

# Évaluer le modèle
state = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.select_action(state, evaluate=True)
    state, reward, done, info = env.step(action)
    total_reward += reward

print(f"Test Reward: {total_reward:.2f}")
print(f"Total Return: {info['total_return']:.2%}")
print(f"Sharpe Ratio: {info['sharpe_ratio']:.2f}")
```

---

## 6. Paramètres et Optimisation

### Configuration d'Entraînement Recommandée

**Pour un entraînement rapide (test) :**

```python
trainer = KaggleTrainer(
    num_episodes=50,
    eval_frequency=5,
    checkpoint_frequency=10
)
```

**Pour un entraînement complet :**

```python
trainer = KaggleTrainer(
    num_episodes=500,
    eval_frequency=10,
    checkpoint_frequency=20
)
```

**Pour un entraînement long (production) :**

```python
trainer = KaggleTrainer(
    num_episodes=1000,
    eval_frequency=20,
    checkpoint_frequency=50
)
```

### Durée Estimée

Avec GPU T4 x2 sur Kaggle :

- 1 épisode ≈ 2-5 minutes (selon longueur de l'épisode)
- 100 épisodes ≈ 3-8 heures
- 500 épisodes ≈ 16-40 heures

⚠️ **Limite Kaggle** : Les notebooks gratuits ont une limite de 12h/semaine de GPU. Planifiez en conséquence !

### Optimisations pour Kaggle

**1. Réduire la taille du replay buffer :**

Modifiez dans `backend/sac_agent.py` ou créez un SACConfig custom :

```python
config = SACConfig(
    buffer_capacity=50000,  # Au lieu de 100000
    batch_size=512,         # Au lieu de 1024
)
```

**2. Utiliser des épisodes plus courts :**

Modifiez dans `backend/trading_env.py` ou TradingEnvConfig :

```python
env_config = TradingEnvConfig(
    episode_lengths=[2000],  # Au lieu de [3000]
)
```

**3. Réduire les évaluations :**

```python
trainer = KaggleTrainer(
    eval_frequency=20,  # Évaluer moins souvent
)
```

---

## 7. Troubleshooting

### Problème : "FileNotFoundError: h5 file not found"

**Solution :**
- Vérifiez le chemin exact de votre dataset dans Kaggle
- Dans le panneau "Data", copiez le chemin affiché
- Utilisez ce chemin exact dans `--h5-path`

```python
# Trouver le chemin exact
!ls /kaggle/input/
!ls /kaggle/input/trading-data/
```

### Problème : "ModuleNotFoundError: No module named 'backend'"

**Solution :**
- Assurez-vous que tous les fichiers backend sont dans `/kaggle/working/backend/`
- Vérifiez que vous avez bien ajouté `sys.path.insert(0, '/kaggle/working')`

```python
import sys
sys.path.insert(0, '/kaggle/working')

# Vérifier
!ls /kaggle/working/backend/
```

### Problème : "CUDA out of memory"

**Solution :**
- Réduisez `batch_size` dans SACConfig
- Réduisez `buffer_capacity`
- Utilisez CPU au lieu de GPU : `--device cpu`

### Problème : "Notebook timeout (12h limit)"

**Solution :**
- Sauvegardez des checkpoints fréquents
- Relancez l'entraînement depuis le dernier checkpoint :

```python
# Charger depuis checkpoint
agent.load("/kaggle/working/checkpoints/agent_1_ep50.pt")

# Continuer l'entraînement
trainer.run_training()
```

### Problème : "Reward collapse (agent donne toujours 0)"

**Solution :**
- Vérifiez que `warmup_steps` est suffisant (5000 par défaut)
- Augmentez le nombre d'épisodes
- Vérifiez les hyperparamètres (learning rates, gamma, etc.)

---

## 📊 Monitoring de l'Entraînement

### Visualiser les Statistiques

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Charger les stats
with open('/kaggle/working/logs/training_stats.json', 'r') as f:
    stats = json.load(f)

# Plot des récompenses
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(stats['episode_rewards'])
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 3, 2)
plt.plot(stats['actor_losses'])
plt.title('Actor Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.subplot(1, 3, 3)
plt.plot(stats['alpha_values'])
plt.title('Alpha (Entropy Coefficient)')
plt.xlabel('Episode')
plt.ylabel('Alpha')

plt.tight_layout()
plt.show()
```

### Métriques Importantes à Surveiller

- ✅ **Episode Reward** : Doit augmenter progressivement
- ✅ **Actor/Critic Loss** : Doit se stabiliser
- ✅ **Alpha** : Doit converger vers une valeur stable
- ✅ **Eval Reward** : Doit être cohérent avec training reward
- ✅ **Sharpe Ratio** : Doit être > 0 (idéalement > 1)

---

## 🎯 Workflow Complet Recommandé

1. **Préparation (Local)** :
   - Générer `processed_data.h5`
   - Tester le code localement (quelques épisodes)

2. **Upload Kaggle** :
   - Créer dataset avec `processed_data.h5`
   - Créer dataset avec fichiers backend
   - Créer nouveau notebook

3. **Test Rapide (Kaggle)** :
   - Entraîner 10 épisodes pour vérifier que tout fonctionne
   - Vérifier les logs et métriques

4. **Entraînement Complet (Kaggle)** :
   - Lancer 100-500 épisodes
   - Surveiller régulièrement
   - Sauvegarder checkpoints

5. **Récupération (Local)** :
   - Télécharger `agent_X_best.pt`
   - Charger dans le code local
   - Valider sur données de test

6. **Production (Local)** :
   - Utiliser le meilleur modèle
   - Monitoring en temps réel
   - Re-entraîner périodiquement sur Kaggle

---

## 📝 Checklist Avant de Lancer

- [ ] Fichier `processed_data.h5` uploadé sur Kaggle Dataset
- [ ] Tous les fichiers backend uploadés (sac_agent.py, trading_env.py, etc.)
- [ ] Script `train_sac_kaggle.py` uploadé
- [ ] GPU activé (si disponible)
- [ ] Chemins vérifiés (`/kaggle/input/...`)
- [ ] Paramètres d'entraînement configurés
- [ ] Temps estimé < limite Kaggle (12h)

---

## 🚀 Prêt à Lancer !

Vous avez maintenant tout ce qu'il faut pour entraîner vos agents SAC sur Kaggle. Le modèle entraîné sera 100% compatible avec votre code local et prêt à l'emploi.

**Bon entraînement !** 🎉

---

## 📞 Support

En cas de problème :

1. Vérifiez la section [Troubleshooting](#7-troubleshooting)
2. Consultez les logs : `/kaggle/working/logs/`
3. Vérifiez les checkpoints : `/kaggle/working/checkpoints/`

---

**Version :** 1.0
**Date :** 2025-11-23
**Auteur :** Trading SAC System
