# Action Space Discret - Documentation

## Résumé

L'environnement de trading utilise maintenant un **action space discret** avec 3 actions possibles.

## Actions Disponibles

| Action | Valeur | Comportement |
|--------|--------|--------------|
| **0** | `0.0` | **Flat** - Ferme toutes les positions et reste à plat |
| **1** | `1.0` | **Long** - Ouvre une position longue (et ferme short si existant) |
| **2** | `-1.0` | **Short** - Ouvre une position courte (et ferme long si existant) |

## Changements Techniques

### Avant
```python
# Action space continu
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
```

### Après
```python
# Action space discret
self.action_space = spaces.Discrete(3)  # Actions : 0, 1, 2
```

## Fichiers Modifiés

### `backend/trading_env.py`

1. **Configuration** (ligne 94-95) :
```python
# Action space - Discrete: {0: flat, 1: long, 2: short}
n_actions: int = 3  # Discrete action space with 3 actions
```

2. **Initialisation** (ligne 550-551) :
```python
# Discrete action space: 0 = flat (0.0), 1 = long (1.0), 2 = short (-1.0)
self.action_space = spaces.Discrete(self.config.n_actions)
```

3. **Mapping des actions** (ligne 594-612) :
```python
def _convert_discrete_action(self, action: int) -> float:
    """Convertit l'action discrète en valeur continue pour le position sizer."""
    action_mapping = {
        0: 0.0,   # Flat - ferme tout
        1: 1.0,   # Long - ferme short et ouvre long
        2: -1.0   # Short - ferme long et ouvre short
    }
    return action_mapping.get(int(action), 0.0)
```

4. **Step()** (ligne 649-661) :
```python
def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
    """Execute one step with discrete action (0, 1, or 2)."""
    action_continuous = self._convert_discrete_action(action)
    # ... reste du code
```

## Utilisation

```python
from backend.trading_env import TradingEnvironment

# Créer l'environnement
env = TradingEnvironment(data=train_data, features=train_features)

# Reset
obs = env.reset()

# Utiliser les actions discrètes
action = 1  # Long
obs, reward, done, info = env.step(action)

action = 0  # Flat
obs, reward, done, info = env.step(action)

action = 2  # Short
obs, reward, done, info = env.step(action)
```

## Compatibilité avec les Algorithmes

L'action space discret est compatible avec :

- ✅ **DQN** (Deep Q-Network)
- ✅ **PPO** (version discrète)
- ✅ **A2C/A3C** (avec politique catégorielle)
- ⚠️ **SAC** - Non compatible directement (conçu pour actions continues)

## Comportement Détaillé

Lorsque tu appelles `env.step(action)` :

1. **Action 0 (Flat)** :
   - Si tu as une position ouverte → elle est fermée
   - Reste à plat (aucune nouvelle position)

2. **Action 1 (Long)** :
   - Si tu as une position short → elle est fermée
   - Ouvre une position longue avec sizing basé sur le risque (0.05% du capital)

3. **Action 2 (Short)** :
   - Si tu as une position long → elle est fermée
   - Ouvre une position courte avec sizing basé sur le risque (0.05% du capital)

## Position Sizing

Le sizing des positions est automatique et basé sur :
- Risque : 0.05% du capital par trade
- Stop-Loss : 3× ATR
- Take-Profit : 6× ATR
- Leverage max : 2×

**Tu n'as pas à gérer le sizing** - l'environnement le fait automatiquement en fonction de l'action choisie (0, 1, ou 2).

## Exemple Complet

```python
from backend.trading_env import TradingEnvironment
from backend.data_pipeline import DataPipeline
from backend.feature_engineering import FeaturePipeline

# Charger les données
data_pipeline = DataPipeline()
train_data, _, _ = data_pipeline.get_processed_data()

feature_pipeline = FeaturePipeline()
train_features, _, _ = feature_pipeline.run_full_pipeline(
    train_data, train_data, train_data
)

# Créer l'environnement
env = TradingEnvironment(
    data=train_data['EURUSD'],
    features=train_features
)

# Test simple
obs = env.reset()
print(f"Action space: {env.action_space}")  # Discrete(3)

for i in range(10):
    action = env.action_space.sample()  # 0, 1, ou 2
    obs, reward, done, info = env.step(action)

    print(f"Step {i}: Action={action}, Position={info['position']:.4f}, "
          f"Equity=${info['equity']:.2f}")

    if done:
        break
```

## Questions Fréquentes

**Q: Puis-je avoir des positions partielles ?**
R: Non, avec l'action space discret tu as 3 choix : flat (0), long complet (1), ou short complet (2).

**Q: Comment fermer une position ?**
R: Utilise l'action 0 (flat) pour fermer toutes les positions.

**Q: Que se passe-t-il si j'envoie action=1 alors que j'ai déjà un long ?**
R: L'environnement détecte que la position est déjà correcte et ne fait rien (évite les frais de transaction inutiles).

**Q: Les coûts de transaction sont-ils appliqués ?**
R: Oui, chaque changement de position (open/close) a un coût réaliste (spread + slippage + market impact).
