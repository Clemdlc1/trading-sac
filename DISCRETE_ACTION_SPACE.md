# Action Space Discret - Documentation

## Résumé

L'environnement de trading utilise maintenant un **action space discret** avec 3 actions possibles.

## Actions Disponibles

L'environnement accepte **deux types** d'actions :

### 1. Actions Discrètes (recommandé)

| Action | Valeur | Comportement |
|--------|--------|--------------|
| **0** | `0.0` | **Flat** - Ferme toutes les positions et reste à plat |
| **1** | `1.0` | **Long** - Ouvre une position longue (et ferme short si existant) |
| **2** | `-1.0` | **Short** - Ouvre une position courte (et ferme long si existant) |

### 2. Actions Continues (pour compatibilité SAC)

L'environnement accepte aussi des actions continues `[-1, 1]` qui sont automatiquement discrétisées :

| Valeur Continue | Action Discrète | Comportement |
|----------------|----------------|--------------|
| `< -0.33` | 2 (Short) | Ouvre position courte |
| `[-0.33, 0.33]` | 0 (Flat) | Ferme tout |
| `> 0.33` | 1 (Long) | Ouvre position longue |

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

### Avec Actions Discrètes

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

### Avec SAC (Actions Continues)

```python
from backend.trading_env import TradingEnvironment
from backend.sac_agent import SACAgent, SACConfig

# Créer l'environnement
env = TradingEnvironment(data=train_data, features=train_features)

# Créer l'agent SAC
config = SACConfig(state_dim=30, action_dim=1)
agent = SACAgent(config=config)

# Entraînement
obs = env.reset()
action = agent.select_action(obs)  # Génère action continue comme [0.7]
obs, reward, done, info = env.step(action)  # Automatiquement converti en action 1 (Long)
```

**Comment ça marche** : SAC génère une action continue (ex: `[0.7]`), l'environnement la convertit automatiquement en action discrète (ex: `1` pour Long), puis exécute l'action.

## Compatibilité avec les Algorithmes

L'action space discret est compatible avec :

- ✅ **DQN** (Deep Q-Network) - Actions discrètes natives
- ✅ **PPO** (version discrète) - Actions discrètes natives
- ✅ **A2C/A3C** (avec politique catégorielle) - Actions discrètes natives
- ✅ **SAC** - Compatible ! Les actions continues sont automatiquement discrétisées

**Note importante pour SAC** : L'agent SAC génère des actions continues `[-1, 1]` qui sont automatiquement converties en actions discrètes par l'environnement. Aucune modification de l'agent n'est nécessaire !

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

**Q: Est-ce que SAC fonctionne avec l'action space discret ?**
R: Oui ! L'environnement accepte automatiquement les actions continues de SAC et les convertit en actions discrètes. SAC qui génère `[0.8]` sera converti en action `1` (Long).

**Q: Comment fonctionne la conversion continue → discret ?**
R: Les seuils sont :
- `< -0.33` → Short (action 2)
- `[-0.33, 0.33]` → Flat (action 0)
- `> 0.33` → Long (action 1)

**Q: Puis-je changer les seuils de discrétisation ?**
R: Oui, modifie la méthode `step()` dans `trading_env.py` (lignes 681-686 et 693-698).
