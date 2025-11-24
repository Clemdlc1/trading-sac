# Modification : Action Space Discret

## Résumé des changements

L'action space de l'environnement de trading a été modifié de **continu** à **discret**.

### Avant
- **Type** : `spaces.Box(low=-1.0, high=1.0, shape=(1,))`
- **Actions** : Valeurs continues entre -1.0 et 1.0
- **Signification** :
  - -1.0 = Position courte maximale
  - 0.0 = Pas de position (flat)
  - 1.0 = Position longue maximale

### Après
- **Type** : `spaces.Discrete(3)`
- **Actions** : Valeurs discrètes {0, 1, 2}
- **Signification** :
  - 0 = Pas de position (flat) → converti en 0.0
  - 1 = Position longue → converti en 1.0
  - 2 = Position courte → converti en -1.0

## Fichiers modifiés

### 1. `backend/trading_env.py`

**Changements principaux :**
- `TradingEnvConfig` : Remplacé `action_min` et `action_max` par `n_actions = 3`
- `TradingEnvironment.__init__()` : Action space changé de `Box` à `Discrete(3)`
- Nouvelle méthode `_convert_discrete_action()` : Convertit les actions discrètes (0, 1, 2) en valeurs continues (0.0, 1.0, -1.0)
- `step()` : Utilise maintenant `_convert_discrete_action()` pour convertir l'action avant le traitement

**Mapping des actions :**
```python
def _convert_discrete_action(self, action: int) -> float:
    action_mapping = {
        0: 0.0,   # Flat (pas de position)
        1: 1.0,   # Long (position longue)
        2: -1.0   # Short (position courte)
    }
    return action_mapping.get(int(action), 0.0)
```

### 2. `backend/action_discretizer_wrapper.py` (NOUVEAU)

Wrapper créé pour permettre aux agents à actions continues (comme SAC) de fonctionner avec l'environnement à actions discrètes.

**Fonctionnalité :**
- Présente un action space `Box(low=-1, high=1)` à l'agent
- Convertit automatiquement les actions continues en discrètes
- Règle de discrétisation avec seuils :
  - `action < -0.33` → 2 (short)
  - `-0.33 ≤ action ≤ 0.33` → 0 (flat)
  - `action > 0.33` → 1 (long)

## Utilisation

### Option 1 : Environnement discret pur

Pour utiliser l'environnement avec des actions discrètes (par exemple, avec DQN) :

```python
from backend.trading_env import TradingEnvironment

env = TradingEnvironment(data=train_data, features=train_features)
# env.action_space est maintenant Discrete(3)

obs = env.reset()
action = env.action_space.sample()  # Retourne 0, 1, ou 2
obs, reward, done, info = env.step(action)
```

### Option 2 : Wrapper pour agents continus (SAC)

Pour continuer à utiliser l'agent SAC avec l'environnement discret :

```python
from backend.trading_env import TradingEnvironment
from backend.action_discretizer_wrapper import ActionDiscretizerWrapper

base_env = TradingEnvironment(data=train_data, features=train_features)
env = ActionDiscretizerWrapper(base_env)
# env.action_space est maintenant Box(low=-1, high=1, shape=(1,))

# L'agent SAC peut maintenant fonctionner normalement
obs = env.reset()
action = agent.select_action(obs)  # Retourne une action continue
obs, reward, done, info = env.step(action)  # Le wrapper la convertit en discrète
```

## Impact sur l'entraînement

### Avantages
- **Simplicité** : Seulement 3 actions possibles au lieu d'un espace infini
- **Stabilité** : Moins de variance dans les actions
- **Interprétabilité** : Actions clairement définies (flat, long, short)

### Inconvénients potentiels
- **Moins de flexibilité** : Impossible d'avoir des positions partielles
- **Pas de scaling graduel** : Passage brutal entre états

## Algorithmes recommandés

Avec l'action space discret, les algorithmes suivants sont recommandés :

1. **DQN (Deep Q-Network)** - Optimal pour actions discrètes
2. **PPO (Proximal Policy Optimization)** - Version discrète
3. **A2C/A3C** - Avec politique catégorielle
4. **SAC avec wrapper** - En utilisant `ActionDiscretizerWrapper`

## Tests

Un script de test est disponible : `test_discrete_actions.py`

```bash
python test_discrete_actions.py
```

Ce script teste :
- La création de l'environnement avec action space discret
- Le mapping des actions (0, 1, 2) → (0.0, 1.0, -1.0)
- L'échantillonnage d'actions aléatoires
- L'exécution de steps avec chaque action

## Compatibilité

- ✅ **Trading Environment** : Fully compatible
- ✅ **SAC Agent** : Compatible avec `ActionDiscretizerWrapper`
- ✅ **Data Pipeline** : Aucun changement nécessaire
- ✅ **Feature Engineering** : Aucun changement nécessaire
- ⚠️ **Validation/Evaluation** : Peut nécessiter des ajustements si utilisation directe de l'environnement

## Questions fréquentes

**Q: Pourquoi 3 actions seulement ?**
R: Les 3 actions représentent les décisions fondamentales de trading : ne rien faire (flat), acheter (long), ou vendre (short).

**Q: Puis-je toujours utiliser SAC ?**
R: Oui, en utilisant le `ActionDiscretizerWrapper` qui convertit les actions continues de SAC en actions discrètes.

**Q: Comment changer les seuils de discrétisation ?**
R: Modifiez la méthode `_discretize_action()` dans `action_discretizer_wrapper.py`.

**Q: Les performances seront-elles impactées ?**
R: Potentiellement. L'action space discret peut être plus facile à apprendre mais moins flexible. Des tests comparatifs sont recommandés.
