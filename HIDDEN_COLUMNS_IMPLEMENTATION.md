# Hidden Columns Implementation

## Résumé

Cette implémentation ajoute **2 colonnes cachées** au dataset qui ne sont **PAS visibles par l'agent** mais utilisées pour des calculs précis :

1. **`raw_close`** - Prix non normalisé (précis) pour les calculs de PnL
2. **`timestamp`** - Timestamp exact pour les calculs temporels

## Pourquoi ?

### Problème
Les calculs d'equity et PnL étaient effectués sur des données normalisées (z-score), ce qui réduit la précision :
- Les prix normalisés perdent l'échelle réelle
- Les calculs de profit/perte en dollars deviennent approximatifs
- Les métriques de trading (Sharpe, Sortino, etc.) sont moins précises

### Solution
Stocker le **prix réel non normalisé** (`raw_close`) séparément :
- L'agent continue de voir uniquement les 30 features normalisées
- L'environnement utilise `raw_close` pour les calculs de PnL en dollars
- Aucune fuite d'information vers l'agent

## Modifications apportées

### 1. `data_pipeline.py`

**Lignes 888-889, 900-901, 912-913** : Ajout de `raw_close` lors de la sauvegarde
```python
# Hidden columns for precise calculations (not features)
pair_grp.create_dataset('raw_close', data=df['close'].values)  # Non-normalized price
```

**Lignes 963-967, 980-984, 997-1001** : Chargement de `raw_close` avec compatibilité arrière
```python
# Load raw_close if available (for backward compatibility)
if 'raw_close' in pair_grp:
    df_dict['raw_close'] = pair_grp['raw_close'][:]
else:
    df_dict['raw_close'] = pair_grp['close'][:]  # Fallback
```

### 2. `feature_engineering.py`

**Lignes 685-689** : Ajout de paramètres pour `raw_close`
```python
train_raw_close: pd.Series,
val_raw_close: pd.Series,
test_raw_close: pd.Series,
```

**Lignes 754-770** : Sauvegarde des colonnes cachées dans groupe `/hidden/`
```python
# NEW: Save hidden columns (not visible to agent)
hidden_grp = f.create_group('hidden')

# Training hidden columns
hidden_train_grp = hidden_grp.create_group('train')
hidden_train_grp.create_dataset('raw_close', data=train_raw_close.values)
hidden_train_grp.create_dataset('timestamp', data=train_timestamps.astype('int64') // 10**9)
```

**Lignes 855-867** : Chargement des colonnes cachées
```python
# Load hidden columns if available (for backward compatibility)
if 'hidden' in f:
    # Training hidden columns
    if 'hidden/train/raw_close' in f:
        train_features['raw_close'] = f['hidden/train/raw_close'][:]
    # ... (same for val/test)
```

**Lignes 935-937** : Passage de `raw_close` à la fonction de sauvegarde
```python
train_data['EURUSD']['raw_close'],
val_data['EURUSD']['raw_close'],
test_data['EURUSD']['raw_close'],
```

### 3. `trading_env.py`

**Lignes 542-554** : Extraction et suppression des colonnes cachées
```python
# Extract hidden columns for precise calculations
# These are NOT visible to the agent (not in observation space)
if 'raw_close' in self.features.columns:
    self.raw_close = self.features['raw_close'].values
    logger.info("Using raw_close for precise PnL calculations")
else:
    # Fallback to normalized close if raw_close not available
    self.raw_close = self.data['close'].values
    logger.warning("raw_close not found, using normalized close (less precise)")

# Remove hidden columns from features (agent must not see them)
hidden_columns = ['raw_close', 'timestamp']
self.features = self.features.drop(columns=[col for col in hidden_columns if col in self.features.columns])
```

**Ligne 680** : Utilisation de `raw_close` pour les calculs de prix
```python
# IMPORTANT: Use raw_close for precise PnL calculations (non-normalized)
current_price = self.raw_close[idx]
```

**Lignes 871-888** : Documentation de la fonction `_get_observation()`
```python
"""
Get current observation (features).

IMPORTANT: Returns only the 30 normalized features.
Hidden columns (raw_close, timestamp) are NOT included in observations.
"""
```

## Structure HDF5

### Avant
```
/train/<pair>/
  - timestamp
  - open, high, low, close

/features/
  - train/val/test/ (30 features normalisées)
```

### Après
```
/train/<pair>/
  - timestamp
  - open, high, low, close
  - raw_close  ← NOUVEAU (non normalisé)

/features/
  - train/val/test/ (30 features normalisées)

/hidden/  ← NOUVEAU (invisible à l'agent)
  - train/val/test/
    - raw_close  (prix non normalisé)
    - timestamp  (temps exact)
```

## Garanties de sécurité

### ✅ L'agent ne voit PAS les colonnes cachées
1. `raw_close` et `timestamp` sont **supprimés** de `env.features` dans `__init__`
2. L'observation space reste **exactement 30 features**
3. `_get_observation()` retourne uniquement `env.features` (sans colonnes cachées)

### ✅ Compatibilité arrière
- Si `raw_close` n'existe pas dans les anciens fichiers, fallback sur `close`
- Pas de breakage des datasets existants

### ✅ Précision des calculs
- Tous les calculs de PnL utilisent maintenant `self.raw_close[idx]`
- Les prix sont en dollars réels (ex: 1.0856 EUR/USD)
- Les calculs d'equity, Sharpe ratio, Sortino sont plus précis

## Test de validation

Un script de test complet est fourni : `test_hidden_columns.py`

### Tests effectués
1. **Data Pipeline** - Vérification que `raw_close` est sauvegardé
2. **Feature Engineering** - Vérification que les colonnes cachées sont préservées
3. **Trading Environment** - Vérification que `raw_close` est utilisé pour les PnL
4. **Observation Space** - Vérification qu'aucune fuite d'information

### Exécution des tests
```bash
cd "tarding sac/backend"
python test_hidden_columns.py
```

## Impact sur l'apprentissage

### Avant (avec normalisation)
```python
# Prix normalisé (z-score)
current_price = 0.234  # Aucune signification en dollars

# PnL approximatif
pnl = position * 100000 * (0.234 - 0.230)  # Erreur!
```

### Après (avec raw_close)
```python
# Prix réel en dollars
current_price = 1.08567  # EUR/USD réel

# PnL précis en dollars
pnl = position * 100000 * (1.08567 - 1.08534)  # Correct!
```

## Métriques améliorées

Avec `raw_close`, les métriques suivantes sont maintenant **précises** :

- **Equity curve** - En dollars réels
- **PnL par trade** - En dollars réels
- **Transaction costs** - Calculés sur prix réels
- **Sharpe Ratio** - Basé sur returns réels
- **Sortino Ratio** - Basé sur downside réel
- **Max Drawdown** - En dollars/pourcentage réels
- **Profit Factor** - Ratio de gains/pertes réels
- **Win Rate** - Basé sur PnL réels

## Notes importantes

1. **Les features restent normalisées** - L'agent continue de voir des features normalisées (z-score) pour faciliter l'apprentissage
2. **Seuls les calculs internes changent** - Les calculs d'equity/PnL utilisent maintenant les prix réels
3. **Pas d'information supplémentaire à l'agent** - L'observation space reste identique (30 features)
4. **Backward compatible** - Les anciens datasets fonctionnent avec fallback

## Export CSV

Les données, features **et logs d'entraînement** sont maintenant **automatiquement exportés en CSV** avec les colonnes cachées :

### Data Pipeline CSV
```
data/processed/csv/
  ├── train/
  │   ├── EURUSD.csv
  │   ├── USDJPY.csv
  │   └── ... (toutes les paires)
  ├── val/
  │   └── ... (mêmes paires)
  └── test/
      └── ... (mêmes paires)
```

**Colonnes dans chaque CSV :**
- `timestamp` - Date et heure exacte (datetime)
- `open`, `high`, `low`, `close` - Prix OHLC
- `raw_close` - Prix non normalisé (identique à close)

### Features CSV
```
data/normalized/csv/
  ├── train_features.csv
  ├── val_features.csv
  └── test_features.csv
```

**Colonnes dans chaque CSV :**
- 30 features normalisées (return_5min, rsi_14, etc.)
- `timestamp` - Date et heure exacte
- `raw_close` - Prix non normalisé pour PnL

### Training Logs CSV (web_app)
```
logs/training_csv/
  ├── training_ep5_agent0.csv
  ├── training_ep10_agent0.csv
  └── ... (sauvegardé tous les 5 épisodes)
```

**Colonnes dans chaque CSV :**
- Colonnes existantes (episode, agent_id, step, action, reward, done, equity, position, etc.)
- **30 observations** (obs_0, obs_1, ..., obs_29) - Features normalisées
- `raw_close` - **NOUVEAU** - Prix non normalisé (ex: 1.08567 EUR/USD)
- `timestamp` - **NOUVEAU** - Datetime exact du step (ISO format)

### Utilité des CSV
- ✅ Inspection manuelle des données
- ✅ Analyse avec Excel/Python externe
- ✅ Debugging facile (lisible)
- ✅ Vérification des calculs de PnL
- ✅ Visualisation des prix réels vs normalisés
- ✅ **Analyse des décisions de trading avec prix réels**
- ✅ **Corrélation actions/conditions de marché**
- ✅ **Validation PnL calculé vs prix réels**

## Prochaines étapes recommandées

1. **Régénérer les datasets** - Lancer le pipeline pour créer les nouveaux fichiers HDF5 et CSV avec `raw_close`
2. **Inspecter les CSV** - Vérifier manuellement que `raw_close` et `timestamp` sont présents
3. **Vérifier les tests** - Exécuter `test_hidden_columns.py` pour valider l'implémentation
4. **Comparer les métriques** - Entraîner un agent et comparer les métriques avant/après
5. **Valider le PnL** - Vérifier que les calculs de PnL correspondent aux prix réels du marché

## Conclusion

Cette implémentation améliore significativement la **précision des calculs** tout en maintenant l'**intégrité de l'apprentissage** :
- ✅ Calculs de PnL précis en dollars réels
- ✅ Métriques de trading fiables
- ✅ Aucune fuite d'information vers l'agent
- ✅ Features normalisées pour l'apprentissage
- ✅ Compatibilité arrière assurée
