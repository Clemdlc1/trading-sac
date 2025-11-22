# Système de Métriques d'Entraînement (Style TensorBoard)

## 📊 Vue d'ensemble

Le système de trading SAC inclut maintenant un système complet de tracking de métriques, similaire à TensorBoard, qui enregistre **tous les épisodes depuis l'épisode 0** sans aucune limitation.

## 🎯 Métriques Trackées

### 📈 Récompenses
- `episode_rewards`: Récompense brute par épisode
- `episode_rewards_mean`: Moyenne mobile (100 derniers épisodes)
- `episode_rewards_std`: Écart-type de la moyenne mobile

### 🎓 Losses (Entraînement)
- `critic_losses`: Loss du critique à chaque épisode
- `actor_losses`: Loss de l'acteur à chaque épisode
- `alpha_losses`: Loss du paramètre alpha (température)

### 🔧 Paramètres SAC
- `alpha_values`: Valeur du coefficient d'entropie (température)
- `target_entropies`: Entropie cible (adaptive)
- `actor_lr`: Learning rate de l'acteur
- `critic_lr`: Learning rate du critique

### 💾 Replay Buffer
- `buffer_sizes`: Taille du buffer à chaque épisode
- `buffer_winning_ratio`: Ratio de transitions gagnantes
- `buffer_losing_ratio`: Ratio de transitions perdantes
- `buffer_neutral_ratio`: Ratio de transitions neutres

### 🎲 Exploration
- `action_mean`: Moyenne des actions (exploration vs exploitation)
- `action_std`: Écart-type des actions (niveau d'exploration)

### 💰 Performance Trading
- `sharpe_ratios`: Ratio de Sharpe
- `sortino_ratios`: Ratio de Sortino
- `win_rates`: Taux de victoire
- `max_drawdowns`: Drawdown maximum
- `total_returns`: Retour total
- `final_equities`: Équité finale
- `profit_factors`: Facteur de profit
- `total_trades`: Nombre total de trades
- `winning_trades`: Nombre de trades gagnants
- `losing_trades`: Nombre de trades perdants

### 📊 Informations Générales
- `episodes`: Numéro d'épisode
- `timestamps`: Timestamp de chaque épisode
- `episode_steps`: Nombre de steps par épisode
- `total_steps`: Nombre total de steps

## 📁 Fichiers Générés

### Métriques JSON
```
logs/training_metrics.json
```
Fichier JSON contenant **toutes** les métriques depuis l'épisode 0. Ce fichier est:
- Sauvegardé tous les 10 épisodes pendant l'entraînement
- Sauvegardé à la fin de l'entraînement
- Chargeable pour continuer un entraînement

### CSVs de Transitions
```
logs/training_csvs/training_ep{episode}_agent{agent_id}.csv
```
Fichiers CSV contenant les transitions détaillées pour chaque épisode de checkpoint (tous les 100 épisodes).

## 🖥️ Utilisation

### 1. Visualisation des Métriques

Le script `visualize_metrics.py` génère automatiquement **tous les graphiques** à partir des métriques sauvegardées:

```bash
# Utilisation basique (utilise les chemins par défaut)
python visualize_metrics.py

# Avec chemins personnalisés
python visualize_metrics.py \
    --metrics-file logs/training_metrics.json \
    --output-dir reports/plots
```

#### Graphiques Générés

1. **rewards.png**: Récompenses avec moyenne mobile et bande de confiance
2. **losses.png**: Évolution des 3 losses (critic, actor, alpha)
3. **sac_params.png**: Alpha, target entropy, learning rates, buffer size
4. **exploration.png**: Mean et std des actions (exploration)
5. **performance.png**: 6 sous-graphiques de performance (Sharpe, Sortino, Win Rate, etc.)
6. **buffer_composition.png**: Composition du replay buffer (stacked area)
7. **trades_evolution.png**: Évolution du nombre de trades
8. **equity_curve.png**: Courbe d'équité finale
9. **training_summary.txt**: Statistiques résumées en texte

### 2. API REST pour Récupérer les Métriques

Endpoint pour récupérer les métriques via l'API web:

```bash
GET /api/training/metrics
```

Réponse:
```json
{
  "success": true,
  "num_episodes": 1000,
  "metrics": {
    "episodes": [1, 2, 3, ...],
    "episode_rewards": [10.5, 12.3, ...],
    ...
  },
  "available_metrics": ["episodes", "episode_rewards", ...],
  "file_path": "logs/training_metrics.json",
  "timestamp": "2025-11-22T10:30:00"
}
```

### 3. WebSocket (Temps Réel)

Pendant l'entraînement, toutes les métriques sont émises en temps réel via SocketIO:

```javascript
socket.on('training_progress', (data) => {
    console.log('Episode:', data.episode);
    console.log('Reward:', data.reward);
    console.log('Critic Loss:', data.critic_loss);
    console.log('Alpha:', data.alpha);
    // ... toutes les métriques disponibles

    // Historique complet depuis l'épisode 0
    console.log('Historique:', data.metrics_history);
});
```

## 🔄 Continuation d'Entraînement

Les métriques sont automatiquement chargées si vous continuez un entraînement:

```python
# Le fichier logs/training_metrics.json est automatiquement chargé
# et l'historique continue depuis le dernier épisode
```

## 📊 Exemple d'Analyse

### Analyse de Convergence

```python
import json
import numpy as np

# Charger les métriques
with open('logs/training_metrics.json', 'r') as f:
    metrics = json.load(f)

# Analyser la convergence
rewards = np.array(metrics['episode_rewards'])
window = 100

# Moyenne mobile pour voir la tendance
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

# Détection de plateau (convergence)
variance = np.var(moving_avg[-100:])
print(f"Variance des 100 derniers épisodes: {variance}")

# Analyser l'exploration
action_std = np.array(metrics['action_std'])
print(f"Std des actions - Début: {action_std[0]:.4f}, Fin: {action_std[-1]:.4f}")
```

## 📈 Comparaison avec TensorBoard

| Fonctionnalité | TensorBoard | Notre Système |
|----------------|-------------|---------------|
| Tracking de losses | ✅ | ✅ |
| Métriques custom | ✅ | ✅ |
| Graphiques interactifs | ✅ | ⚠️ (matplotlib statique) |
| Pas de limitation d'épisodes | ✅ | ✅ |
| Sauvegarde JSON | ❌ | ✅ |
| WebSocket temps réel | ❌ | ✅ |
| API REST | ❌ | ✅ |

## 🎓 Métriques Avancées

### Détection de Surapprentissage

Comparez les métriques sur les premiers et derniers épisodes:

```python
# Si action_std diminue trop, l'agent explore moins (possible surapprentissage)
early_std = np.mean(metrics['action_std'][:100])
late_std = np.mean(metrics['action_std'][-100:])

if late_std < 0.5 * early_std:
    print("⚠️ Warning: Exploration a beaucoup diminué")
```

### Analyse du Learning Rate Decay

```python
import matplotlib.pyplot as plt

plt.plot(metrics['episodes'], metrics['actor_lr'], label='Actor LR')
plt.plot(metrics['episodes'], metrics['critic_lr'], label='Critic LR')
plt.yscale('log')
plt.legend()
plt.show()
```

## 🐛 Dépannage

### Métriques Non Sauvegardées

Si `logs/training_metrics.json` n'existe pas:
1. Vérifiez que l'entraînement a duré au moins 10 épisodes
2. Vérifiez les permissions du dossier `logs/`
3. Consultez les logs pour voir les erreurs de sauvegarde

### Graphiques Vides

Si `visualize_metrics.py` génère des graphiques vides:
1. Vérifiez que le fichier JSON contient des données
2. Installez les dépendances: `pip install matplotlib seaborn numpy`

## 📝 Notes

- **Pas de limitation**: Tous les épisodes sont conservés (pas de limite à 100 ou 200)
- **Performance**: La sauvegarde JSON est optimisée (tous les 10 épisodes)
- **Compatibilité**: Les métriques sont au format JSON standard (facile à analyser)
- **Extensibilité**: Vous pouvez ajouter vos propres métriques dans `web_app.py`

## 🎯 Prochaines Améliorations

- [ ] Dashboard interactif avec Plotly Dash
- [ ] Export vers TensorBoard natif
- [ ] Comparaison multi-agents
- [ ] Alertes automatiques sur métriques anormales
- [ ] Integration avec Weights & Biases (wandb)
