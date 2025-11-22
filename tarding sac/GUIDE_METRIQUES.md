# 📊 Guide Complet des Métriques d'Entraînement SAC

## Vue d'Ensemble

Ce guide explique **toutes les métriques** trackées pendant l'entraînement de l'agent SAC (Soft Actor-Critic) pour le trading. Chaque métrique est accompagnée de :
- 🎯 **À quoi ça sert**
- 📈 **Comment ça évolue normalement**
- ⚠️ **Signaux d'alerte**
- 💡 **Comment l'interpréter**

---

## 📊 1. Métriques de Performance de Trading

### 📈 Episode Reward
**Ce que c'est :**
- La récompense totale accumulée pendant un épisode
- Somme de toutes les récompenses reçues à chaque step
- Peut être positive (profit) ou négative (perte)

**Comment ça doit évoluer :**
- ✅ **Début** : Très volatile, souvent négatif (l'agent explore au hasard)
- ✅ **Milieu** : Commence à augmenter progressivement
- ✅ **Fin** : Se stabilise autour d'une valeur positive (idéalement)

**Interprétation :**
- 📈 Tendance haussière = l'agent apprend à gagner de l'argent
- 📉 Tendance baissière après une montée = surapprentissage possible
- 📊 Haute volatilité = forte exploration (normal au début)
- 📊 Faible volatilité + valeur haute = bon apprentissage

**Signaux d'alerte :**
- 🔴 Reste négatif après 500+ épisodes
- 🔴 Chute soudaine après une période de stabilité
- 🔴 Oscille violemment sans converger

---

### 📊 Sharpe Ratio
**Ce que c'est :**
- Mesure le rendement ajusté au risque
- Formule : `(Rendement moyen - Rendement sans risque) / Écart-type des rendements`
- Plus il est élevé, meilleur est le ratio rendement/risque

**Comment ça doit évoluer :**
- ✅ **Début** : Souvent négatif ou proche de 0
- ✅ **Milieu** : Augmente progressivement vers des valeurs positives
- ✅ **Cible** : > 1.0 est bon, > 2.0 est excellent

**Interprétation :**
- Sharpe > 0 : Stratégie profitable avec risque contrôlé
- Sharpe > 1 : Bon ratio rendement/risque
- Sharpe > 2 : Excellent ratio rendement/risque
- Sharpe > 3 : Exceptionnel (rare)

**Signaux d'alerte :**
- 🔴 Reste < 0 après 1000 épisodes
- 🟡 Oscille autour de 0 sans monter
- 🟢 Augmente régulièrement = bon signe !

---

### 📊 Sortino Ratio
**Ce que c'est :**
- Variante du Sharpe qui ne pénalise QUE la volatilité à la baisse
- Mesure le rendement par rapport au risque de perte
- Formule : `Rendement / Écart-type des rendements négatifs`

**Comment ça doit évoluer :**
- ✅ Généralement plus élevé que le Sharpe
- ✅ Augmente avec le Sharpe
- ✅ Cible : > 1.5 est bon, > 3.0 est excellent

**Interprétation :**
- Sortino > Sharpe = L'agent a plus de gains que de pertes
- Sortino >> Sharpe = Excellente asymétrie (gros gains, petites pertes)

**Utilité :**
- Plus pertinent que Sharpe pour le trading
- Montre si l'agent "coupe ses pertes et laisse courir ses gains"

---

### 🎯 Win Rate (Taux de Victoire)
**Ce que c'est :**
- Pourcentage de trades gagnants
- Formule : `Nombre de trades gagnants / Nombre total de trades`
- Valeur entre 0% et 100%

**Comment ça doit évoluer :**
- ✅ **Début** : Autour de 50% (aléatoire)
- ✅ **Milieu** : Augmente progressivement
- ✅ **Cible** : 55-65% est bon pour le trading

**Interprétation :**
- 50% = Trading aléatoire (pièce de monnaie)
- 55-60% = Bon edge statistique
- 65-70% = Excellent edge
- > 70% = Suspicieux (possible surapprentissage)

**Note importante :**
- ⚠️ Un Win Rate élevé n'est PAS toujours bon !
- Un agent avec 40% de Win Rate PEUT être profitable si ses gains > ses pertes
- **Regardez toujours le Win Rate AVEC le Profit Factor**

---

### 📉 Max Drawdown
**Ce que c'est :**
- Perte maximale depuis le pic d'équité
- Mesure le "pire moment" en termes de perte
- Formule : `(Pic - Creux) / Pic`

**Comment ça doit évoluer :**
- ✅ **Début** : Très élevé (50-90%)
- ✅ **Milieu** : Diminue progressivement
- ✅ **Cible** : < 20% est bon, < 10% est excellent

**Interprétation :**
- < 10% = Risque très faible
- 10-20% = Risque acceptable
- 20-30% = Risque élevé
- > 30% = Risque très élevé

**Signaux d'alerte :**
- 🔴 Augmente soudainement après stabilisation
- 🔴 Reste > 40% après 1000 épisodes
- 🟢 Diminue régulièrement = bon contrôle du risque

---

### 💰 Profit Factor
**Ce que c'est :**
- Ratio entre les gains totaux et les pertes totales
- Formule : `Somme des gains / Somme des pertes`
- Mesure la qualité globale de la stratégie

**Comment ça doit évoluer :**
- ✅ **Début** : Autour de 1.0 (gains = pertes)
- ✅ **Milieu** : Augmente progressivement
- ✅ **Cible** : > 1.5 est bon, > 2.0 est excellent

**Interprétation :**
- 1.0 = Break-even (gains = pertes)
- 1.5 = Pour chaque €1 perdu, on gagne €1.50
- 2.0 = Pour chaque €1 perdu, on gagne €2.00
- > 3.0 = Stratégie exceptionnelle (ou surapprentissage)

**Signaux d'alerte :**
- 🔴 Reste < 1.0 = stratégie perdante
- 🟡 Oscille autour de 1.0 = pas d'edge
- 🟢 > 1.5 stable = bonne stratégie

---

### 📊 Total Return
**Ce que c'est :**
- Rendement total en pourcentage
- Formule : `(Équité finale - Équité initiale) / Équité initiale`

**Comment ça doit évoluer :**
- ✅ **Début** : Souvent négatif (-20% à -50%)
- ✅ **Milieu** : Remonte vers 0% puis positif
- ✅ **Fin** : Positif et croissant

**Interprétation :**
- Montre la performance globale de l'épisode
- À combiner avec le Sharpe (rendement ajusté au risque)

---

### 💵 Final Equity (Équité Finale)
**Ce que c'est :**
- Capital final à la fin de l'épisode
- Capital initial = 100,000$ (par défaut)

**Comment ça doit évoluer :**
- ✅ **Début** : Souvent < capital initial
- ✅ **Milieu** : Augmente progressivement
- ✅ **Fin** : > capital initial (profit)

**Interprétation :**
- > 100,000$ = Épisode profitable
- < 100,000$ = Épisode perdant
- Tendance croissante = apprentissage positif

---

### 📊 Nombre de Trades
**Ce que c'est :**
- Nombre total de trades exécutés pendant l'épisode
- Se divise en : Total, Winning (gagnants), Losing (perdants)

**Comment ça doit évoluer :**
- ✅ **Début** : Beaucoup de trades (sur-trading)
- ✅ **Milieu** : Diminue (l'agent devient plus sélectif)
- ✅ **Fin** : Se stabilise à un niveau optimal

**Interprétation :**
- Trop de trades = Sur-trading (coûts de transaction élevés)
- Trop peu de trades = Agent trop prudent (opportunités manquées)
- **Regarder la qualité (Win Rate, Profit Factor) plus que la quantité**

---

## 🎓 2. Métriques d'Entraînement (Losses)

### 📉 Critic Loss
**Ce que c'est :**
- Erreur du réseau critique (Q-function)
- Mesure la précision des estimations de valeur Q
- Formule : MSE entre Q prédit et Q cible

**Comment ça doit évoluer :**
- ✅ **Début** : Très élevée (>100)
- ✅ **Milieu** : Décroît rapidement
- ✅ **Fin** : Se stabilise à un niveau bas (<10)

**Interprétation :**
- Décroissance = Le critique apprend à prédire les valeurs
- Oscillations normales = données non-stationnaires (normal en RL)
- Trop basse trop vite = possible surapprentissage

**Signaux d'alerte :**
- 🔴 Reste très élevée (>50) après 500 épisodes
- 🔴 Augmente soudainement après avoir diminué
- 🔴 Tombe à quasi 0 = surapprentissage sur le replay buffer
- 🟢 Diminue régulièrement puis se stabilise = bon signe

---

### 📉 Actor Loss
**Ce que c'est :**
- Erreur du réseau acteur (politique)
- Mesure à quel point la politique maximise la valeur Q
- Négative (car on maximise, pas minimise)

**Comment ça doit évoluer :**
- ✅ **Début** : Très négative (ex: -20)
- ✅ **Milieu** : Devient moins négative
- ✅ **Fin** : Se stabilise (ex: -5 à -10)

**Interprétation :**
- Plus négative = La politique trouve des actions de haute valeur
- Moins négative = La politique est plus conservatrice
- **À analyser avec le Critic Loss**

**Signaux d'alerte :**
- 🔴 Oscille violemment sans stabilisation
- 🔴 Augmente fortement (devient moins négative) soudainement
- 🟢 Se stabilise autour d'une valeur = convergence

---

### 📉 Alpha Loss
**Ce que c'est :**
- Erreur du paramètre d'entropie (température)
- Contrôle l'équilibre exploration/exploitation
- Ajuste automatiquement l'exploration

**Comment ça doit évoluer :**
- ✅ **Début** : Oscille pour trouver bon équilibre
- ✅ **Milieu** : Se stabilise
- ✅ **Fin** : Faible et stable (proche de 0)

**Interprétation :**
- Stable = Bon équilibre exploration/exploitation trouvé
- Oscille beaucoup = Recherche de l'équilibre optimal

**Note :**
- Métrique moins critique que Critic/Actor Loss
- Sert surtout à vérifier que le mécanisme d'auto-tuning fonctionne

---

## 🔧 3. Paramètres SAC

### 🌡️ Alpha (Temperature)
**Ce que c'est :**
- Coefficient d'entropie qui contrôle l'exploration
- Plus alpha est élevé, plus l'agent explore
- Plus alpha est bas, plus l'agent exploite

**Comment ça doit évoluer :**
- ✅ **Début** : Élevé (~0.2-0.5) pour explorer
- ✅ **Milieu** : Diminue progressivement (adaptive entropy)
- ✅ **Fin** : Bas (~0.05-0.1) pour exploiter

**Interprétation :**
- Alpha élevé = Actions plus aléatoires (exploration)
- Alpha bas = Actions plus déterministes (exploitation)
- **Décroissance normale** = Bon passage exploration → exploitation

**Signaux d'alerte :**
- 🔴 Reste très élevé (>0.5) après 1000 épisodes = trop d'exploration
- 🔴 Tombe trop vite à 0 = risque de convergence prématurée
- 🟢 Décroît lentement et se stabilise = excellent !

**Lien avec la performance :**
- Si rewards montent ALORS QUE alpha baisse = **excellent signe**
- L'agent trouve de meilleures stratégies tout en devenant plus certain

---

### 📈 Target Entropy
**Ce que c'est :**
- Entropie cible pour l'auto-tuning de alpha
- Valeur par défaut : -1.0 (= -dim(action))
- **Adaptive** : Décroît de -1.0 à -0.5 pendant l'entraînement

**Comment ça doit évoluer :**
- ✅ Décroît lentement de -1.0 vers -0.5
- ✅ Force l'agent à explorer moins avec le temps

**Interprétation :**
- -1.0 = Maximum d'exploration autorisée
- -0.5 = Moins d'exploration (exploitation)
- **Cette décroissance est programmée** (pas apprise)

---

### 📉 Actor Learning Rate
**Ce que c'est :**
- Taux d'apprentissage du réseau acteur
- Contrôle la vitesse de mise à jour des poids
- Valeur initiale : 3e-4 (0.0003)

**Comment ça doit évoluer :**
- ✅ **Début** : 3e-4
- ✅ **Milieu** : Décroît progressivement (LR decay)
- ✅ **Fin** : ~1e-5 (minimum)

**Interprétation :**
- LR élevé = Apprentissage rapide mais instable
- LR bas = Apprentissage lent mais stable
- **Décroissance** = On affine progressivement la politique

**Pourquoi décroître :**
- Au début : Besoin de changements rapides
- À la fin : Besoin de fine-tuning subtil

**Signaux d'alerte :**
- 🔴 Décroît trop vite = risque de "gel" de l'apprentissage
- 🟢 Décroissance exponentielle régulière = normal

---

### 📉 Critic Learning Rate
**Ce que c'est :**
- Taux d'apprentissage du réseau critique
- Même principe que Actor LR
- Valeur initiale : 3e-4

**Comment ça doit évoluer :**
- ✅ Identique à Actor LR (décroissance synchronisée)

**Interprétation :**
- Même logique que Actor LR
- Important qu'il décroisse en parallèle avec Actor LR

---

## 🎲 4. Métriques d'Exploration

### 📊 Action Mean (Moyenne des Actions)
**Ce que c'est :**
- Moyenne des actions prises pendant l'épisode
- Actions entre -1 et +1 (tanh squashing)
- -1 = Full short, 0 = Neutre, +1 = Full long

**Comment ça doit évoluer :**
- ✅ **Début** : Proche de 0 (aléatoire)
- ✅ **Milieu** : Peut diverger de 0 (bias directionnel)
- ✅ **Fin** : Dépend de la stratégie apprise

**Interprétation :**
- Mean ~0 = Stratégie équilibrée (long et short)
- Mean >0 = Biais long (plus de positions acheteuses)
- Mean <0 = Biais short (plus de positions vendeuses)

**Ce qui est normal :**
- Pour EUR/USD, souvent un léger biais long (Mean >0)
- L'important est que ça soit **cohérent** avec la performance

**Signaux d'alerte :**
- 🔴 Oscille violemment = instabilité
- 🟢 Se stabilise autour d'une valeur = stratégie convergée

---

### 📊 Action Std (Écart-type des Actions)
**Ce que c'est :**
- Écart-type des actions (mesure de la dispersion)
- **Métrique clé d'exploration**
- Élevé = Explore, Bas = Exploite

**Comment ça doit évoluer :**
- ✅ **Début** : Élevé (~0.8-1.0) = forte exploration
- ✅ **Milieu** : Décroît progressivement
- ✅ **Fin** : Bas (~0.1-0.3) = forte exploitation

**Interprétation :**
- Std élevé = Agent essaie beaucoup d'actions différentes
- Std bas = Agent confiant dans ses actions

**Lien avec Alpha :**
- Devrait décroître en parallèle avec Alpha
- Si Alpha baisse MAIS Std reste élevé = problème

**Signaux d'alerte :**
- 🔴 Reste >0.8 après 1000 épisodes = sur-exploration
- 🔴 Tombe <0.05 trop vite = risque de local optimum
- 🟢 Décroît lentement de 0.8 à 0.2 = parfait !

**Indicateur de convergence :**
- Std faible + Rewards élevés + Sharpe >1 = **Convergence réussie**

---

## 💾 5. Métriques du Replay Buffer

### 📊 Buffer Size (Taille du Buffer)
**Ce que c'est :**
- Nombre de transitions stockées dans le replay buffer
- Capacité max : 100,000 transitions
- Se remplit progressivement

**Comment ça doit évoluer :**
- ✅ **Début** : 0 → 5,000 (warmup)
- ✅ **Milieu** : 5,000 → 100,000
- ✅ **Fin** : Plein (100,000)

**Interprétation :**
- < 5,000 = Phase de warmup (pas d'updates)
- 5,000-100,000 = Remplissage progressif
- 100,000 = Buffer plein (mode FIFO)

**Signaux importants :**
- À 5,000 : "🚀 REPLAY BUFFER READY - Starting model updates!"
- Les updates commencent SEULEMENT après 5,000 transitions

---

### 📊 Buffer Composition (Winning/Losing/Neutral Ratios)
**Ce que c'est :**
- Distribution des transitions dans le buffer
- **Winning** : Transitions avec reward >0.01
- **Losing** : Transitions avec reward <-0.01
- **Neutral** : Transitions avec |reward| <0.01

**Comment ça doit évoluer :**
- ✅ **Début** : Surtout neutral + losing
- ✅ **Milieu** : Winning augmente progressivement
- ✅ **Fin** : Plus de winning que losing

**Interprétation :**
- Ratio Winning/Losing élevé = Bonnes expériences dominantes
- Équilibré = Stratégie mixte (normal)

**Utilité :**
- Le buffer fait du **stratified sampling** :
  - 20% winning, 20% losing, 60% neutral
- Assure que l'agent apprend des bons ET mauvais trades

**Signaux d'alerte :**
- 🔴 Trop de losing (>50%) après 1000 épisodes
- 🟢 Winning augmente régulièrement = bon signe

---

## 📈 6. Métriques Contextuelles

### 📊 Episode Steps
**Ce que c'est :**
- Nombre de steps (décisions) dans l'épisode
- Dépend de la longueur des données

**Interprétation :**
- Varie selon l'épisode
- Pas une métrique d'apprentissage
- Utile pour diagnostiquer des episodes courts = terminaison prématurée

---

### 📊 Total Steps
**Ce que c'est :**
- Nombre total de steps depuis le début de l'entraînement
- Compteur cumulatif

**Utilité :**
- Déclenche les décroissances de LR (tous les 1000 steps)
- Suit la progression globale

---

## 🎯 Comment Interpréter Ensemble les Métriques

### ✅ Signes d'un Bon Entraînement

1. **Performance Trading :**
   - Rewards augmente ✅
   - Sharpe >1.0 ✅
   - Win Rate 55-65% ✅
   - Drawdown <20% ✅
   - Profit Factor >1.5 ✅

2. **Losses :**
   - Critic Loss décroît puis se stabilise ✅
   - Actor Loss se stabilise ✅
   - Alpha Loss stable ✅

3. **Exploration :**
   - Alpha décroît ✅
   - Action Std décroît ✅
   - Les deux décroissent EN MÊME TEMPS ✅

4. **Buffer :**
   - Se remplit progressivement ✅
   - Ratio Winning augmente ✅

### ⚠️ Signaux d'Alerte

| Symptôme | Cause Probable | Solution |
|----------|----------------|----------|
| Rewards négatifs après 1000 ep | Mauvais hyperparamètres | Ajuster gamma, LR |
| Critic Loss augmente soudain | Catastrophic forgetting | Réduire LR |
| Action Std reste élevé | Trop d'exploration | Réduire alpha |
| Sharpe oscille autour de 0 | Pas d'edge | Changer features |
| Win Rate >80% | Surapprentissage | Régulariser, diversifier données |
| Drawdown >40% | Trop de risque | Ajuster reward shaping |

---

## 📚 Résumé des Cibles

| Métrique | Début | Fin Cible | Excellent |
|----------|-------|-----------|-----------|
| **Episode Reward** | Négatif | Positif | >50 |
| **Sharpe Ratio** | <0 | >1.0 | >2.0 |
| **Win Rate** | ~50% | 55-60% | 60-65% |
| **Max Drawdown** | >50% | <20% | <10% |
| **Profit Factor** | ~1.0 | >1.5 | >2.0 |
| **Critic Loss** | >100 | <10 | <5 |
| **Alpha** | 0.2-0.5 | 0.05-0.1 | ~0.05 |
| **Action Std** | 0.8-1.0 | 0.1-0.3 | ~0.15 |

---

## 🎓 Concepts Avancés

### Relation Alpha ↔ Action Std ↔ Performance

**Scénario idéal :**
```
Épisode 0:     Alpha = 0.3, Std = 0.9, Reward = -20
Épisode 500:   Alpha = 0.15, Std = 0.5, Reward = 10
Épisode 1000:  Alpha = 0.08, Std = 0.2, Reward = 40
```

**Pattern :**
- Alpha décroît → Std décroît → Mais Rewards MONTE
- = L'agent devient plus certain ET meilleur

### Learning Rate Decay

**Pourquoi ça décroît :**
- Début : Grandes mises à jour = exploration rapide de l'espace
- Fin : Petites mises à jour = fine-tuning précis

**Courbe typique :**
```
LR = 3e-4 × 0.995^(steps/1000)
Steps 0:     3e-4
Steps 10k:   2e-4
Steps 50k:   1e-4
Steps 100k:  5e-5 (minimum)
```

---

## 💡 Conseils de Monitoring

### Pendant l'Entraînement

**À surveiller toutes les 100 épisodes :**
1. Tendance des Rewards (monte ?)
2. Sharpe Ratio (>0 ?)
3. Action Std (baisse ?)
4. Critic Loss (se stabilise ?)

**Si stagnation après 500 épisodes :**
1. Vérifier que le buffer est plein
2. Vérifier que les LR n'ont pas trop décru
3. Regarder la diversité du buffer
4. Vérifier Alpha (pas trop bas ?)

### Après l'Entraînement

**Analyse complète :**
1. Plotter TOUS les graphiques
2. Vérifier la convergence (courbes stables)
3. Regarder les derniers 100 épisodes (performances finales)
4. Comparer Sharpe vs Sortino (asymétrie)
5. Analyser Win Rate vs Profit Factor (qualité des trades)

---

## 📖 Glossaire Rapide

- **Exploration** : Essayer de nouvelles actions pour découvrir
- **Exploitation** : Utiliser les meilleures actions connues
- **Convergence** : Stabilisation des métriques (apprentissage terminé)
- **Overfitting** : Surapprentissage (bon sur train, mauvais sur test)
- **Catastrophic Forgetting** : L'agent "oublie" ce qu'il a appris
- **Replay Buffer** : Mémoire des expériences passées
- **Episode** : Une séquence complète de trading (du début à la fin des données)
- **Step** : Une décision de trading (une action)

---

## 🎯 Conclusion

**Les 3 métriques les plus importantes :**

1. **Episode Reward** : Est-ce que l'agent gagne de l'argent ?
2. **Sharpe Ratio** : Est-ce que c'est un bon risque/rendement ?
3. **Action Std** : Est-ce que l'agent converge (devient certain) ?

**Le signal ultime de réussite :**
```
Rewards ↗ + Sharpe >1.5 + Action Std ↘ = 🎉 Succès !
```

**En cas de doute :**
- Regarder les graphiques visuellement
- Chercher des tendances, pas des valeurs ponctuelles
- Comparer les 100 premiers vs 100 derniers épisodes
- La **stabilité** est souvent plus importante que les valeurs absolues
