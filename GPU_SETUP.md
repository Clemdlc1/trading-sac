# Configuration GPU pour l'Entraînement Accéléré

## Vérification Actuelle

Le système détecte automatiquement si CUDA est disponible. Vous verrez dans les logs :
- ✅ **GPU Accéléré** : Message vert avec le nom du GPU
- ⚠️ **Mode CPU** : Message d'avertissement jaune

## Installation PyTorch avec Support GPU

### Prérequis
1. **GPU NVIDIA compatible CUDA** (GTX 10xx, RTX 20xx, RTX 30xx, RTX 40xx, etc.)
2. **Drivers NVIDIA à jour** : https://www.nvidia.com/download/index.aspx
3. **CUDA Toolkit** (optionnel, PyTorch l'inclut)

### Installation Rapide

#### Windows / Linux / macOS

```bash
# Désinstaller PyTorch actuel (si installé)
pip uninstall torch torchvision torchaudio

# Installer PyTorch avec CUDA 11.8 (recommandé)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OU avec CUDA 12.1 (pour GPUs plus récents)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Vérification de l'Installation

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Sortie attendue avec GPU :**
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU Name: NVIDIA GeForce RTX 3080
GPU Count: 1
GPU Memory: 10.00 GB
```

## Gains de Performance

### Temps d'Entraînement Estimés (1000 épisodes)

| Configuration | Temps estimé | Accélération |
|---------------|--------------|--------------|
| CPU (Intel i7) | ~48 heures | 1x |
| GPU RTX 3060 | ~4-6 heures | 8-12x |
| GPU RTX 3080 | ~2-3 heures | 16-24x |
| GPU RTX 4090 | ~1-1.5 heures | 32-48x |

### Optimisations Automatiques

Le code utilise automatiquement :
- ✅ **Mixed Precision** (si disponible)
- ✅ **CUDA Graphs** (optimisations automatiques)
- ✅ **Pinned Memory** (transferts CPU→GPU plus rapides)
- ✅ **cuDNN** (convolutions optimisées)

## Dépannage

### Erreur : "CUDA out of memory"

**Solution 1** : Réduire le batch size
```python
# Dans l'interface, réduire de 256 → 128 ou 64
batch_size = 128
```

**Solution 2** : Libérer la mémoire GPU
```bash
# Windows
taskkill /F /IM python.exe

# Linux
pkill -9 python
```

### Erreur : "CUDA driver version is insufficient"

**Solution** : Mettre à jour les drivers NVIDIA
- Windows : https://www.nvidia.com/download/index.aspx
- Linux : `sudo apt update && sudo apt install nvidia-driver-535`

### Le GPU n'est pas détecté

**Vérifications** :
```bash
# Vérifier que le GPU est visible
nvidia-smi

# Vérifier la version CUDA
nvcc --version
```

**Si nvidia-smi fonctionne mais PyTorch ne voit pas CUDA** :
- Réinstaller PyTorch avec la bonne version CUDA
- Vérifier que les drivers correspondent à la version CUDA de PyTorch

## Support Multi-GPU

Le code supporte automatiquement plusieurs GPUs. Le premier GPU (cuda:0) sera utilisé par défaut.

Pour utiliser un GPU spécifique :
```python
# Dans sac_agent.py, changer :
device = torch.device("cuda:0")  # GPU 0
device = torch.device("cuda:1")  # GPU 1
```

## Monitoring GPU Pendant l'Entraînement

### Windows
```bash
# Ouvrir un terminal et exécuter :
nvidia-smi -l 1
```

### Linux
```bash
watch -n 1 nvidia-smi
```

Vous verrez :
- **GPU Utilization** : devrait être ~95-100% pendant l'entraînement
- **Memory Usage** : allocation mémoire GPU
- **Temperature** : température du GPU (idéalement <85°C)
- **Power Draw** : consommation électrique

## Recommandations

✅ **GPU Recommandés pour le Trading SAC** :
- Budget : RTX 3060 (12 GB VRAM) - ~350€
- Performance : RTX 3080 (10-12 GB) - ~700€
- Optimal : RTX 4090 (24 GB) - ~2000€

⚠️ **Minimum requis** :
- 6 GB VRAM minimum
- Compute Capability 6.0+ (GTX 10xx et plus récent)

## Support

Si vous rencontrez des problèmes, vérifiez :
1. Les logs dans `logs/web_app.log` (affichent le device détecté)
2. L'interface web (badge vert = GPU, badge jaune = CPU)
3. La console système lors du démarrage de l'entraînement

Le système fonctionne sur **CPU et GPU**, mais le GPU est **fortement recommandé** pour l'entraînement de réseaux de neurones profonds.
