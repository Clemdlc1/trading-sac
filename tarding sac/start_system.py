#!/usr/bin/env python
"""
Script de démarrage pour l'application SAC EUR/USD.

Il vérifie l'environnement, rappelle l'installation des dépendances
et lance l'interface Flask via `python -m backend.web_app`.
"""

import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"


def ensure_pythonpath() -> None:
    """S'assure que le dossier racine est dans PYTHONPATH."""
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [p for p in pythonpath.split(os.pathsep) if p]
    if str(ROOT_DIR) not in parts:
        parts.insert(0, str(ROOT_DIR))
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def ensure_runtime_dirs() -> None:
    """Crée les dossiers nécessaires avant lancement."""
    required = [
        ROOT_DIR / "logs",
        ROOT_DIR / "config",
        ROOT_DIR / "reports",
        ROOT_DIR / "models" / "checkpoints",
        ROOT_DIR / "models" / "production",
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=== SAC Trading System Starter ===")
    print("1) Installez les dépendances: pip install -r requirement.txt")
    print("2) Lancement de l'application web...\n")

    ensure_pythonpath()
    ensure_runtime_dirs()

    cmd = [sys.executable, "-m", "backend.web_app"]
    try:
        subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)
    except subprocess.CalledProcessError as exc:
        print("\n[ERREUR] Le serveur ne s'est pas lancé correctement.")
        print(f"Code retour: {exc.returncode}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

