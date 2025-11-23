# Script pour créer un fichier h5 combiné (OHLC + Features)
# À exécuter en LOCAL avant d'uploader sur Kaggle

import h5py
import pandas as pd
import numpy as np
from pathlib import Path

def merge_h5_files(
    processed_data_path: str,
    features_path: str,
    output_path: str = "combined_data.h5"
):
    """
    Merge processed_data.h5 et features_normalized.h5 en un seul fichier.

    Args:
        processed_data_path: Chemin vers processed_data.h5
        features_path: Chemin vers features_normalized.h5
        output_path: Chemin de sortie pour le fichier combiné
    """
    print(f"📊 Fusion des fichiers h5...")
    print(f"   Input 1: {processed_data_path}")
    print(f"   Input 2: {features_path}")
    print(f"   Output: {output_path}")

    with h5py.File(output_path, 'w') as f_out:
        # Charger processed_data.h5
        with h5py.File(processed_data_path, 'r') as f_data:
            # Copier train/EURUSD
            print("\n📋 Copie des données OHLC...")

            for split in ['train', 'val', 'test']:
                print(f"   {split}...")
                split_grp = f_out.create_group(split)
                eurusd_grp = split_grp.create_group('EURUSD')

                # Copier OHLC
                for field in ['timestamp', 'open', 'high', 'low', 'close']:
                    data = f_data[f'/{split}/EURUSD/{field}'][:]
                    eurusd_grp.create_dataset(field, data=data)

        # Charger features_normalized.h5
        with h5py.File(features_path, 'r') as f_feat:
            print("\n📊 Copie des features...")

            for split in ['train', 'val', 'test']:
                print(f"   {split}...")
                features_grp = f_out[split].create_group('features')

                # Récupérer les noms de features
                feature_names = [name for name in f_feat[f'/features/{split}'].keys()]

                # Copier chaque feature
                for feat_name in feature_names:
                    data = f_feat[f'/features/{split}/{feat_name}'][:]
                    features_grp.create_dataset(feat_name, data=data)

                print(f"      {len(feature_names)} features copiées")

    print(f"\n✅ Fichier combiné créé: {output_path}")

    # Vérifier la structure
    print("\n🔍 Vérification de la structure:")
    with h5py.File(output_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"   {name}: shape={obj.shape}")

        f.visititems(print_structure)

    print("\n✅ Terminé! Uploadez ce fichier sur Kaggle.")
    return output_path


if __name__ == "__main__":
    # À adapter selon vos chemins locaux
    merge_h5_files(
        processed_data_path="data/processed/processed_data.h5",
        features_path="data/normalized/features_normalized.h5",
        output_path="combined_data.h5"
    )
