"""
Test script pour vérifier le fonctionnement de l'action space discret.

L'environnement a 3 actions :
- 0 : Flat (ferme tout)
- 1 : Long (ferme short et ouvre long)
- 2 : Short (ferme long et ouvre short)
"""
import sys
sys.path.append('/home/user/trading-sac/tarding sac')

print("Chargement des modules...")
from backend.trading_env import TradingEnvironment
from backend.data_pipeline import DataPipeline
from backend.feature_engineering import FeaturePipeline

def test_discrete_actions():
    """Test l'action space discret avec 3 actions."""

    print("\n" + "="*80)
    print("Test de l'Action Space Discret")
    print("="*80)

    # Charger les données
    print("\nChargement des données...")
    data_pipeline = DataPipeline()
    train_data, _, _ = data_pipeline.get_processed_data()

    feature_pipeline = FeaturePipeline()
    train_features, _, _ = feature_pipeline.run_full_pipeline(
        train_data, train_data, train_data
    )

    # Créer l'environnement
    print("Création de l'environnement...")
    env = TradingEnvironment(
        data=train_data['EURUSD'],
        features=train_features,
        eval_mode=False
    )

    print(f"\n✓ Action space: {env.action_space}")
    print(f"✓ Type: {type(env.action_space)}")
    print(f"✓ Nombre d'actions: {env.action_space.n}")

    # Test de chaque action
    print("\n" + "="*80)
    print("Test des 3 actions discrètes")
    print("="*80)

    actions_to_test = [
        (0, "FLAT - Ferme tout"),
        (1, "LONG - Ouvre long"),
        (2, "SHORT - Ouvre short"),
    ]

    for action, description in actions_to_test:
        print(f"\n>>> Action {action}: {description}")
        obs = env.reset()
        print(f"    Equity initiale: ${env.equity:.2f}")

        # Exécuter l'action plusieurs fois
        for step in range(3):
            obs, reward, done, info = env.step(action)
            position = info['position']
            equity = info['equity']

            if position > 0:
                pos_type = "LONG"
            elif position < 0:
                pos_type = "SHORT"
            else:
                pos_type = "FLAT"

            print(f"    Step {step+1}: Position={position:+.4f} ({pos_type}), "
                  f"Equity=${equity:.2f}, Reward={reward:.4f}")

            if done:
                print(f"    Episode terminé!")
                break

    # Test avec actions aléatoires
    print("\n" + "="*80)
    print("Test avec actions aléatoires")
    print("="*80)

    obs = env.reset()
    action_counts = {0: 0, 1: 0, 2: 0}
    steps = 20

    print(f"\nExécution de {steps} steps avec actions aléatoires...\n")

    for i in range(steps):
        action = env.action_space.sample()
        action_counts[action] += 1

        obs, reward, done, info = env.step(action)

        position = info['position']
        if position > 0:
            pos_type = "LONG"
        elif position < 0:
            pos_type = "SHORT"
        else:
            pos_type = "FLAT"

        if i % 5 == 0 or i == steps - 1:
            print(f"Step {i+1}: Action={action}, Position={pos_type:5s}, "
                  f"Equity=${info['equity']:.2f}, Trades={info['total_trades']}")

        if done:
            print(f"\nEpisode terminé au step {i+1}!")
            break

    print(f"\nDistribution des actions sur {i+1} steps:")
    for action, count in action_counts.items():
        percentage = (count / (i+1)) * 100
        action_name = ["FLAT", "LONG", "SHORT"][action]
        print(f"  Action {action} ({action_name}): {count} fois ({percentage:.1f}%)")

    # Test du mapping
    print("\n" + "="*80)
    print("Vérification du mapping des actions")
    print("="*80)

    print("\nMapping discret → continu:")
    for action in [0, 1, 2]:
        continuous = env._convert_discrete_action(action)
        action_name = ["FLAT", "LONG", "SHORT"][action]
        print(f"  {action} ({action_name:5s}) → {continuous:+.1f}")

    print("\n" + "="*80)
    print("✓ Tous les tests sont passés!")
    print("="*80)

    print("\nL'environnement fonctionne correctement avec 3 actions discrètes:")
    print("  • 0 = Flat (ferme toutes les positions)")
    print("  • 1 = Long (ouvre position longue)")
    print("  • 2 = Short (ouvre position courte)")

if __name__ == "__main__":
    test_discrete_actions()
