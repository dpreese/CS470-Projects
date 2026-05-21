import random
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def simulate_single_battle_round(na, nd):
    attacker_dice = sorted([random.randint(1, 6) for _ in range(na)], reverse=True)
    defender_dice = sorted([random.randint(1, 6) for _ in range(nd)], reverse=True)

    a_losses = 0
    d_losses = 0
    for a, d in zip(attacker_dice, defender_dice):
        if a > d:
            d_losses += 1
        else:
            a_losses += 1
    return a_losses, d_losses

def part1_single_round_probabilities(trials=100000):
    outcomes = []
    for na in [1, 2, 3]:
        for nd in [1, 2]:
            result_counter = Counter()
            for _ in range(trials):
                outcome = simulate_single_battle_round(na, nd)
                result_counter[outcome] += 1
            total = sum(result_counter.values())
            for (a_loss, d_loss), count in sorted(result_counter.items()):
                outcomes.append({
                    "Attacker Dice": na,
                    "Defender Dice": nd,
                    "Attacker Losses": a_loss,
                    "Defender Losses": d_loss,
                    "Probability": round(count / total, 4)
                })
    df = pd.DataFrame(outcomes)
    print("\nPART 1: Single-Round Battle Outcome Probabilities")
    print(df.to_string(index=False))
    return df

def simulate_battle_round(attacker_armies, defender_armies):
    na = min(3, attacker_armies - 1)
    nd = min(2, defender_armies)

    attacker_dice = sorted([random.randint(1, 6) for _ in range(na)], reverse=True)
    defender_dice = sorted([random.randint(1, 6) for _ in range(nd)], reverse=True)

    a_losses = 0
    d_losses = 0
    for a, d in zip(attacker_dice, defender_dice):
        if a > d:
            d_losses += 1
        else:
            a_losses += 1
    return a_losses, d_losses

def simulate_full_battle(attacker, defender):
    while attacker > 1 and defender > 0:
        a_loss, d_loss = simulate_battle_round(attacker, defender)
        attacker -= a_loss
        defender -= d_loss
    return attacker, defender

def part2_attacker_win_probability(defender=5, trials=10000):
    results = []
    for attacker_armies in range(2, 21):
        wins = 0
        for _ in range(trials):
            final_a, final_d = simulate_full_battle(attacker_armies, defender)
            if final_d == 0:
                wins += 1
        win_rate = wins / trials
        results.append((attacker_armies, win_rate))

    x_vals, y_vals = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='50% Threshold')
    plt.axhline(y=0.8, color='red', linestyle='--', label='80% Threshold')
    plt.title("Probability Attacker Wins vs. Starting Armies (Defender = 5)")
    plt.xlabel("Attacker Starting Armies")
    plt.ylabel("Probability of Winning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    threshold_50 = next((a for a, p in results if p >= 0.5), None)
    threshold_80 = next((a for a, p in results if p >= 0.8), None)
    print(f"\nPART 2: Minimum attacker armies needed for ≥50% win: {threshold_50}")
    print(f"PART 2: Minimum attacker armies needed for ≥80% win: {threshold_80}")

def part3_end_state_distribution(attacker_start=10, defender_start=10, trials=10000):
    outcomes = defaultdict(int)
    for _ in range(trials):
        a, d = simulate_full_battle(attacker_start, defender_start)
        outcomes[(a, d)] += 1

    total = sum(outcomes.values())
    result_list = []
    for (a, d), count in sorted(outcomes.items(), key=lambda x: (-x[0][0], x[0][1])):
        result_list.append({
            "Attacker Remaining": a,
            "Defender Remaining": d,
            "Probability": round(count / total, 4)
        })

    df = pd.DataFrame(result_list)
    print("\nPART 3: End-State Probability Distribution (10 vs 10)")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    part1_single_round_probabilities()
    part2_attacker_win_probability()
    part3_end_state_distribution()
