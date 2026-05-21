import random
import pandas as pd
import matplotlib.pyplot as plt

def st_petersburg_simulation(n_games):
    results = []
    for _ in range(n_games):
        flips = 1
        while random.choice(['H', 'T']) != 'T':
            flips += 1
        payout = 2 ** flips
        results.append(payout)
    average_win = sum(results) / len(results)
    max_win = max(results)
    return average_win, max_win, results

# Run simulations for different game counts
sim_results = {}
game_counts = [100, 10_000, 1_000_000]

for num_games in game_counts:
    avg, max_payout, all_results = st_petersburg_simulation(num_games)
    sim_results[num_games] = {
        "average_win": avg,
        "max_win": max_payout,
        "all_results": all_results
    }

# Create summary table for terminal display
summary_df = pd.DataFrame({
    "Games Played": game_counts,
    "Average Win": [sim_results[n]["average_win"] for n in game_counts],
    "Max Win": [sim_results[n]["max_win"] for n in game_counts]
})

print("St. Petersburg Simulation Summary:")
print(summary_df)
