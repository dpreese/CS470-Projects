import random
import matplotlib.pyplot as plt

def simulate_monty_hall(n_games=1000):
    stay_wins = 0
    switch_wins = 0

    for _ in range(n_games):
        doors = [0, 1, 2]
        car = random.choice(doors)
        initial_choice = random.choice(doors)

        # Host reveals a goat door that is not the car or the initial choice
        remaining_doors = [d for d in doors if d != initial_choice and d != car]
        door_opened = random.choice(remaining_doors)

        # The other unopened door
        switch_choice = next(d for d in doors if d != initial_choice and d != door_opened)

        # Win conditions
        if initial_choice == car:
            stay_wins += 1
        if switch_choice == car:
            switch_wins += 1

    stay_pct = (stay_wins / n_games) * 100
    switch_pct = (switch_wins / n_games) * 100
    return stay_pct, switch_pct

# Run the simulation
stay_win_pct, switch_win_pct = simulate_monty_hall(1000)

print("Results for 1000 Monty Hall Games:")
print(f"Stay Strategy Win Percentage: {stay_win_pct:.2f}%")
print(f"Switch Strategy Win Percentage: {switch_win_pct:.2f}%")

# Plotting convergence
def plot_convergence(max_games=10000, step=500):
    stay_results = []
    switch_results = []
    game_counts = list(range(step, max_games + 1, step))

    for n in game_counts:
        stay_pct, switch_pct = simulate_monty_hall(n)
        stay_results.append(stay_pct)
        switch_results.append(switch_pct)

    plt.figure(figsize=(10, 6))
    plt.plot(game_counts, stay_results, label="Stay Strategy", color="red")
    plt.plot(game_counts, switch_results, label="Switch Strategy", color="green")
    plt.axhline(y=33.33, linestyle='--', color='red', alpha=0.5)
    plt.axhline(y=66.67, linestyle='--', color='green', alpha=0.5)
    plt.xlabel("Number of Games Simulated")
    plt.ylabel("Win Percentage")
    plt.title("Monty Hall Simulation: Strategy Convergence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_convergence()
