import os, numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from Player import make_move, get_valid_moves, is_winning_state

ROWS, COLS = 6, 7

def count_windows(board, player, length=4):
    rows, cols = board.shape
    total_any = 0
    counts_by_fill = {0:0, 1:0, 2:0, 3:0, 4:0}

    def consider(window):
        nonlocal total_any
        if all(v in (0, player) for v in window):
            total_any += 1
            k = sum(1 for v in window if v == player)
            counts_by_fill[k] += 1

    # horizontal
    for r in range(rows):
        for c in range(cols - length + 1):
            consider(list(board[r, c:c+length]))
    # vertical
    for c in range(cols):
        col_vals = list(board[:, c])
        for r in range(rows - length + 1):
            consider(col_vals[r:r+length])
    # diagonal \
    for r in range(rows - length + 1):
        for c in range(cols - length + 1):
            consider([board[r+i, c+i] for i in range(length)])
    # diagonal /
    for r in range(length - 1, rows):
        for c in range(cols - length + 1):
            consider([board[r-i, c+i] for i in range(length)])
    return total_any, counts_by_fill

def immediate_wins(board, player):
    wins = 0
    for c in get_valid_moves(board):
        b2 = np.copy(board)
        make_move(b2, c, player)
        if is_winning_state(b2, player):
            wins += 1
    return wins

def center_counts(board, player):
    center_col = board.shape[1] // 2
    return list(board[:, center_col]).count(player)

def gravity_height(board, col):
    for r in range(board.shape[0]):
        if board[r, col] != 0:
            return r
    return board.shape[0]

def potential_forks(board, player):
    forks = 0
    for c in get_valid_moves(board):
        b2 = np.copy(board)
        make_move(b2, c, player)
        if immediate_wins(b2, player) >= 2:
            forks += 1
    return forks

def build_features(board, to_move):
    me = to_move
    opp = 1 if to_move == 2 else 2
    me_center = center_counts(board, me)
    opp_center = center_counts(board, opp)
    me_mob = len(get_valid_moves(board))
    opp_mob = me_mob

    me_clean4, me_cnts = count_windows(board, me, 4)
    opp_clean4, opp_cnts = count_windows(board, opp, 4)

    me_immediate = immediate_wins(board, me)
    opp_immediate = immediate_wins(board, opp)

    me_forks = potential_forks(board, me)
    opp_forks = potential_forks(board, opp)

    center_height = gravity_height(board, board.shape[1]//2)
    empties = int(np.count_nonzero(board == 0))

    feats = [
        float(me_center), float(opp_center),
        float(me_mob), float(opp_mob),
        float(me_clean4),
        float(me_cnts[4]), float(me_cnts[3]), float(me_cnts[2]), float(me_cnts[1]), float(me_cnts[0]),
        float(opp_clean4),
        float(opp_cnts[4]), float(opp_cnts[3]), float(opp_cnts[2]), float(opp_cnts[1]), float(opp_cnts[0]),
        float(me_immediate), float(opp_immediate),
        float(me_forks), float(opp_forks),
        float(center_height), float(empties),
    ]
    return np.array(feats, dtype=float)

def rollout_value(board, to_move, K=5):
    """
    Returns (label, ev):
      label ∈ {-1,0,1} for classifiers
      ev ∈ [-1,1]      for regressors (expected value of outcome for to_move)
    """
    wins = draws = losses = 0
    for _ in range(K):
        b2 = np.copy(board)
        player = to_move
        while True:
            moves = get_valid_moves(b2)
            if not moves:
                draws += 1
                break
            m = np.random.choice(moves)
            make_move(b2, m, player)
            if is_winning_state(b2, player):
                if player == to_move:
                    wins += 1
                else:
                    losses += 1
                break
            player = 1 if player == 2 else 2

    # continuous expected value in [-1,1] ranges
    ev = (wins - losses) / max(1, K)

    # convert EV to a discrete label with a small deadzone
    if ev > 0.2:
        label = 1
    elif ev < -0.2:
        label = -1
    else:
        label = 0

    return label, ev

def sample_states(n_samples=20000, max_plies=20):
    X, y_cls, y_reg = [], [], []
    for _ in range(n_samples):
        b = np.zeros((ROWS, COLS), dtype=int)
        to_move = 1
        plies = np.random.randint(4, max_plies + 1)
        terminal = False
        for _t in range(plies):
            moves = get_valid_moves(b)
            if not moves:
                terminal = True
                break
            m = np.random.choice(moves)
            make_move(b, m, to_move)
            if is_winning_state(b, to_move):
                terminal = True
                break
            to_move = 1 if to_move == 2 else 2

        if terminal:
            continue  # skip terminal for training eval of non-terminal states

        feats = build_features(b, to_move)
        label, ev = rollout_value(b, to_move, K=5)
        X.append(feats)
        y_cls.append(int(label))   # {-1,0,1} for NBC/DT
        y_reg.append(float(ev))    # [-1,1]   for LR/NN

    return np.array(X), np.array(y_cls), np.array(y_reg, dtype=float)

def main():
    np.random.seed(0)
    os.makedirs("models", exist_ok=True)

    print("Sampling dataset...")
    X, y_cls, y_reg = sample_states(n_samples=20000, max_plies=20)
    print("Samples:", X.shape[0], "Class distribution:", Counter(y_cls))

    # Split
    Xtr, Xte, ytr_c, yte_c = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    _,  Xte2, ytr_r, yte_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # Naive Bayes (classifier)
    nb = GaussianNB()
    nb.fit(Xtr, ytr_c)
    nb_pred = nb.predict(Xte)
    nb_acc = accuracy_score(yte_c, nb_pred)
    print(f"Naive Bayes acc: {nb_acc:.3f}")
    joblib.dump(nb, "models/nbc.pkl")

    # Decision Tree (classifier)
    dt = DecisionTreeClassifier(max_depth=8, random_state=0)
    dt.fit(Xtr, ytr_c)
    dt_pred = dt.predict(Xte)
    dt_acc = accuracy_score(yte_c, dt_pred)
    print(f"Decision Tree acc: {dt_acc:.3f}")
    joblib.dump(dt, "models/dt.pkl")

    # Linear Regression (regressor)
    lr = LinearRegression()
    lr.fit(Xtr, ytr_r)
    lr_pred = lr.predict(Xte2)
    lr_mse = mean_squared_error(yte_r, lr_pred)
    print(f"Linear Regression MSE: {lr_mse:.4f}")
    joblib.dump(lr, "models/lr.pkl")

    # Optional MLP (regressor)
    nn = MLPRegressor(hidden_layer_sizes=(64,64), activation="relu", max_iter=200, random_state=0)
    nn.fit(Xtr, ytr_r)
    nn_pred = nn.predict(Xte2)
    nn_mse = mean_squared_error(yte_r, nn_pred)
    print(f"MLP Regressor MSE: {nn_mse:.4f}")
    joblib.dump(nn, "models/nn.pkl")

    print("Saved models in ./models")

if __name__ == "__main__":
    main()
