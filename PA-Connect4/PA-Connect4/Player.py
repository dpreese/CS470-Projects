# Modified 10.3.2023 by Chris Archibald to
#  - incorporate MCTS with other code
#  - pass command line param string to each AI

import os
import numpy as np
try:
    import joblib
except Exception:
    joblib = None


class AIPlayer:
    def __init__(self, player_number, name, ptype, param, modelType):
        self._eval_cache = {}
        self.player_number = player_number
        self.name = name
        self.type = ptype
        self.modelType = modelType
        self.player_string = 'Player {}: '.format(player_number) + self.name
        self.other_player_number = 1 if player_number == 2 else 2

        # Parameters for the different agents
        self.depth_limit = 3  
        if self.type == 'ab' and param:
            self.depth_limit = int(param)

        if self.type == 'expmax' and param:
            self.depth_limit = int(param)

        self.max_iterations = 1000  
        if self.type == 'mcts' and param:
            self.max_iterations = int(param)

        # cache for loaded ML models
        self._model_cache = {}

    # -----------------------------
    # Public move selectors
    # -----------------------------
    def get_alpha_beta_move(self, board):
        """
        Alpha-Beta pruning (depth-limited). Returns best column for self.player_number.
        """
        def is_terminal(b):
            return (is_winning_state(b, self.player_number) or
                    is_winning_state(b, self.other_player_number) or
                    len(get_valid_moves(b)) == 0)

        def minimax(b, depth, alpha, beta, maximizing):
            valid_moves = get_valid_moves(b)
            if depth == 0 or is_terminal(b):
                return None, self.evaluation_function(b)

            if maximizing:
                value = -np.inf
                best_cols = []
                for col in valid_moves:
                    child = np.copy(b)
                    make_move(child, col, self.player_number)
                    _, child_val = minimax(child, depth - 1, alpha, beta, False)
                    if child_val > value:
                        value = child_val
                        best_cols = [col]
                    elif child_val == value:
                        best_cols.append(col)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                move = np.random.choice(best_cols) if best_cols else np.random.choice(valid_moves)
                return move, value
            else:
                value = np.inf
                best_cols = []
                for col in valid_moves:
                    child = np.copy(b)
                    make_move(child, col, self.other_player_number)
                    _, child_val = minimax(child, depth - 1, alpha, beta, True)
                    if child_val < value:
                        value = child_val
                        best_cols = [col]
                    elif child_val == value:
                        best_cols.append(col)
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                move = np.random.choice(best_cols) if best_cols else np.random.choice(valid_moves)
                return move, value

        best_move, _ = minimax(board, self.depth_limit, -np.inf, np.inf, True)
        if best_move is None:
            moves = get_valid_moves(board)
            best_move = np.random.choice(moves)
        return best_move

    def get_mcts_move(self, board):
        """
        Use MCTS to get the next move.
        """
        max_iterations = self.max_iterations
        root = MCTSNode(board, self.player_number, None)

        for _ in range(max_iterations):
            cur_node = root.select()
            cur_node.simulate()

        root.print_node()
        print('MCTS chooses action', root.max_child())
        return root.max_child()

    def get_expectimax_move(self, board):
        cache = {}

        def is_terminal(b):
            return (is_winning_state(b, self.player_number) or
                    is_winning_state(b, self.other_player_number) or
                    len(get_valid_moves(b)) == 0)

        def expectimax(b, depth, maximizing):
            key = (b.tobytes(), depth, maximizing)
            if key in cache:
                return cache[key]

            valid_moves = get_valid_moves(b)
            if depth == 0 or is_terminal(b):
                res = (None, self.evaluation_function(b))
                cache[key] = res
                return res

            if maximizing:
                best_val = -np.inf
                best_move = valid_moves[0]
                for col in valid_moves:
                    child = np.copy(b)
                    make_move(child, col, self.player_number)
                    _, v = expectimax(child, depth - 1, False)
                    if v > best_val:
                        best_val, best_move = v, col
                res = (best_move, best_val)
                cache[key] = res
                return res
            else:
                # Uniform chance over opponent's valid moves
                if not valid_moves:
                    res = (None, self.evaluation_function(b))
                    cache[key] = res
                    return res
                total = 0.0
                for col in valid_moves:
                    child = np.copy(b)
                    make_move(child, col, self.other_player_number)
                    _, v = expectimax(child, depth - 1, True)
                    total += v
                res = (None, total / len(valid_moves))
                cache[key] = res
                return res

        move, _ = expectimax(board, self.depth_limit, True)
        if move is None:
            moves = get_valid_moves(board)
            move = np.random.choice(moves)
        return move


    # -----------------------------
    # Evaluation (heuristic + ML)
    # -----------------------------
    def evaluation_function(self, board):
        """
        Return a scalar evaluation for the board from self.player_number's perspective.
        If self.modelType is one of the ML options, use that model; otherwise use heuristic.
        """
        # Store board so get_features can read it (starter signature lacked a board param)
        self._current_eval_board = board

        # cache by board + model type + my player id to improve run time as depths 3-5 are timing out atm
        key = (board.tobytes(), self.player_number, self.modelType)
        if key in self._eval_cache:
            return self._eval_cache[key]

        if self.modelType == "nbc":
            score = self.get_naive_bayes_evaluation(self.get_features(self.modelType), board)
        elif self.modelType == "dt":
            score = self.get_decision_tree_evaluation(self.get_features(self.modelType), board)
        elif self.modelType == "lr":
            score = self.get_linear_regression_evaluation(self.get_features(self.modelType), board)
        elif self.modelType == "nn":
            score = self.get_neural_network_evaluation(self.get_features(self.modelType), board)
        elif self.modelType == "diff":
            score = self.get_different_ML_model_evaluation(self.get_features(self.modelType), board)
        else:
            score = self._heuristic_eval(board)
        
        self._eval_cache[key] = score
        return score
    
    # Handcrafted heuristic split into its own helper so ML can fall back cleanly in case it fails
    # TO DO: Implement notification system in terminal with details for when this is triggered and why. Display stats and exception calls
    def _heuristic_eval(self, board):
        me = self.player_number
        opp = self.other_player_number

        if is_winning_state(board, me):
            return 1_000_000
        if is_winning_state(board, opp):
            return -1_000_000
        if len(get_valid_moves(board)) == 0:
            return 0

        def evaluate_window(window):
            score = 0
            me_count = window.count(me)
            opp_count = window.count(opp)
            empty = window.count(0)
            if me_count == 4:
                score += 10_000
            elif me_count == 3 and empty == 1:
                score += 100
            elif me_count == 2 and empty == 2:
                score += 10
            if opp_count == 3 and empty == 1:
                score -= 80
            if opp_count == 4:
                score -= 10_000
            return score

        def score_position(b):
            score = 0
            rows, cols = b.shape
            center_col = cols // 2
            center_array = list(b[:, center_col])
            score += center_array.count(me) * 6

            # Horizontal
            for r in range(rows):
                row_array = list(b[r, :])
                for c in range(cols - 3):
                    score += evaluate_window(row_array[c:c+4])

            # Vertical
            for c in range(cols):
                col_array = list(b[:, c])
                for r in range(rows - 3):
                    score += evaluate_window(col_array[r:r+4])

            # Diagonals
            for r in range(rows - 3):
                for c in range(cols - 3):
                    score += evaluate_window([b[r+i, c+i] for i in range(4)])
            for r in range(3, rows):
                for c in range(cols - 3):
                    score += evaluate_window([b[r-i, c+i] for i in range(4)])

            return score

        return score_position(board)

    # -----------------------------
    # Feature engineering for ML
    # -----------------------------
    def get_features(self, modelType):
        """
        Returns a 22-dim feature vector from the current player's perspective.
        (Uses the board stored in self._current_eval_board.)
        """
        b = getattr(self, "_current_eval_board", None)
        if b is None:
            return []

        me = self.player_number
        opp = self.other_player_number
        rows, cols = b.shape

        def count_windows(board, player, length=4):
            total_any = 0
            counts_by_fill = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

            def consider(window):
                nonlocal total_any
                if all(v in (0, player) for v in window):
                    total_any += 1
                    k = sum(1 for v in window if v == player)
                    counts_by_fill[k] += 1

            # horizontal
            for rr in range(rows):
                for cc in range(cols - length + 1):
                    consider(list(board[rr, cc:cc+length]))
            # vertical
            for cc in range(cols):
                col_vals = list(board[:, cc])
                for rr in range(rows - length + 1):
                    consider(col_vals[rr:rr+length])
            # diag \
            for rr in range(rows - length + 1):
                for cc in range(cols - length + 1):
                    consider([board[rr+i, cc+i] for i in range(length)])
            # diag /
            for rr in range(length - 1, rows):
                for cc in range(cols - length + 1):
                    consider([board[rr-i, cc+i] for i in range(length)])

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

        me_center = center_counts(b, me)
        opp_center = center_counts(b, opp)
        me_mob = float(len(get_valid_moves(b)))
        opp_mob = me_mob
        me_clean4, me_cnts = count_windows(b, me, 4)
        opp_clean4, opp_cnts = count_windows(b, opp, 4)

        me_immediate = immediate_wins(b, me)
        opp_immediate = immediate_wins(b, opp)

        me_forks = potential_forks(b, me)
        opp_forks = potential_forks(b, opp)

        center_height = gravity_height(b, cols // 2)
        empties = int(np.count_nonzero(b == 0))

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
        return np.array(feats, dtype=float).reshape(1, -1)

    # -----------------------------
    # ML evaluation helpers
    # -----------------------------
    def _load_model(self, path):
        if joblib is None:
            return None
        if path in self._model_cache:
            return self._model_cache[path]
        if not os.path.exists(path):
            return None
        model = joblib.load(path)
        self._model_cache[path] = model
        return model

    def _prob_margin(self, model, X):
        """
        For classifiers with classes in {-1,0,1}, return P(win)-P(loss).
        """
        if not hasattr(model, "predict_proba"):
            pred = model.predict(X)[0]
            try:
                return float(pred)
            except Exception:
                return 0.0
        proba = model.predict_proba(X)[0]
        cls_to_idx = {cls: i for i, cls in enumerate(model.classes_)}
        p_win = proba[cls_to_idx.get(1, 0)] if 1 in cls_to_idx else 0.0
        p_loss = proba[cls_to_idx.get(-1, 0)] if -1 in cls_to_idx else 0.0
        return float(p_win - p_loss)

    def get_naive_bayes_evaluation(self, features, board):
        model = self._load_model(os.path.join("models", "nbc.pkl"))
        if model is None:
            return self._heuristic_eval(board)
        X = self.get_features(self.modelType)  # 1x22
        return self._prob_margin(model, X)

    def get_decision_tree_evaluation(self, features, board):
        model = self._load_model(os.path.join("models", "dt.pkl"))
        if model is None:
            return self._heuristic_eval(board)
        X = self.get_features(self.modelType)
        return self._prob_margin(model, X)

    def get_linear_regression_evaluation(self, features, board):
        model = self._load_model(os.path.join("models", "lr.pkl"))
        if model is None:
            return self._heuristic_eval(board)
        X = self.get_features(self.modelType)
        try:
            return float(model.predict(X)[0])
        except Exception:
            return self._heuristic_eval(board)

    def get_neural_network_evaluation(self, features, board):
        model = self._load_model(os.path.join("models", "nn.pkl"))
        if model is None:
            return self._heuristic_eval(board)
        X = self.get_features(self.modelType)
        try:
            return float(model.predict(X)[0])
        except Exception:
            return self._heuristic_eval(board)

    def get_different_ML_model_evaluation(self, features, board):
        return self._heuristic_eval(board)


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.name = 'random'
        self.player_string = 'Player {}: random'.format(player_number)

    def get_move(self, board):
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)
        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.name = 'human'
        self.player_string = 'Player {}: human'.format(player_number)

    def get_move(self, board):
        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        print(f"VALID MOVES (column index): {valid_cols}")
        move = input('Enter your move, Human: ')
        if move:
            move = int(move)
        else:
            move = -1

        while move not in valid_cols:
            print('Column full or invalid, choose from:{}'.format(valid_cols))
            move = input('Enter your move, Human: ')
            if move:
                move = int(move)
            else:
                move = -1
        return move


# CODE FOR MCTS
class MCTSNode:
    def __init__(self, board, player_number, parent):
        self.board = board
        self.player_number = player_number
        self.other_player_number = 1 if player_number == 2 else 2
        self.parent = parent
        self.moves = get_valid_moves(board)
        self.terminal = (len(self.moves) == 0) or is_winning_state(board, player_number) or is_winning_state(board, self.other_player_number)
        self.children = {m: None for m in self.moves}

        # MCTS stats (from PARENT perspective)
        self.n = 0
        self.w = 0
        self.c = np.sqrt(2)

    def print_node(self):
        print('Total Node visits and wins: ', self.n, self.w)
        print('Children: ')
        for m in self.moves:
            if self.children[m] is None:
                print('   ', m, ' is None')
            else:
                print('   ', m, ':', self.children[m].n, self.children[m].w, 'UB: ', self.children[m].upper_bound(self.n))

    def print_tree(self):
        print("****")
        print_node(self)
        for m in self.moves:
            if self.children[m]:
                self.children[m].print_tree()
        print("****")

    def max_child(self):
        max_n, max_m = -1, None
        for m in self.moves:
            if self.children[m] is not None and self.children[m].n > max_n:
                max_n, max_m = self.children[m].n, m
        return max_m if max_m is not None else np.random.choice(self.moves)

    def upper_bound(self, N):
        if self.n == 0:
            return np.inf
        return (self.w / self.n) + self.c * np.sqrt(np.log(max(1, N)) / self.n)

    def select(self):
        if self.terminal:
            return self

        max_ub = -np.inf
        max_child = None
        for m in self.moves:
            if self.children[m] is None:
                new_board = np.copy(self.board)
                make_move(new_board, m, self.player_number)
                self.children[m] = MCTSNode(new_board, self.other_player_number, self)
                return self.children[m]

            current_ub = self.children[m].upper_bound(self.n)
            if current_ub > max_ub:
                max_ub = current_ub
                max_child = m

        return self.children[max_child].select()

    def simulate(self):
        # If terminal at entry, score immediately
        if self.terminal:
            if is_winning_state(self.board, self.other_player_number):
                result = 1
            elif is_winning_state(self.board, self.player_number):
                result = -1
            else:
                result = 0
            self.n += 1
            self.w += result
            if self.parent is not None:
                self.parent.back(-result)
            return

        # Random rollout
        rollout = np.copy(self.board)
        cur = self.player_number
        while True:
            moves = get_valid_moves(rollout)
            if not moves:
                result = 0
                break
            m = np.random.choice(moves)
            make_move(rollout, m, cur)
            if is_winning_state(rollout, cur):
                result = 1 if cur == self.other_player_number else -1
                break
            cur = 1 if cur == 2 else 2

        self.n += 1
        self.w += result
        if self.parent is not None:
            self.parent.back(-result)

    def back(self, score):
        self.n += 1
        self.w += score
        if self.parent is not None:
            self.parent.back(-score)


# UTILITY FUNCTIONS

def make_move(board, move, player_number):
    row = 0
    while row < 6 and board[row, move] == 0:
        row += 1
    board[row - 1, move] = player_number

def get_valid_moves(board):
    valid_moves = []
    for c in range(7):
        if 0 in board[:, c]:
            valid_moves.append(c)
    return valid_moves

def is_winning_state(board, player_num):
    player_win_str = '{0}{0}{0}{0}'.format(player_num)
    to_str = lambda a: ''.join(a.astype(str))

    def check_horizontal(b):
        for row in b:
            if player_win_str in to_str(row):
                return True
        return False

    def check_verticle(b):
        return check_horizontal(b.T)

    def check_diagonal(b):
        for op in [None, np.fliplr]:
            op_board = op(b) if op else b

            root_diag = np.diagonal(op_board, offset=0).astype(int)
            if player_win_str in to_str(root_diag):
                return True

            for i in range(1, b.shape[1] - 3):
                for offset in [i, -i]:
                    diag = np.diagonal(op_board, offset=offset)
                    diag = to_str(diag.astype(int))
                    if player_win_str in diag:
                        return True
        return False

    return (check_horizontal(board) or
            check_verticle(board) or
            check_diagonal(board))
