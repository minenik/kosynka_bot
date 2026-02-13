import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import time
from time import sleep
import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
from kosynka import KosynkaBot

SUITS = ['b', 'h', 'k', 'p']
RANKS = ['t', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'v', 'd', 'k']

VALUE_MAP = {r: i + 1 for i, r in enumerate(RANKS)}


def resolve_torch_device(requested: str = "auto") -> torch.device:
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def card_value(card: str) -> int:
    for r in RANKS:
        if card.startswith(r):
            return VALUE_MAP[r]
    return 0


def card_color(card: str) -> str:
    return 'black' if card[-1] in ('p', 'k') else 'red'


ACTION_TYPES = [
    "draw",
    "waste_to_tableau",
    "waste_to_foundation",
    "tableau_to_foundation",
    "tableau_to_tableau",
    "move_stack",
    "foundation_to_tableau",
]
ACTION_TYPE_TO_IDX = {name: i for i, name in enumerate(ACTION_TYPES)}

STATE_FEATURE_SIZE = 43
ACTION_FEATURE_SIZE = len(ACTION_TYPES) + 3 + len(SUITS)


def _norm_card_value(card: str) -> float:
    return card_value(card) / 13.0 if card else 0.0


def _card_color_flag(card: str) -> float:
    if not card:
        return 0.0
    return 1.0 if card_color(card) == "red" else -1.0


def encode_action_features(action: Tuple[Any, ...]) -> List[float]:
    vec = [0.0] * ACTION_FEATURE_SIZE
    a_type = action[0]
    type_idx = ACTION_TYPE_TO_IDX.get(a_type)
    if type_idx is not None:
        vec[type_idx] = 1.0

    src_idx = len(ACTION_TYPES)
    dst_idx = src_idx + 1
    start_idx = src_idx + 2
    suit_idx = src_idx + 3

    if a_type == "waste_to_tableau":
        _, j = action
        vec[dst_idx] = j / 6.0
    elif a_type == "tableau_to_foundation":
        _, i = action
        vec[src_idx] = i / 6.0
    elif a_type == "tableau_to_tableau":
        _, i, j = action
        vec[src_idx] = i / 6.0
        vec[dst_idx] = j / 6.0
    elif a_type == "move_stack":
        _, from_col, start, to_col = action
        vec[src_idx] = from_col / 6.0
        vec[dst_idx] = to_col / 6.0
        vec[start_idx] = min(start, 18) / 18.0
    elif a_type == "foundation_to_tableau":
        _, suit, j = action
        vec[dst_idx] = j / 6.0
        if suit in SUITS:
            vec[suit_idx + SUITS.index(suit)] = 1.0
    return vec


def env_state_vector(env: "SolitaireEnv") -> List[float]:
    vec: List[float] = []

    foundation_counts = [len(env.foundation[s]) for s in SUITS]
    vec.extend([c / 13.0 for c in foundation_counts])

    waste_top = env.waste[-1] if env.waste else ""
    vec.append(_norm_card_value(waste_top))
    vec.append(_card_color_flag(waste_top))
    vec.append(len(env.stock) / 24.0)
    vec.append(len(env.waste) / 24.0)

    for i in range(7):
        col = env.tableau[i]
        total = len(col)
        face_up = sum(1 for _, f in col if f)
        hidden = total - face_up
        top = env._top_face_up(i) or ""

        vec.append(total / 19.0)
        vec.append(face_up / 19.0)
        vec.append(hidden / 19.0)
        vec.append(_norm_card_value(top))
        vec.append(_card_color_flag(top))

    return vec


def board_state_vector(foundation, deck_name, tableau_tops) -> List[float]:
    vec: List[float] = []
    vec.extend([v / 13.0 for v in foundation])
    vec.append(_norm_card_value(deck_name))
    vec.append(_card_color_flag(deck_name))
    vec.append(0.0)  # unknown stock size in GUI mode
    vec.append(0.0)  # unknown waste size in GUI mode

    for top in tableau_tops:
        has = 1.0 if top else 0.0
        vec.append(has)          # approximate total
        vec.append(has)          # approximate face_up
        vec.append(0.0)          # unknown hidden
        vec.append(_norm_card_value(top))
        vec.append(_card_color_flag(top))
    return vec


def gui_action_to_env_action(action: Tuple[Any, ...]) -> Tuple[Any, ...]:
    if action[0] == "waste_to_tableau":
        return ("waste_to_tableau", action[1] - 1)
    if action[0] == "tableau_to_foundation":
        return ("tableau_to_foundation", action[1] - 1)
    if action[0] == "tableau_to_tableau":
        return ("tableau_to_tableau", action[1] - 1, action[2] - 1)
    return action


class SolitaireEnv:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.max_steps = 700
        self.max_stagnation = 180
        self.reset()

    def reset(self) -> Tuple[Any, ...]:
        deck = [r + s for s in SUITS for r in RANKS]
        self.rng.shuffle(deck)
        self.tableau: List[List[Tuple[str, bool]]] = []
        for i in range(7):
            column: List[Tuple[str, bool]] = []
            for j in range(i + 1):
                card = deck.pop(0)
                column.append((card, j == i))
            self.tableau.append(column)
        self.stock = deck
        self.waste: List[str] = []
        self.foundation: Dict[str, List[str]] = {s: [] for s in SUITS}
        self.steps = 0
        self.stagnation = 0
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.state_visits[self._state_signature()] = 1
        return self.get_state()

    def get_state(self) -> Tuple[Any, ...]:
        foundation_counts = tuple(len(self.foundation[s]) for s in SUITS)
        deck_card = self.waste[-1] if self.waste else ''
        tableau_tops = []
        for col in self.tableau:
            top = ''
            for card, face in reversed(col):
                if face:
                    top = card
                    break
            tableau_tops.append(top)
        return foundation_counts, deck_card, tuple(tableau_tops)

    def _top_face_up(self, idx: int) -> str | None:
        col = self.tableau[idx]
        for c, face in reversed(col):
            if face:
                return c
        return None

    def _remove_top_face_up(self, idx: int) -> str | None:
        col = self.tableau[idx]
        for i in range(len(col) - 1, -1, -1):
            card, face = col[i]
            if face:
                col.pop(i)
                if i > 0 and not col[i - 1][1]:
                    col[i - 1] = (col[i - 1][0], True)
                return card
        return None

    def _can_place(self, card: str, target_idx: int) -> bool:
        target_top = self._top_face_up(target_idx)
        if target_top is None:
            return card_value(card) == 13
        return (
            card_color(card) != card_color(target_top)
            and card_value(card) + 1 == card_value(target_top)
        )

    def legal_actions(self) -> List[Tuple[Any, ...]]:
        actions: List[Tuple[Any, ...]] = []
        
        if self.stock or self.waste:
            actions.append(('draw',))
        
        deck_card = self.waste[-1] if self.waste else None
        if deck_card:
            val = card_value(deck_card)
            suit = deck_card[-1]
            if val == len(self.foundation[suit]) + 1:
                actions.append(('waste_to_foundation',))
            for j in range(7):
                if self._can_place(deck_card, j):
                    actions.append(('waste_to_tableau', j))

        for i in range(7):
            top_card = self._top_face_up(i)
            if top_card:
                val = card_value(top_card)
                suit = top_card[-1]
                if val == len(self.foundation[suit]) + 1:
                    actions.append(('tableau_to_foundation', i))

            col = self.tableau[i]
            for start in range(len(col)):
                card, face = col[start]
                if not face:
                    continue
                tail = col[start:]
                if not self._is_valid_stack(tail):
                    continue
                for j in range(7):
                    if i == j:
                        continue
                    if self._can_place(tail[0][0], j):
                        actions.append(('move_stack', i, start, j))

        for suit, stack in self.foundation.items():
            if not stack:
                continue
            card = stack[-1]
            for j in range(7):
                if self._can_place(card, j):
                    actions.append(('foundation_to_tableau', suit, j))

        return actions

    def _is_valid_stack(self, tail: List[Tuple[str, bool]]) -> bool:
        for a, b in zip(tail, tail[1:]):
            if not a[1] or not b[1]:
                return False
            if card_color(a[0]) == card_color(b[0]):
                return False
            if card_value(a[0]) != card_value(b[0]) + 1:
                return False
        return True

    def foundation_count(self) -> int:
        return sum(len(p) for p in self.foundation.values())

    def hidden_count(self) -> int:
        hidden = 0
        for col in self.tableau:
            hidden += sum(1 for _, face in col if not face)
        return hidden

    def _state_signature(self) -> str:
        tableau_sig = []
        for col in self.tableau:
            tableau_sig.append(tuple((c, f) for c, f in col))
        return f"{tuple(tableau_sig)}|{tuple(self.waste)}|{tuple(self.stock)}|{tuple((s, tuple(self.foundation[s])) for s in SUITS)}"

    def step(self, action: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], float, bool]:
        prev_foundation = self.foundation_count()
        prev_hidden = self.hidden_count()
        prev_empty = sum(1 for col in self.tableau if len(col) == 0)
        reward = -0.01
        recycled = False

        match action[0]:
            case 'draw':
                if self.stock:
                    self.waste.append(self.stock.pop(0))
                    reward -= 0.02
                else:
                    self.stock = list(reversed(self.waste))
                    self.waste = []
                    if self.stock:
                        self.waste.append(self.stock.pop(0))
                        recycled = True
                    reward -= 0.2

            case 'waste_to_tableau':
                _, j = action
                card = self.waste.pop()
                self.tableau[j].append((card, True))
                reward += 0.1

            case 'waste_to_foundation':
                card = self.waste.pop()
                self.foundation[card[-1]].append(card)
                reward += 8.0

            case 'tableau_to_foundation':
                _, i = action
                card = self._remove_top_face_up(i)
                if card:
                    self.foundation[card[-1]].append(card)
                    reward += 8.0

            case 'tableau_to_tableau':
                _, i, j = action
                card = self._remove_top_face_up(i)
                if card:
                    self.tableau[j].append((card, True))
                    reward += 0.15 if i != j else -0.5

            case 'move_stack':
                _, from_col, start_idx, to_col = action
                stack = self.tableau[from_col][start_idx:]
                self.tableau[to_col].extend(stack)
                del self.tableau[from_col][start_idx:]
                if start_idx > 0 and not self.tableau[from_col][start_idx - 1][1]:
                    self.tableau[from_col][start_idx - 1] = (self.tableau[from_col][start_idx - 1][0], True)
                reward += 0.1

            case 'foundation_to_tableau':
                _, suit, j = action
                if self.foundation[suit]:
                    card = self.foundation[suit].pop()
                    self.tableau[j].append((card, True))
                    reward -= 2.0

        self.steps += 1
        foundation_gain = self.foundation_count() - prev_foundation
        hidden_gain = prev_hidden - self.hidden_count()
        empty_gain = sum(1 for col in self.tableau if len(col) == 0) - prev_empty

        if foundation_gain > 0:
            reward += 5.0 * foundation_gain
        if hidden_gain > 0:
            reward += 3.0 * hidden_gain
        if empty_gain > 0:
            reward += 1.5 * empty_gain

        if foundation_gain == 0 and hidden_gain == 0 and action[0] != 'draw':
            reward -= 0.05
        if recycled:
            reward -= 0.15

        if foundation_gain == 0 and hidden_gain == 0:
            self.stagnation += 1
        else:
            self.stagnation = 0

        sig = self._state_signature()
        self.state_visits[sig] += 1
        if self.state_visits[sig] > 1:
            # Penalize loops; stronger for repeated revisits.
            reward -= min(0.5, 0.04 * (self.state_visits[sig] - 1))

        done = self.foundation_count() == 52
        if done:
            reward += 120.0
        elif self.steps >= self.max_steps or self.stagnation >= self.max_stagnation:
            done = True

        return self.get_state(), reward, done
    
    def completion_percent(self) -> float:
        total = sum(len(pile) for pile in self.foundation.values())
        return 100.0 * total / 52


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.12,
        gamma: float = 0.995,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
    ):
        self.q: Dict[Tuple[str, str], float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _key(self, state: Tuple[Any, ...], action: Tuple[Any, ...]) -> Tuple[str, str]:
        return str(state), str(action)

    def choose(self, state: Tuple[Any, ...], actions: List[Tuple[Any, ...]]) -> Tuple[Any, ...] | None:
        if not actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(actions)
        best = None
        best_q = float('-inf')
        for a in actions:
            q = self.q[self._key(state, a)]
            if q > best_q or best is None:
                best_q = q
                best = a
        return best

    def update(
        self,
        state: Tuple[Any, ...],
        action: Tuple[Any, ...],
        reward: float,
        next_state: Tuple[Any, ...],
        next_actions: List[Tuple[Any, ...]],
        done: bool,
    ):
        key = self._key(state, action)
        best_next = 0.0 if done else max((self.q[self._key(next_state, a)] for a in next_actions), default=0.0)
        self.q[key] += self.alpha * (reward + self.gamma * best_next - self.q[key])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)



def train_q_learning(episodes: int = 1000, verbose: bool = False) -> Dict[Tuple[str, str], float]:
    env = SolitaireEnv()
    agent = QLearningAgent()
    completion_scores = []
    wins = 0

    for ep in range(episodes):
        state = env.reset()
        done = False

        if verbose:
            print_tableau(env.tableau, env.stock, env.waste, env.foundation)
            print("Reward: -")

        while not done:
            actions = env.legal_actions()
            action = agent.choose(state, actions)
            if action is None:
                if verbose:
                    print("No legal actions.")
                break

            next_state, reward, done = env.step(action)
            next_actions = env.legal_actions()
            agent.update(state, action, reward, next_state, next_actions, done)
            state = next_state

            if verbose:
                print_tableau(env.tableau, env.stock, env.waste, env.foundation)
                print(f"Reward: {reward}")
                bar_width = 20
                filled = int((ep + 1) / episodes * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(f"Episode {ep+1}/{episodes} [{bar}] {100*(ep+1)/episodes:.1f}% | Completion: {env.completion_percent():.1f}%")

        percent = env.completion_percent()
        completion_scores.append(percent)
        if percent >= 100.0:
            wins += 1
        agent.decay_epsilon()
        bar_width = 20
        filled = int((ep + 1) / episodes * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"\rEpisode {ep+1}/{episodes} [{bar}] {100*(ep+1)/episodes:.1f}% "
            f"| Completion: {percent:.1f}% | eps: {agent.epsilon:.3f}",
            end='',
            flush=True,
        )

    avg_completion = sum(completion_scores) / len(completion_scores)
    win_rate = 100.0 * wins / max(1, episodes)
    print(f"\nAverage completion over {episodes} episodes: {avg_completion:.1f}% of cards.")
    print(f"Wins: {wins}/{episodes} ({win_rate:.2f}%)")
    print()
    return agent.q
def print_tableau(tableau, stock, waste, foundation):
    print("====== Состояние поля ======")
    print("Домики (foundation): ", end="")
    for suit in SUITS:
        if foundation[suit]:
            print(f"{foundation[suit][-1].upper()}", end=" ")
        else:
            print("..", end=" ")
    print()

    print("Колода: ", end="")
    if stock:
        print("??", end=" | ")
    else:
        print("--", end=" | ")

    print("Сброс: ", end="")
    for i in range(1, 4):
        if len(waste) >= i:
            print(waste[-i].upper(), end=" ")
        else:
            print("..", end=" ")
    print("\n")

    for i, col in enumerate(tableau):
        print(f"{i+1}: ", end="")
        for card, face in col:
            print(card.upper() if face else "??", end=" ")
        print()

    total = sum(len(pile) for pile in foundation.values())
    percent = 100.0 * total / 52
    print(f"\nПрогресс: {total}/52 карт в домиках ({percent:.1f}%)")
    print("============================\n")

def save_model(q_table: Dict[Tuple[str, str], float], path: str):
    with open(path, 'wb') as f:
        pickle.dump(dict(q_table), f)


def load_model(path: str) -> Dict[Tuple[str, str], float]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return defaultdict(float, data)


class RLKosynkaBot(KosynkaBot):
    def __init__(self, q_table: Dict[Tuple[str, str], float]):
        self.q_table = q_table
        super().__init__()

    def state_from_board(self, deck_card, field) -> Tuple[Any, ...]:
        foundation = (
            self.home_state['b'],
            self.home_state['h'],
            self.home_state['k'],
            self.home_state['p'],
        )
        deck_name = deck_card[0] if deck_card else ''
        tableau_tops = []
        for col in range(1, 8):
            mem = self.column_memory.get(col, [])
            tableau_tops.append(mem[-1] if mem else '')
        return foundation, deck_name, tuple(tableau_tops)

    def board_actions(self, deck_card, field, ox, oy):
        actions = []
        seen_moves = set()

        if deck_card:
            name, pos = deck_card
            val = card_value(name)
            suit = name[-1]
            if val == self.home_state[suit] + 1:
                actions.append((('waste_to_foundation',), ('double', pos)))
            for j in range(1, 8):
                if self.can_place(name, field.get(j, [])):
                    dst = self.estimate_card_position(j, field[j], False)
                    move_key = f"{name}_to_{j}"
                    if move_key not in seen_moves:
                        seen_moves.add(move_key)
                        actions.append((('waste_to_tableau', j), ('drag', pos, dst)))

        for i in range(1, 8):
            cards = field.get(i)
            if not cards:
                continue
            top_name = cards[-1][0]
            pos = self.estimate_card_position(i, cards)
            val = card_value(top_name)
            suit = top_name[-1]

            if val == self.home_state[suit] + 1:
                actions.append((('tableau_to_foundation', i), ('double', pos)))

            for j in range(1, 8):
                if i == j:
                    continue
                if self.can_place(top_name, field.get(j, [])):
                    dst = self.estimate_card_position(j, field[j], False)

                    if field[j] and field[j][-1][0] == top_name:
                        continue

                    move_key = f"{top_name}_{i}_to_{j}"
                    if move_key not in seen_moves:
                        seen_moves.add(move_key)
                        actions.append((('tableau_to_tableau', i, j), ('drag', pos, dst)))

        actions.append((('draw',), ('click_deck',)))
        return actions

    def choose_gui_action(self, state, actions):
        best = actions[0]
        best_q = self.q_table[(str(state), str(actions[0][0]))]
        for a in actions[1:]:
            q = self.q_table[(str(state), str(a[0]))]
            if q > best_q:
                best_q = q
                best = a
        return best

    def run_loop(self):
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        self.root.withdraw()
        self.root.update()
        while self.running:
            screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            field_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            gray_field = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
            self.img_width = field_img.shape[1]
            self.img_height = field_img.shape[0]
            matches = self.match_all_cards(gray_field)
            matches = self.filter_duplicates(matches)
            field = self.build_field_structure_dynamic(matches, field_img.shape[1])
            deck_card = self.detect_deck_card(matches, self.img_width, self.img_height)
            homes = self.detect_home_cards(matches, self.img_width, self.img_height)
            self.update_stock_click(deck_card)
            self.update_home_state_from_homes(homes)
            state = self.state_from_board(deck_card, field)
            actions = self.board_actions(deck_card, field, x, y)
            chosen_id, exec_info = self.choose_gui_action(state, actions)
            match exec_info[0]:
                case 'double':
                    _, pos = exec_info
                    pyautogui.doubleClick(x + pos[0], y + pos[1])
                case 'drag':
                    _, src, dst = exec_info
                    pyautogui.moveTo(x + src[0], y + src[1])
                    pyautogui.dragTo(x + dst[0], y + dst[1], duration=0.4)
                case 'click_deck':
                    self.click_stock(x, y)
            time.sleep(1.0)
        self.root.deiconify()


def _is_foundation_move_safe(card: str, foundation: Dict[str, List[str]]) -> bool:
    v = card_value(card)
    suit = card[-1]
    if v <= 2:
        return True
    if suit in ("p", "k"):
        opp = min(len(foundation["b"]), len(foundation["h"]))
    else:
        opp = min(len(foundation["p"]), len(foundation["k"]))
    return opp >= (v - 2)


def heuristic_action_score(env: SolitaireEnv, action: Tuple[Any, ...]) -> float:
    a_type = action[0]
    score = 0.0

    if a_type == "draw":
        score -= 3.0
    elif a_type == "waste_to_foundation":
        if env.waste:
            card = env.waste[-1]
            score += 60.0 if _is_foundation_move_safe(card, env.foundation) else 8.0
    elif a_type == "tableau_to_foundation":
        _, i = action
        card = env._top_face_up(i)
        if card:
            score += 55.0 if _is_foundation_move_safe(card, env.foundation) else 6.0
            col = env.tableau[i]
            if col:
                top_idx = len(col) - 1
                if top_idx > 0 and not col[top_idx - 1][1]:
                    score += 30.0
    elif a_type == "waste_to_tableau":
        _, j = action
        score += 10.0 if len(env.tableau[j]) == 0 else 4.0
    elif a_type == "tableau_to_tableau":
        _, i, j = action
        score += 7.0
        if len(env.tableau[j]) == 0:
            card = env._top_face_up(i)
            if card and card_value(card) == 13:
                score += 18.0
        col = env.tableau[i]
        if col:
            top_idx = len(col) - 1
            if top_idx > 0 and not col[top_idx - 1][1]:
                score += 26.0
    elif a_type == "move_stack":
        _, from_col, start_idx, to_col = action
        score += 8.0
        if len(env.tableau[to_col]) == 0:
            card = env.tableau[from_col][start_idx][0]
            if card_value(card) == 13:
                score += 18.0
        if start_idx > 0 and not env.tableau[from_col][start_idx - 1][1]:
            score += 26.0
    elif a_type == "foundation_to_tableau":
        score -= 25.0

    return score


def choose_heuristic_action(env: SolitaireEnv, legal_actions: List[Tuple[Any, ...]]) -> Tuple[Any, ...] | None:
    if not legal_actions:
        return None
    best = None
    best_score = float("-inf")
    for action in legal_actions:
        s = heuristic_action_score(env, action)
        if s > best_score:
            best_score = s
            best = action
    return best


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device: torch.device,
        lr=3e-4,
        gamma=0.995,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99998,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=120000)
        self.loss_fn = nn.SmoothL1Loss()
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.batch_size = 384 if self.device.type == "cuda" else 96
        self.learn_steps = 0
        self.target_sync_every = 600
        self.replay_start_size = 6000 if self.device.type == "cuda" else 1500

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size + ACTION_FEATURE_SIZE, 384),
            nn.ReLU(),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
        )

    def _state_action_batch(self, states, actions):
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a_features = [encode_action_features(a) for a in actions]
        a = torch.tensor(a_features, dtype=torch.float32, device=self.device)
        return torch.cat([s, a], dim=1)

    def q_for_actions(self, state, actions):
        if not actions:
            return []
        state_batch = [state] * len(actions)
        sa = self._state_action_batch(state_batch, actions)
        with torch.no_grad():
            q = self.model(sa).squeeze(1)
        return [float(v) for v in q]

    def remember(self, state, action, reward, next_state, next_legal_actions, done):
        self.memory.append((state, action, reward, next_state, list(next_legal_actions), done))

    def act(self, state, legal_actions):
        if not legal_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        q_values = self.q_for_actions(state, legal_actions)
        best_idx = max(range(len(legal_actions)), key=lambda i: q_values[i])
        return legal_actions[best_idx]

    def replay(self):
        if len(self.memory) < self.replay_start_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, next_legals, dones = zip(*batch)

        with torch.no_grad():
            targets = [float(r) for r in rewards]
            flat_states = []
            flat_actions = []
            owner_idx = []
            for i in range(self.batch_size):
                if dones[i]:
                    continue
                legal = list(next_legals[i])
                if not legal:
                    continue
                ns = next_states[i]
                for a in legal:
                    flat_states.append(ns)
                    flat_actions.append(a)
                    owner_idx.append(i)

            if flat_actions:
                sa_next = self._state_action_batch(flat_states, flat_actions)
                online_q = self.model(sa_next).squeeze(1)
                target_q = self.target_model(sa_next).squeeze(1)

                by_owner: Dict[int, List[int]] = defaultdict(list)
                for idx, owner in enumerate(owner_idx):
                    by_owner[owner].append(idx)

                for owner, idxs in by_owner.items():
                    best_local = max(idxs, key=lambda t: float(online_q[t].item()))
                    best_next = float(target_q[best_local].item())
                    targets[owner] = float(rewards[owner]) + self.gamma * best_next

            target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            sa_current = self._state_action_batch(states, actions)
            predicted = self.model(sa_current).squeeze(1)
            loss = self.loss_fn(predicted, target_tensor)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def save_dqn_train_checkpoint(path: str, agent: DQNAgent, episode_done: int, best_completion: float):
    torch.save(
        {
            "model_state": agent.model.state_dict(),
            "target_state": agent.target_model.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "learn_steps": agent.learn_steps,
            "episode_done": episode_done,
            "best_completion": best_completion,
        },
        path,
    )


def load_dqn_train_checkpoint(path: str, agent: DQNAgent):
    data = torch.load(path, map_location=agent.device)
    if isinstance(data, dict) and "model_state" in data:
        agent.model.load_state_dict(data["model_state"])
        if "target_state" in data:
            agent.target_model.load_state_dict(data["target_state"])
        else:
            agent.target_model.load_state_dict(agent.model.state_dict())
        if "optimizer_state" in data:
            agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.epsilon = float(data.get("epsilon", agent.epsilon))
        agent.learn_steps = int(data.get("learn_steps", agent.learn_steps))
        return int(data.get("episode_done", 0)), float(data.get("best_completion", 0.0))

    # Fallback: plain model weights.
    agent.model.load_state_dict(data)
    agent.target_model.load_state_dict(agent.model.state_dict())
    return 0, 0.0


def train_dqn(
    episodes=1000,
    model_path="dqn_model.pt",
    verbose=False,
    device="auto",
    resume=False,
    log_interval=50,
    save_every=500,
    update_every=2,
    gradient_updates=2,
):
    env = SolitaireEnv()
    state_size = STATE_FEATURE_SIZE
    action_size = 1
    torch_device = resolve_torch_device(device)
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if update_every < 1:
        update_every = 1
    if gradient_updates < 1:
        gradient_updates = 1
    if log_interval < 1:
        log_interval = 1
    if save_every < 0:
        save_every = 0

    print(f"Training device: {torch_device}")
    agent = DQNAgent(state_size, action_size, device=torch_device)
    completion_scores = []
    wins = 0
    best_completion = 0.0
    recent_scores = deque(maxlen=200)
    train_checkpoint_path = model_path + ".train"
    start_episode = 0
    total_env_steps = 0

    if resume:
        if os.path.exists(train_checkpoint_path):
            start_episode, best_completion = load_dqn_train_checkpoint(train_checkpoint_path, agent)
            print(
                f"Resumed from {train_checkpoint_path} "
                f"(episode={start_episode}, eps={agent.epsilon:.3f}, best={best_completion:.1f}%)"
            )
        elif os.path.exists(model_path):
            _, best_completion = load_dqn_train_checkpoint(model_path, agent)
            print(f"Loaded weights from {model_path}. Optimizer state is fresh.")
        else:
            print(f"[!] Resume requested, but no checkpoint found at {train_checkpoint_path} or {model_path}")

    for ep in range(start_episode, start_episode + episodes):
        env.reset()
        state = env_state_vector(env)
        done = False
        # Early training is heavily guided by heuristic policy; then fades out.
        guide_prob = max(0.0, 0.75 * (1.0 - ((ep - start_episode) / max(1, episodes * 0.65))))

        if verbose:
            print_tableau(env.tableau, env.stock, env.waste, env.foundation)
            print("Reward: -")

        while not done:
            legal_actions = env.legal_actions()
            if not legal_actions:
                if verbose:
                    print("No legal actions. Ending episode.")
                break

            if random.random() < guide_prob:
                action = choose_heuristic_action(env, legal_actions)
                if action is None:
                    action = agent.act(state, legal_actions)
            else:
                action = agent.act(state, legal_actions)
            _, reward, done = env.step(action)
            next_state = env_state_vector(env)
            next_legal = env.legal_actions()
            agent.remember(state, action, reward, next_state, next_legal, done)
            state = next_state
            total_env_steps += 1
            if total_env_steps % update_every == 0:
                for _ in range(gradient_updates):
                    agent.replay()

            if verbose:
                print_tableau(env.tableau, env.stock, env.waste, env.foundation)
                print(f"Reward: {reward}")

        percent = env.completion_percent()
        completion_scores.append(percent)
        recent_scores.append(percent)
        if percent >= 100.0:
            wins += 1

        if percent > best_completion:
            best_completion = percent
            torch.save(agent.model.state_dict(), model_path + ".best")

        local_ep = ep - start_episode + 1
        if local_ep == 1 or local_ep % log_interval == 0 or local_ep == episodes:
            bar_width = 20
            filled = int(local_ep / episodes * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"Episode {local_ep}/{episodes} [{bar}] {100*local_ep/episodes:.1f}% "
                f"| Completion: {percent:.1f}% | best: {best_completion:.1f}% "
                f"| avg200: {sum(recent_scores)/len(recent_scores):.1f}% | eps: {agent.epsilon:.3f}"
            )

        if save_every and (local_ep % save_every == 0):
            save_dqn_train_checkpoint(train_checkpoint_path, agent, ep + 1, best_completion)
            torch.save(agent.model.state_dict(), model_path)

    avg_completion = sum(completion_scores) / len(completion_scores)
    win_rate = 100.0 * wins / max(1, episodes)
    print(f"\nAverage completion over {episodes} episodes: {avg_completion:.1f}% of cards.\n")
    print(f"Wins: {wins}/{episodes} ({win_rate:.2f}%)")
    print(f"Best single-episode completion: {best_completion:.1f}%")

    torch.save(agent.model.state_dict(), model_path)
    save_dqn_train_checkpoint(train_checkpoint_path, agent, start_episode + episodes, best_completion)
    print(f"DQN model saved to {model_path}")

def flatten_state(state: Tuple[Any, ...]) -> List[float]:
    foundation, deck_card, tableau = state
    f_vec = list(foundation)
    d_vec = [card_value(deck_card)] if deck_card else [0]
    t_vec = [card_value(c) if c else 0 for c in tableau]
    return f_vec + d_vec + t_vec

def generate_action_space() -> List[Tuple[Any, ...]]:
    space = [('draw',)]
    for j in range(7):
        space.append(('waste_to_tableau', j))
    for i in range(7):
        space.append(('tableau_to_foundation', i))
    for i in range(7):
        for j in range(7):
            space.append(('tableau_to_tableau', i, j))
    for i in range(7):
        for s in range(20):
            for j in range(7):
                space.append(('move_stack', i, s, j))
    for suit in SUITS:
        for j in range(7):
            space.append(('foundation_to_tableau', suit, j))
    space.append(('waste_to_foundation',))
    return space

def get_legal_action_ids(legal_actions: List[Tuple[Any, ...]], action_space: List[Tuple[Any, ...]]) -> List[int]:
    legal_set = set(legal_actions)
    return [i for i, act in enumerate(action_space) if act in legal_set]


def play_with_model(model_path: str, use_dqn: bool = False, device: str = "auto"):
    if use_dqn or model_path.endswith(".pt"):
        state_size = STATE_FEATURE_SIZE
        action_size = 1
        torch_device = resolve_torch_device(device)
        agent = DQNAgent(state_size, action_size, device=torch_device)
        try:
            agent.model.load_state_dict(torch.load(model_path, map_location=torch_device))
            agent.target_model.load_state_dict(agent.model.state_dict())
        except FileNotFoundError:
            print(f"[!] Model file not found: {model_path}")
            return
        agent.model.eval()
        agent.epsilon = 0.0

        class DQNKosynkaBot(RLKosynkaBot):
            def __init__(self):
                super().__init__(defaultdict(float))

            def run_loop(self_inner):
                x, y = self_inner.root.winfo_x(), self_inner.root.winfo_y()
                w, h = self_inner.root.winfo_width(), self_inner.root.winfo_height()
                self_inner.root.withdraw()
                while self_inner.running:
                    screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                    field_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    gray_field = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
                    self_inner.img_width = field_img.shape[1]
                    self_inner.img_height = field_img.shape[0]
                    matches = self_inner.match_all_cards(gray_field)
                    matches = self_inner.filter_duplicates(matches)
                    field = self_inner.build_field_structure_dynamic(matches, field_img.shape[1])
                    deck_card = self_inner.detect_deck_card(matches, self_inner.img_width, self_inner.img_height)
                    homes = self_inner.detect_home_cards(matches, self_inner.img_width, self_inner.img_height)
                    self_inner.update_stock_click(deck_card)
                    self_inner.update_home_state_from_homes(homes)

                    foundation = (
                        self_inner.home_state['b'],
                        self_inner.home_state['h'],
                        self_inner.home_state['k'],
                        self_inner.home_state['p'],
                    )
                    deck_name = deck_card[0] if deck_card else ''
                    tableau_tops = []
                    for col in range(1, 8):
                        mem = self_inner.column_memory.get(col, [])
                        tableau_tops.append(mem[-1] if mem else '')
                    state = board_state_vector(foundation, deck_name, tuple(tableau_tops))

                    all_actions = self_inner.board_actions(deck_card, field, x, y)
                    env_actions = [gui_action_to_env_action(a[0]) for a in all_actions]
                    if not env_actions:
                        self_inner.click_stock(x, y)
                        time.sleep(1.0)
                        continue

                    chosen_env_action = agent.act(state, env_actions)
                    exec_map = {}
                    for idx, env_a in enumerate(env_actions):
                        if env_a not in exec_map:
                            exec_map[env_a] = all_actions[idx][1]
                    exec_info = exec_map.get(chosen_env_action, ('click_deck',))

                    match exec_info[0]:
                        case 'double':
                            _, pos = exec_info
                            pyautogui.doubleClick(x + pos[0], y + pos[1])
                        case 'drag':
                            _, src, dst = exec_info
                            pyautogui.moveTo(x + src[0], y + src[1])
                            pyautogui.dragTo(x + dst[0], y + dst[1], duration=0.4)
                        case 'click_deck':
                            self_inner.click_stock(x, y)
                    time.sleep(1.0)
                self_inner.root.deiconify()

        DQNKosynkaBot()

    else:
        q = load_model(model_path)
        RLKosynkaBot(q)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Solitaire RL tools')
    parser.add_argument('--train', action='store_true', help='train Q-learning model')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--model', type=str, default='model.pkl')
    parser.add_argument('--play', action='store_true', help='play using a saved model')
    parser.add_argument('--verbose', action='store_true', help='enable verbose training output')
    parser.add_argument('--dqn', action='store_true', help='train DQN neural network instead of Q-learning')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='device for DQN (auto/cpu/cuda)')
    parser.add_argument('--resume', action='store_true', help='resume DQN training from model.train or model')
    parser.add_argument('--log-interval', type=int, default=50, help='print progress every N episodes during DQN training')
    parser.add_argument('--save-every', type=int, default=500, help='save DQN checkpoint every N episodes (0 disables periodic save)')
    parser.add_argument('--update-every', type=int, default=2, help='run backprop every N environment steps during DQN training')
    parser.add_argument('--gradient-updates', type=int, default=2, help='number of replay updates when a backprop step is triggered')

    args = parser.parse_args()
    if args.train:
        if args.dqn:
            train_dqn(
                args.episodes,
                model_path=args.model,
                verbose=args.verbose,
                device=args.device,
                resume=args.resume,
                log_interval=args.log_interval,
                save_every=args.save_every,
                update_every=args.update_every,
                gradient_updates=args.gradient_updates,
            )
        else:
            q = train_q_learning(args.episodes, verbose=args.verbose)
            save_model(q, args.model)
            print(f'Model saved to {args.model}')
    elif args.play:
        play_with_model(args.model, use_dqn=args.dqn, device=args.device)
    else:
        parser.print_help()



