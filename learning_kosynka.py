import random
import torch
import torch.nn as nn
import torch.optim as optim
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


def card_value(card: str) -> int:
    for r in RANKS:
        if card.startswith(r):
            return VALUE_MAP[r]
    return 0


def card_color(card: str) -> str:
    return 'black' if card[-1] in ('p', 'k') else 'red'


class SolitaireEnv:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
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
        
        if self.stock:
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

    def step(self, action: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], int, bool]:
        reward = 0
        match action[0]:
            case 'draw':
                if self.stock:
                    self.waste.append(self.stock.pop(0))
                else:
                    self.stock = list(reversed(self.waste))
                    self.waste = []
                    if self.stock:
                        self.waste.append(self.stock.pop(0))

            case 'waste_to_tableau':
                _, j = action
                card = self.waste.pop()
                self.tableau[j].append((card, True))

            case 'waste_to_foundation':
                card = self.waste.pop()
                self.foundation[card[-1]].append(card)
                reward = 1

            case 'tableau_to_foundation':
                _, i = action
                card = self._remove_top_face_up(i)
                if card:
                    self.foundation[card[-1]].append(card)
                    reward = 1

            case 'tableau_to_tableau':
                _, i, j = action
                card = self._remove_top_face_up(i)
                if card:
                    self.tableau[j].append((card, True))
                    reward = -0.1 if i != j else -0.5

            case 'move_stack':
                _, from_col, start_idx, to_col = action
                stack = self.tableau[from_col][start_idx:]
                self.tableau[to_col].extend(stack)
                del self.tableau[from_col][start_idx:]
                if start_idx > 0 and not self.tableau[from_col][start_idx - 1][1]:
                    self.tableau[from_col][start_idx - 1] = (self.tableau[from_col][start_idx - 1][0], True)

            case 'foundation_to_tableau':
                _, suit, j = action
                if self.foundation[suit]:
                    card = self.foundation[suit].pop()
                    self.tableau[j].append((card, True))
                    reward = -0.5 

        done = sum(len(p) for p in self.foundation.values()) == 52
        return self.get_state(), reward, done
    
    def completion_percent(self) -> float:
        total = sum(len(pile) for pile in self.foundation.values())
        return 100.0 * total / 52


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.q: Dict[Tuple[str, str], float] = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

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

    def update(self, state: Tuple[Any, ...], action: Tuple[Any, ...], reward: float, next_state: Tuple[Any, ...], next_actions: List[Tuple[Any, ...]]):
        key = self._key(state, action)
        best_next = max((self.q[self._key(next_state, a)] for a in next_actions), default=0.0)
        self.q[key] += self.alpha * (reward + self.gamma * best_next - self.q[key])



def train_q_learning(episodes: int = 1000, verbose: bool = False) -> Dict[Tuple[str, str], float]:
    env = SolitaireEnv()
    agent = QLearningAgent()
    completion_scores = [] 

    for ep in range(episodes):
        state_counter = defaultdict(int)
        state = env.reset()
        done = False
        progress = 100.0 * (ep + 1) / episodes

        if verbose:
            print_tableau(env.tableau, env.stock, env.waste, env.foundation)
            print("Reward: —") 



        while not done:
            actions = env.legal_actions()
            action = agent.choose(state, actions)
            if action is None:
                if verbose:
                    print("No legal actions!")
                break

            next_state, reward, done = env.step(action)
            next_actions = env.legal_actions()
            agent.update(state, action, reward, next_state, next_actions)
            state = next_state

            state_counter[state] += 1
            if state_counter[state] > 3:
                if verbose:
                    print("🔁 Зацикливание. Завершаем эпизод.")
                break

            if verbose:
                print_tableau(env.tableau, env.stock, env.waste, env.foundation)
                print(f"Reward: {reward}")
                bar_width = 20
                filled = int((ep + 1) / episodes * bar_width)
                bar = '█' * filled + '▒' * (bar_width - filled)
                print(f"Эпизод {ep+1}/{episodes} [{bar}] {100*(ep+1)/episodes:.1f}% | Прогресс: {env.completion_percent():.1f}%\n")


        percent = env.completion_percent()
        completion_scores.append(percent)
        bar_width = 20
        filled = int((ep + 1) / episodes * bar_width)
        bar = '█' * filled + '▒' * (bar_width - filled)
        print(f"\rЭпизод {ep+1}/{episodes} [{bar}] {100*(ep+1)/episodes:.1f}% | Прогресс: {percent:.1f}%", end='', flush=True)


    avg_completion = sum(completion_scores) / len(completion_scores)
    print(f"\nСредний прогресс за {episodes} эпизодов: {avg_completion:.1f}% карт.")
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
            matches = self.match_all_cards(gray_field)
            matches = self.filter_duplicates(matches)
            field = self.build_field_structure_dynamic(matches, field_img.shape[1])
            deck_card = self.detect_deck_card(matches)
            homes = self.detect_home_cards(matches)
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
                    pyautogui.click(x + 60, y + 60)
            time.sleep(1.0)
        self.root.deiconify()

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.loss_fn = nn.MSELoss()

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_t).detach().numpy()[0]
        best = max(legal_actions, key=lambda a: q_values[a])
        return best

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        targets = self.model(states).detach().clone()
        next_q = self.model(next_states).detach().max(1)[0]
        for i in range(batch_size):
            targets[i][actions[i]] = rewards[i] + (0 if dones[i] else self.gamma * next_q[i])
        self.model.train()
        predictions = self.model(states)
        loss = self.loss_fn(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_dqn(episodes=1000, model_path="dqn_model.pt", verbose=False):
    env = SolitaireEnv()
    state_size = 12
    action_space = generate_action_space()
    action_size = len(action_space)
    agent = DQNAgent(state_size, action_size)
    completion_scores = []

    for ep in range(episodes):
        state = flatten_state(env.reset())
        done = False

        if verbose:
            print_tableau(env.tableau, env.stock, env.waste, env.foundation)
            print("Reward: —")

        state_counter = defaultdict(int)
        while not done:
            legal = get_legal_action_ids(env.legal_actions(), action_space)
            if not legal:
                if verbose:
                    print("⚠️ Нет допустимых действий! Завершаем эпизод.")
                break
            action_idx = agent.act(state, legal)
            action = action_space[action_idx]
            next_state_raw, reward, done = env.step(action)
            next_state = flatten_state(next_state_raw)
            agent.remember(state, action_idx, reward, next_state, done)
            state = next_state
            state_counter[tuple(next_state)] += 1
            if state_counter[tuple(next_state)] > 3:
                if verbose:
                    print("🔁 Зацикливание. Завершаем эпизод.")
                break
            agent.replay()

            if verbose:
                print_tableau(env.tableau, env.stock, env.waste, env.foundation)
                print(f"Reward: {reward}")
                bar_width = 20
                filled = int((ep + 1) / episodes * bar_width)
                bar = '█' * filled + '▒' * (bar_width - filled)
                print(f"Эпизод {ep+1}/{episodes} [{bar}] {100*(ep+1)/episodes:.1f}% | Прогресс: {env.completion_percent():.1f}%\n")

        percent = env.completion_percent()
        completion_scores.append(percent)
        bar_width = 20
        filled = int((ep + 1) / episodes * bar_width)
        bar = '█' * filled + '▒' * (bar_width - filled)
        print(f"\rЭпизод {ep+1}/{episodes} [{bar}] {100*(ep+1)/episodes:.1f}% | Прогресс: {percent:.1f}%", end='', flush=True)

    avg_completion = sum(completion_scores) / len(completion_scores)
    print(f"\nСредний прогресс за {episodes} эпизодов: {avg_completion:.1f}% карт.\n")

    torch.save(agent.model.state_dict(), model_path)
    print(f"DQN модель сохранена в {model_path}")


    torch.save(agent.model.state_dict(), model_path)
    print(f"DQN модель сохранена в {model_path}")

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


def play_with_model(model_path: str, use_dqn: bool = False):
    if use_dqn or model_path.endswith(".pt"):
        state_size = 12
        action_space = generate_action_space()
        action_size = len(action_space)
        agent = DQNAgent(state_size, action_size)
        try:
            agent.model.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print(f"[!] Файл модели {model_path} не найден.")
            return
        agent.model.eval()

        class DQNKosynkaBot(KosynkaBot):
            def run_loop(self_inner):
                x, y = self_inner.root.winfo_x(), self_inner.root.winfo_y()
                w, h = self_inner.root.winfo_width(), self_inner.root.winfo_height()
                self_inner.root.withdraw()
                while self_inner.running:
                    screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                    field_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    gray_field = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
                    self_inner.img_width = field_img.shape[1]
                    matches = self_inner.match_all_cards(gray_field)
                    matches = self_inner.filter_duplicates(matches)
                    field = self_inner.build_field_structure_dynamic(matches, field_img.shape[1])
                    deck_card = self_inner.detect_deck_card(matches)
                    homes = self_inner.detect_home_cards(matches)
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
                    state_raw = (foundation, deck_name, tuple(tableau_tops))
                    state = flatten_state(state_raw)

                    all_actions = self_inner.board_actions(deck_card, field, x, y)
                    action_tuples = [a[0] for a in all_actions]
                    legal_ids = get_legal_action_ids(action_tuples, action_space)
                    chosen_idx = agent.act(state, legal_ids)
                    action, exec_info = all_actions[legal_ids.index(chosen_idx)]

                    match exec_info[0]:
                        case 'double':
                            _, pos = exec_info
                            pyautogui.doubleClick(x + pos[0], y + pos[1])
                        case 'drag':
                            _, src, dst = exec_info
                            pyautogui.moveTo(x + src[0], y + src[1])
                            pyautogui.dragTo(x + dst[0], y + dst[1], duration=0.4)
                        case 'click_deck':
                            pyautogui.click(x + 60, y + 60)
                    time.sleep(1.0)
                self_inner.root.deiconify()

        DQNKosynkaBot().run()

    else:
        q = load_model(model_path)
        RLKosynkaBot(q).run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Solitaire RL tools')
    parser.add_argument('--train', action='store_true', help='train Q-learning model')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--model', type=str, default='model.pkl')
    parser.add_argument('--play', action='store_true', help='play using a saved model')
    parser.add_argument('--verbose', action='store_true', help='enable verbose training output')
    parser.add_argument('--dqn', action='store_true', help='train DQN neural network instead of Q-learning')

    args = parser.parse_args()
    if args.train:
        if args.dqn:
            train_dqn(args.episodes, model_path=args.model, verbose=args.verbose)
        else:
            q = train_q_learning(args.episodes, verbose=args.verbose)
            save_model(q, args.model)
            print(f'Model saved to {args.model}')
    elif args.play:
        play_with_model(args.model, use_dqn=args.dqn)
    else:
        parser.print_help()
