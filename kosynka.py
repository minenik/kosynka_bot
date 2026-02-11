import os
import cv2
import time
import pyautogui
import numpy as np
import tkinter as tk
from PIL import ImageGrab
from collections import defaultdict
from threading import Thread
import statistics

TEMPLATE_FOLDER = "templates/templates"
THRESHOLD = 0.86
TOP_CARD_OFFSET = 50
TOP_ZONE_RATIO = 0.43
TABLEAU_LEFT_RATIO = 0.28
TABLEAU_RIGHT_RATIO = 0.72
TOP_ROW_MIN_RATIO = 0.10
TOP_ROW_MAX_RATIO = 0.34
WASTE_LEFT_RATIO = 0.34
WASTE_RIGHT_RATIO = 0.45
FOUNDATION_LEFT_RATIO = 0.50

class KosynkaBot:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Выдели зону игры")
        self.root.attributes("-topmost", True)
        self.root.geometry("800x400+100+100")
        self.root.configure(bg='gray')
        self.root.attributes('-alpha', 0.3)

        self.button = tk.Button(self.root, text="\U0001f4f7 Старт игры", command=self.toggle_loop)
        self.button.pack(side="bottom", pady=5)

        self.last_move = None
        self.move_history = []
        self.running = False
        self.home_state = {'h': 0, 'b': 0, 'k': 0, 'p': 0}
        self.margin = 30
        self.spacing = 110
        self.column_x = {}
        self.column_memory = {i: [] for i in range(1, 8)}
        self.img_width = 800
        self.img_height = 400
        self.tableau_base_y = 260
        self.stuck_ticks = 0
        self.max_stuck_ticks = 25
        self.undo_attempts = 0
        self.max_undo_attempts = 3
        self.restart_count = 0
        self.last_foundation_total = 0
        self.stock_click_pos = None
        self.last_king_move = None
        self.last_draw_deck_name = None
        self.last_action_was_draw = False
        self.failed_draws = 0
        self.stock_probe_idx = 0
        self.stock_probe_offsets = [
            (0, 0),
            (-12, 0),
            (12, 0),
            (0, -10),
            (0, 10),
            (-18, -8),
            (18, -8),
        ]

        self.root.bind('<Escape>', lambda e: self.stop_loop())
        self.root.bind('<space>', lambda e: self.toggle_loop())

        self.root.mainloop()

    def toggle_loop(self):
        if not self.running:
            self.running = True
            print("\n\U0001f3b2 Старт. Приоритет ходов: колода -> домик, колонки -> домик, король -> пустая, остальные")
            self.thread = Thread(target=self.run_loop, daemon=True)
            self.thread.start()
        else:
            self.running = False

    def stop_loop(self):
        print("\n\U0001f6d1 Остановлено пользователем (Escape)")
        self.running = False

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
            self.observe_draw_result(deck_card)

            self.update_home_state_from_homes(homes)
            self.print_field_state(deck_card, homes, field)

            if self.try_home_drop_deck(deck_card, x, y):
                self.last_action_was_draw = False
                self.register_progress()
                time.sleep(1.0)
                continue

            if self.try_home_drop(field, x, y):
                self.last_action_was_draw = False
                self.register_progress()
                time.sleep(1.0)
                continue

            if self.try_king_to_empty(field, x, y):
                self.last_action_was_draw = False
                time.sleep(1.0)
                continue

            move = self.find_move(field, deck_card)
            if move and move != self.last_move:
                if self.is_repeating_move(move):
                    print("\U0001f501 Обнаружен зацикливающийся ход, остановка")
                    self.handle_stuck(x, y, hard=True)
                    time.sleep(1.0)
                    continue
                source_card, src_pos, dst_pos = move
                print(f"\n\U0001f449 Делаем ход: {source_card} с {src_pos} на {dst_pos}")
                pyautogui.moveTo(int(x + src_pos[0]), int(y + src_pos[1]))
                pyautogui.dragTo(int(x + dst_pos[0]), int(y + dst_pos[1]), duration=0.4)
                src_col, src_idx = self.column_by_position(field, src_pos, True)
                dst_col, _ = self.column_by_position(field, dst_pos, False)
                moved_stack = [source_card]
                if src_col is not None:
                    moved_stack = self.column_memory[src_col][src_idx:]
                    self.column_memory[src_col] = self.column_memory[src_col][:src_idx]
                    field[src_col] = field[src_col][:src_idx]
                else:
                    moved_stack = [source_card]

                if dst_col is not None:
                    self.column_memory[dst_col].extend(moved_stack)
                    field[dst_col].extend([(name, dst_pos[0], dst_pos[1]) for name in moved_stack])

                if src_col is not None and self.column_memory.get(src_col):
                    top_name = self.column_memory[src_col][-1]
                    val = self.get_value(top_name)
                    suit = top_name[-1]
                    if val == self.home_state[suit] + 1:
                        pos = self.estimate_card_position(src_col, field[src_col])
                        print(f"\U0001f4a1 Сразу в домик после хода: {top_name} ({val})")
                        pyautogui.doubleClick(int(x + pos[0]), int(y + pos[1]))
                        self.home_state[suit] = val
                        self.column_memory[src_col].pop()
                        field[src_col].pop()
                self.last_move = move
                self.move_history.append(move)
                if len(self.move_history) > 4:
                    self.move_history.pop(0)
                self.last_action_was_draw = False
                self.last_king_move = None
                self.stuck_ticks = max(0, self.stuck_ticks - 1)
                time.sleep(2.0)
            else:
                print("\U0001f504 Листаем колоду")
                self.last_action_was_draw = True
                self.click_stock(x, y)
                self.stuck_ticks += 1
                if self.stuck_ticks >= self.max_stuck_ticks:
                    self.handle_stuck(x, y)
                time.sleep(1.0)

        self.root.deiconify()

    def get_ui_point(self, name):
        presets = {
            'new_game': (0.05, 0.045),
            'restart': (0.16, 0.045),
            'undo': (0.28, 0.045),
            'stock': (0.305, 0.185),
        }
        px, py = presets[name]
        return int(px * self.img_width), int(py * self.img_height)

    def click_stock(self, ox, oy):
        sx, sy = self.get_ui_point('stock')
        dx, dy = self.stock_probe_offsets[self.stock_probe_idx]
        pyautogui.moveTo(int(ox + sx + dx), int(oy + sy + dy))
        pyautogui.click()

    def observe_draw_result(self, deck_card):
        current_name = deck_card[0] if deck_card else ''
        if self.last_action_was_draw:
            if current_name == self.last_draw_deck_name:
                self.failed_draws += 1
                if self.failed_draws >= 2:
                    self.stock_probe_idx = (self.stock_probe_idx + 1) % len(self.stock_probe_offsets)
                    self.failed_draws = 0
                    print(f"🎯 Сдвиг точки клика по колоде: probe #{self.stock_probe_idx + 1}")
            else:
                self.failed_draws = 0
        self.last_draw_deck_name = current_name

    def click_control(self, ox, oy, control):
        cx, cy = self.get_ui_point(control)
        pyautogui.click(int(ox + cx), int(oy + cy))

    def update_stock_click(self, deck_card):
        # Keep stock click fixed for this layout; dynamic offset caused misses.
        return

    def register_progress(self):
        foundation_total = sum(self.home_state.values())
        if foundation_total > self.last_foundation_total:
            self.last_foundation_total = foundation_total
            self.stuck_ticks = 0
            self.undo_attempts = 0
            self.restart_count = 0

    def reset_round_state(self):
        self.last_move = None
        self.last_king_move = None
        self.move_history.clear()
        self.column_memory = {i: [] for i in range(1, 8)}
        self.home_state = {'h': 0, 'b': 0, 'k': 0, 'p': 0}
        self.stuck_ticks = 0
        self.last_foundation_total = 0
        self.last_draw_deck_name = None
        self.last_action_was_draw = False
        self.failed_draws = 0
        self.stock_probe_idx = 0

    def handle_stuck(self, ox, oy, hard=False):
        self.stuck_ticks = 0
        if self.undo_attempts < self.max_undo_attempts and not hard:
            self.undo_attempts += 1
            print(f"↩ Undo ход ({self.undo_attempts}/{self.max_undo_attempts})")
            self.click_control(ox, oy, 'undo')
            time.sleep(0.8)
            return

        self.undo_attempts = 0
        self.restart_count += 1
        if self.restart_count <= 3:
            print("♻ Restart текущей партии")
            self.click_control(ox, oy, 'restart')
        else:
            print("🎲 Новая игра")
            self.restart_count = 0
            self.click_control(ox, oy, 'new_game')
        time.sleep(1.0)
        self.reset_round_state()

    def update_home_state_from_homes(self, homes):
        for card in homes:
            if card:
                val = self.get_value(card)
                suit = card[-1]
                if val > self.home_state[suit]:
                    self.home_state[suit] = val

    def try_home_drop_deck(self, deck_card, ox, oy):
        if not deck_card:
            return False

        card_name, pos = deck_card
        val = self.get_value(card_name)
        suit = card_name[-1]
        if val == self.home_state[suit] + 1:
            print(f"\U0001f4a1 В домик из колоды: {card_name} ({val})")
            pyautogui.doubleClick(ox + pos[0], oy + pos[1])
            self.home_state[suit] = val
            return True
        return False

    def try_home_drop(self, field, ox, oy):
        for col, cards in field.items():
            if not cards:
                continue
            top_name, _, _ = cards[-1]
            val = self.get_value(top_name)
            suit = top_name[-1]
            if val == self.home_state[suit] + 1:
                pos = self.estimate_card_position(col, cards)
                print(f"\U0001f4a1 В домик: {top_name} ({val})")
                pyautogui.doubleClick(ox + pos[0], oy + pos[1])
                self.home_state[suit] = val
                if self.column_memory.get(col):
                    self.column_memory[col].pop()
                return True
        return False

    def try_king_to_empty(self, field, ox, oy):
        empty_cols = [col for col in range(1, 8) if not field.get(col)]
        if not empty_cols:
            return False

        for col, cards in field.items():
            if cards and cards[-1][0].startswith("k"):
                target_col = empty_cols[0]
                if self.last_king_move == (target_col, col):
                    continue

                src_pos = self.estimate_card_position(col, cards)
                dst_x = self.column_x.get(target_col, self.margin + (target_col - 1) * self.spacing)
                dst_y = self.tableau_base_y + 10
                if src_pos[1] < 200:
                    continue
                print(f"King move: column {col} -> {target_col}")
                pyautogui.moveTo(ox + src_pos[0], oy + src_pos[1])
                pyautogui.dragTo(ox + dst_x, oy + dst_y, duration=0.4)
                if self.column_memory.get(col):
                    card = self.column_memory[col].pop()
                    self.column_memory[target_col].append(card)
                self.last_king_move = (col, target_col)
                return True
        return False
    def match_all_cards(self, gray_img):
        matches = []
        for filename in os.listdir(TEMPLATE_FOLDER):
            if not filename.endswith(".png"):
                continue
            card_name = filename.replace(".png", "")
            template = cv2.imread(os.path.join(TEMPLATE_FOLDER, filename), 0)
            w_t, h_t = template.shape[::-1]
            res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= THRESHOLD)
            for pt in zip(*loc[::-1]):
                score = res[pt[1], pt[0]]
                matches.append((pt[0], pt[1], w_t, h_t, card_name, score))
        return matches

    def filter_duplicates(self, matches):
        unique = []
        for m in matches:
            found = False
            for i, u in enumerate(unique):
                if abs(m[0] - u[0]) < 10 and abs(m[1] - u[1]) < 10:
                    found = True
                    if m[5] > u[5]:
                        unique[i] = m
                    break
            if not found:
                unique.append(m)
        return [(x, y, w, h, name) for x, y, w, h, name, _ in unique]

    def build_field_structure_dynamic(self, matches, img_width):
        columns = defaultdict(list)
        self.margin = int(img_width * TABLEAU_LEFT_RATIO)
        right_bound = int(img_width * TABLEAU_RIGHT_RATIO)
        self.spacing = max(40, (right_bound - self.margin) // 6)
        tableau_top = max(180, int(self.img_height * 0.28))
        self.tableau_base_y = max(int(self.img_height * 0.50), tableau_top + 120)

        for x, y, w, h, name in matches:
            if y < tableau_top:
                continue
            cx = x + w // 2
            cy = y + h // 2

            col_idx = round((cx - self.margin) / self.spacing) + 1
            col_idx = max(1, min(7, col_idx))
            columns[col_idx].append((y, name, cx, cy))

        self.column_x = {}
        for i in range(1, 8):
            if columns.get(i):
                xs = [cx for _, _, cx, _ in columns[i]]
                self.column_x[i] = int(statistics.median(xs))
            else:
                self.column_x[i] = self.margin + (i - 1) * self.spacing

        field = {}
        for i in range(1, 8):
            if columns.get(i):
                sorted_cards = [
                    (name, cx, cy) for y, name, cx, cy in sorted(columns[i])
                ]
                field[i] = sorted_cards
                names = [c[0] for c in sorted_cards]
                if len(self.column_memory[i]) >= len(names):
                    self.column_memory[i] = self.column_memory[i][:-len(names)] + names
                else:
                    self.column_memory[i] = names
            else:
                if self.column_memory[i]:
                    x = self.column_x.get(i, self.margin + (i - 1) * self.spacing)
                    field[i] = [
                        (name, x, self.tableau_base_y + idx * 35)
                        for idx, name in enumerate(self.column_memory[i])
                    ]
                else:
                    field[i] = []
        return field

    def detect_deck_card(self, matches, img_width=None, img_height=None):
        img_width = img_width or self.img_width
        img_height = img_height or self.img_height
        top_min = int(img_height * TOP_ROW_MIN_RATIO)
        top_max = int(img_height * TOP_ROW_MAX_RATIO)
        waste_left = int(img_width * WASTE_LEFT_RATIO)
        waste_right = int(img_width * WASTE_RIGHT_RATIO)
        waste_center = int((waste_left + waste_right) / 2)

        candidates = []
        for x, y, w, h, name in matches:
            cy = y + h // 2
            if cy < top_min or cy > top_max:
                continue
            cx = x + w // 2
            if waste_left <= cx <= waste_right:
                candidates.append((abs(cx - waste_center), name, (cx, cy)))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1], candidates[0][2]
        return None

    def detect_home_cards(self, matches, img_width=None, img_height=None):
        img_width = img_width or self.img_width
        img_height = img_height or self.img_height
        top_min = int(img_height * TOP_ROW_MIN_RATIO)
        top_max = int(img_height * TOP_ROW_MAX_RATIO)
        homes = [''] * 4
        top = []
        for x, y, w, h, name in matches:
            cy = y + h // 2
            if top_min <= cy <= top_max:
                top.append((x, y, w, h, name))
        if not top:
            return homes

        right_zone = [m for m in top if m[0] > int(img_width * FOUNDATION_LEFT_RATIO)]
        right_zone.sort(key=lambda m: m[0])

        for i, card in enumerate(right_zone[:4]):
            homes[i] = card[4]

        return homes

    def get_color(self, card):
        return 'black' if card[-1] in ('p', 'k') else 'red'

    def get_value(self, card):
        order = {'t': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                 '7': 7, '8': 8, '9': 9, '10': 10, 'v': 11, 'd': 12, 'k': 13}
        for key in order:
            if card.startswith(key): return order[key]
        return 0

    def is_valid_stack(self, cards):
        for i in range(len(cards) - 1):
            c1, c2 = cards[i], cards[i + 1]
            if self.get_color(c1) == self.get_color(c2):
                return False
            if self.get_value(c1) != self.get_value(c2) + 1:
                return False
        return True

    def find_move(self, field, deck_card):
        field_tops = [(cards[-1][0], col) for col, cards in field.items() if cards]

        for col, cards in field.items():
            names = self.column_memory.get(col, [])
            for idx in range(len(names)):
                subseq = names[idx:]
                if not self.is_valid_stack(subseq):
                    continue
                card1 = subseq[0]

                for card2, col2 in field_tops:
                    if col == col2:
                        continue
                    if (
                        self.get_color(card1) != self.get_color(card2)
                        and self.get_value(card1) + 1 == self.get_value(card2)
                    ):
                        actual_idx = len(field[col]) - len(names) + idx
                        if actual_idx < 0 or actual_idx >= len(field[col]):
                            continue
                        src_pos = self.estimate_card_position(col, field[col], True, actual_idx)
                        dst_pos = self.estimate_card_position(col2, field[col2], False)
                        return (card1, src_pos, dst_pos)

                for col2 in range(1, 8):
                    if col2 == col:
                        continue
                    if not field.get(col2):
                        if self.get_value(card1) == 13:
                            actual_idx = len(field[col]) - len(names) + idx
                            if actual_idx < 0 or actual_idx >= len(field[col]):
                                continue
                            src_pos = self.estimate_card_position(col, field[col], True, actual_idx)
                            dst_pos = self.estimate_card_position(col2, field[col2], False)
                            return (card1, src_pos, dst_pos)

        if deck_card:
            card1, pos = deck_card
            for col2, cards in field.items():
                if not cards:
                    if self.get_value(card1) == 13:
                        dst_pos = self.estimate_card_position(col2, [], False)
                        return (card1, pos, dst_pos)
                    continue
                card2 = cards[-1][0]
                if (
                    self.get_color(card1) != self.get_color(card2)
                    and self.get_value(card1) + 1 == self.get_value(card2)
                ):
                    dst_pos = self.estimate_card_position(col2, cards, False)
                    return (card1, pos, dst_pos)
        return None


    def is_repeating_move(self, move):
        if not self.move_history:
            return False

        card, src, dst = move
        for prev_card, prev_src, prev_dst in self.move_history[-4:]:
            if move == (prev_card, prev_src, prev_dst):
                return True
            if card == prev_card and src == prev_dst and dst == prev_src:
                return True
        return False

    def column_by_position(self, field, pos, is_source=True):
        for col, cards in field.items():
            card_list = cards if cards else [ (name, 0, 0) for name in self.column_memory.get(col, []) ]
            for idx in range(len(card_list)):
                if self.estimate_card_position(col, cards, is_source, idx) == pos:
                    return col, idx
        return None, None

    def estimate_card_position(self, col, cards, is_source=True, index=None, top_only=False):
        if cards and len(cards) > 0:
            if index is None:
                index = len(cards) - 1
            _, x, y = cards[index]
            if is_source and top_only and index == len(cards) - 1:
                return (x, y - TOP_CARD_OFFSET)
            return (x, y)

        if self.column_memory.get(col):
            if index is None:
                index = len(self.column_memory[col]) - 1
            x = self.column_x.get(col, self.margin + (col - 1) * self.spacing)
            y = self.tableau_base_y + index * 35
            if is_source and top_only and index == len(self.column_memory[col]) - 1:
                y -= TOP_CARD_OFFSET
            return (x, y)

        x = self.column_x.get(col, self.margin + (col - 1) * self.spacing)
        return (x, self.tableau_base_y)

    def print_field_state(self, deck_card, homes, field):
        print("\n====== Состояние поля ======")
        print(f"Колода: {deck_card[0] if deck_card else 'пусто'}", end=' | ')
        print("Сброс:", " | ".join(card if card else "_" for card in homes))
        for i in range(1, 8):
            col_cards = field.get(i, [])
            names = [c[0] for c in col_cards]
            print(f"{i}: {', '.join(names) if names else 'пусто'}")
        print("============================\n")
        print("📦 Память по колонкам:")
        for i in range(1, 8):
            mem = self.column_memory.get(i, [])
            print(f"  {i}: {' -> '.join(mem) if mem else 'пусто'}")

    def can_place(self, card, target_stack):
        if not target_stack:
            return self.get_value(card) == 13
        top_card = target_stack[-1][0]
        return (
            self.get_color(card) != self.get_color(top_card)
            and self.get_value(card) + 1 == self.get_value(top_card)
        )

if __name__ == "__main__":
    KosynkaBot()


