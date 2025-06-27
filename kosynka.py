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
THRESHOLD = 0.88
TOP_CARD_OFFSET = 50

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

            matches = self.match_all_cards(gray_field)
            matches = self.filter_duplicates(matches)
            field = self.build_field_structure_dynamic(matches, field_img.shape[1])
            deck_card = self.detect_deck_card(matches)
            homes = self.detect_home_cards(matches)

            self.update_home_state_from_homes(homes)
            self.print_field_state(deck_card, homes, field)

            if self.try_home_drop_deck(deck_card, x, y):
                time.sleep(1.0)
                continue

            if self.try_home_drop(field, x, y):
                time.sleep(1.0)
                continue

            if self.try_king_to_empty(field, x, y):
                time.sleep(1.0)
                continue

            move = self.find_move(field, deck_card)
            if move and move != self.last_move:
                if self.is_repeating_move(move):
                    print("\U0001f501 Обнаружен зацикливающийся ход, остановка")
                    self.running = False
                    break
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
                time.sleep(2.0)
            else:
                print("\U0001f504 Листаем колоду")
                pyautogui.click(x + 60, y + 60)
                time.sleep(1.0)

        self.root.deiconify()

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
                src_pos = self.estimate_card_position(col, cards)
                dst_x = self.column_x.get(empty_cols[0], self.margin + (empty_cols[0] - 1) * self.spacing)
                dst_y = 250 + 10
                if src_pos[1] < 200:
                    continue
                print(f"\U0001f451 Перемещаем короля из колонки {col} в {empty_cols[0]}")
                pyautogui.moveTo(ox + src_pos[0], oy + src_pos[1])
                pyautogui.dragTo(ox + dst_x, oy + dst_y, duration=0.4)
                if self.column_memory.get(col):
                    card = self.column_memory[col].pop()
                    self.column_memory[empty_cols[0]].append(card)
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
        self.margin = 30
        self.spacing = (img_width - 2 * self.margin) // 6
        column_ranges = [
            (
                self.margin + i * self.spacing - self.spacing // 2,
                self.margin + i * self.spacing + self.spacing // 2,
            )
            for i in range(7)
        ]

        for x, y, w, h, name in matches:
            if y < 200:
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
                        (name, x, 260 + idx * 35)
                        for idx, name in enumerate(self.column_memory[i])
                    ]
                else:
                    field[i] = []
        return field

    def detect_deck_card(self, matches):
        for x, y, w, h, name in matches:
            if y > 200:
                continue
            cx = x + w // 2
            if 160 < cx < 280:
                return (name, (cx, y + h // 2))
        return None

    def detect_home_cards(self, matches):
        homes = [''] * 4
        top = [(x, y, w, h, name) for x, y, w, h, name in matches if y < 200]
        if not top:
            return homes

        right_zone = [m for m in top if m[0] > 300]
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
            y = 260 + index * 35
            if is_source and top_only and index == len(self.column_memory[col]) - 1:
                y -= TOP_CARD_OFFSET
            return (x, y)

        x = self.column_x.get(col, self.margin + (col - 1) * self.spacing)
        return (x, 260)

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