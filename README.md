# Kosynka RL Bot

Автоматический бот для **Косынки (Klondike)**:
- играет через GUI (распознавание карт на экране + перетаскивания);
- обучается в симуляторе через **Q-learning** и **DQN**.

## Возможности

- GUI-бот для реальной игры (`kosynka.py`)
- RL-обучение в симуляции (`learning_kosynka.py`)
- Q-learning (таблица)
- DQN (PyTorch)
- Запуск на CPU или GPU (`--device auto|cpu|cuda` для DQN)
- Авто-настройка окружения на Ubuntu скриптом `setup_ubuntu_training.sh`

---

## Быстрый старт (локально)

```bash
pip install -r requirements.txt
```

### Запуск GUI-бота

```bash
python kosynka.py
```

Управление:
- `Space` — старт/пауза
- `Esc` — остановка

---

## Обучение

### Q-learning

```bash
python learning_kosynka.py --train --episodes 5000 --model q_model.pkl
```

### DQN

```bash
python learning_kosynka.py --train --dqn --episodes 5000 --model dqn_model.pt --device auto
```

Параметры:
- `--device auto` — использовать CUDA при доступности, иначе CPU
- `--device cuda` — принудительно GPU
- `--device cpu` — принудительно CPU
- `--verbose` — подробный вывод

---

## Игра обученной моделью

### Q-model

```bash
python learning_kosynka.py --play --model q_model.pkl
```

### DQN-model

```bash
python learning_kosynka.py --play --dqn --model dqn_model.pt --device auto
```

---

## Ubuntu

Для переноса обучения на Ubuntu используйте:

```bash
chmod +x setup_ubuntu_training.sh
./setup_ubuntu_training.sh --device auto
source .venv/bin/activate
python learning_kosynka.py --train --dqn --episodes 5000 --model dqn_model.pt --device auto
```

---

## Проверка GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Если `torch.cuda.is_available()` возвращает `True`, DQN может обучаться на видеокарте.

---

## Структура

```text
.
├── kosynka.py                  # GUI-бот для реальной игры
├── learning_kosynka.py         # RL: Q-learning + DQN
├── setup_ubuntu_training.sh    # Авто-setup для Ubuntu (CPU/GPU)
├── requirements.txt
└── templates/
```

---

## Важно

- Обучение идет в **упрощенной симуляции**, а не напрямую в GUI.
- Качество игры в реальном UI сильно зависит от корректного распознавания поля и карт.
- Q-learning почти не ускоряется на GPU; GPU в первую очередь нужен для DQN.
