import gymnasium as gym
import numpy as np
import random
import time

# Создание среды Taxi-v3
env = gym.make('Taxi-v3', render_mode='ansi')

# Инициализация переменных для хранения информации
num_actions = 10
frames = []

# Запуск 10 случайных действий в среде
for _ in range(num_actions):
    # Сброс среды для получения начального состояния
    state = env.reset()
    
    # Случайный выбор действия из доступного пространства действий
    action = env.action_space.sample()
    
    # Выполнение действия и получение новой информации о состоянии
    new_state, reward, terminated, info, done = env.step(action)
    
    # Сохранение кадра для отображения
    frames.append({
        'frame': env.render(),
        'state': new_state,
        'action': action,
        'reward': reward,
        'timestep': len(frames)  # Значение временного шага
    })
    
    # Пауза для визуализации (опционально)
    # time.sleep(0.5)

# Закрытие среды после завершения действий
env.close()

# Вывод информации о каждом действии
for frame in frames:
    print(frame['frame'])
    print(f"Timestep: {frame['timestep']}")
    print(f"State: {frame['state']}")
    print(f"Action: {frame['action']}")
    print(f"Reward: {frame['reward']}")