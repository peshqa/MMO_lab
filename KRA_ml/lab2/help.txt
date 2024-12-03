import sys, os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def initialize_policy(n_states, n_actions):
 policy = np.ones((n_states, n_actions)) / n_actions
 return policy

def generate_session(env, policy, t_max=10**4):
 """
 Играть до конца или t_max тиков.
 :param policy: массив вида [n_states,n_actions] с вероятностями действий
 :returns: список состояний, список действий и сумма наград
 """
 states, actions = [], []
 total_reward = 0.
 s = env.reset()
 for t in range(t_max):
  # Hint: вы можете использовать np.random.choice для выборки
  # https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
  #print(s[0], s==6, "!!!!")
  a = np.random.choice(len(policy[s[0]]), p=policy[s[0]])
  new_s, r, terminated, info, done = env.step(a)
  # Запись информацию, которая получена из среды.
  states.append(s[0])
  actions.append(a)
  total_reward += r
  s = new_s
  if done or terminated:
   break
 #print(states)
 return states, actions, total_reward
 
def select_elites(states_batch, actions_batch, rewards_batch, percentile):
 """
 Выберите состояния и действия из игры, которые имеют награды >= процентиль
 :paramstates_batch: список списков состояний,states_batch[session_i][t]
 :param action_batch: список списков действий, action_batch[session_i][t]
 :param rewards_batch: список наград, rewards_batch[session_i]
 :returns: elite_states,elite_actions, одномерные списки состояний и соответствующих действий лучших сессий
 """
 reward_threshold = np.percentile(rewards_batch, percentile)
 # Hint: используйте np.percentile()>
 elite_states = []
 elite_actions = []
 for i in range(len(rewards_batch)):
  if rewards_batch[i] >= reward_threshold:
   elite_states.extend(states_batch[i])
   elite_actions.extend(actions_batch[i])
 #print(elite_states)
 return elite_states, elite_actions
 
states_batch = [
 [1, 2, 3], # игра1
 [4, 2, 0, 2], # игра2
 [3, 1], # игра3
]
actions_batch = [
 [0, 2, 4], # игра1
 [3, 2, 0, 1], # игра2
 [3, 3], # игра3
]
rewards_batch = [
 3, # игра1
 4, # игра2
 5, # игра3
]

test_result_0 = select_elites(states_batch, actions_batch, rewards_batch, percentile=0)
test_result_30 = select_elites(states_batch, actions_batch, rewards_batch, percentile=30)
test_result_90 = select_elites(states_batch, actions_batch, rewards_batch, percentile=90)
test_result_100 = select_elites(states_batch, actions_batch, rewards_batch, percentile=100)
assert np.all(test_result_0[0] == [1, 2, 3, 4, 2, 0, 2, 3, 1]) \
 and np.all(test_result_0[1] == [0, 2, 4, 3, 2, 0, 1, 3, 3]), \
 "Для процентиля 0 вы должны вернуть все состояния и действия в хронологическом порядке."
assert np.all(test_result_30[0] == [4, 2, 0, 2, 3, 1]) and \
 np.all(test_result_30[1] == [3, 2, 0, 1, 3, 3]), \
 "Для процентиля 30 вы должны выбрать состояния/действия только из двух первых"
assert np.all(test_result_90[0] == [3, 1]) and \
 np.all(test_result_90[1] == [3, 3]), \
 "Для процентиля 90 вы должны выбирать состояния/действия только из одной игры."
assert np.all(test_result_100[0] == [3, 1]) and \
 np.all(test_result_100[1] == [3, 3]), \
 "Убедитесь, что вы используете >=, а не >. Также дважды проверьте, как вы вычисляете процентиль."
def get_new_policy(elite_states, elite_actions):
 """
 Учитывая список лучших состояний/действий от select_elites,
 возвращает новую политику, где вероятность каждого действия пропорциональна
 policy[s_i,a_i] ~ #[появления s_i и a_i в элитарных состояниях/действиях]
 Не забудьте нормализовать политику, чтобы получить действительные вероятности и обработать случай 0/0.
 Для состояний, в которых вы никогда не находились, используйте равномерное распределение (1/n_actions для всех состояний).
 :param Elite_states: одномерный список состояний лучших сессий.
 :param Elite_actions: одномерный список действий лучших сессий.
 """
 from collections import defaultdict
 #print(elite_states)
 n_states = 500#max(elite_states) + 1  # Предполагаем, что состояния начинаются с 0
 n_actions = 6#max(elite_actions) + 1  # Предполагаем, что действия начинаются с 0
 
 new_policy = np.zeros([n_states, n_actions])
 
 # Подсчет появлений состояний и действий
 counts = defaultdict(lambda: defaultdict(int))
    
 for s, a in zip(elite_states, elite_actions):
  counts[s][a] += 1
    
    # Установка вероятностей для действий в элитарных состояниях
 for s in range(n_states):
  total_count = sum(counts[s].values())
  if total_count > 0:
   for a in range(n_actions):
    new_policy[s][a] = counts[s][a] / total_count
  else:
   # Если состояние не встречалось в элитах, используем равномерное распределение
   new_policy[s] = np.ones(n_actions) / n_actions
 # Не забыть выставить 1/n_действий для всех действий в неизвестных состояниях.
 return new_policy
 

def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10], show=False):
 """
 Удобная функция, отображающая прогресс обучения
 """
 mean_reward = np.mean(rewards_batch)
 threshold = np.percentile(rewards_batch, percentile)
 log.append([mean_reward, threshold])
 if show:
  plt.figure(figsize=[8, 4])
  plt.subplot(1, 2, 1)
  plt.plot(list(zip(*log))[0], label='Mean rewards')
  plt.plot(list(zip(*log))[1], label='Reward thresholds')
  plt.legend()
  plt.grid()
  plt.subplot(1, 2, 2)
  plt.hist(rewards_batch, range=reward_range)
  plt.vlines([np.percentile(rewards_batch, percentile)], [0], [100], label="percentile", color='red')
  plt.legend()
  plt.grid()
  clear_output(True)
  print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
  plt.show()
  


env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()
env.render()
n_states = env.observation_space.n
n_actions = env.action_space.n
print("n_states=%i, n_actions=%i" % (n_states, n_actions))

# сбросить политику на всякий случай
policy = initialize_policy(n_states, n_actions)
#Эксперимент
n_sessions = 250 # число сессий
percentile = 50 # процент сессий с наивысшей наградой
learning_rate = 0.5 # насколько быстро обновляется политика, по шкале от 0 до 1
log = []
for i in range(100):
 sessions = [generate_session(env, policy) for _ in range(n_sessions)]
 states_batch, actions_batch, rewards_batch = zip(*sessions)
 #print(rewards_batch)
 elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
 #print(elite_states)
 new_policy = get_new_policy(elite_states, elite_actions)
 policy = learning_rate * new_policy + (1 - learning_rate) * policy
 # display results on chart
 show = i == 99
 show_progress(rewards_batch, log, percentile, show=show) 



