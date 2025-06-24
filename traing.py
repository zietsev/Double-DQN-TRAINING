# import gym
# from gym import spaces
# import pygame
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import deque
# import random
# import pandas as pd
# import matplotlib.pyplot as plt

# # Auto-select device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * input_shape[1] * input_shape[2], 512),
#             nn.ReLU(),
#             nn.Linear(512, n_actions)
#         )

#     def forward(self, x):
#         return self.net(x)

# class DoubleDQNAgent:
#     def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64, target_update=1000):
#         self.n_actions = n_actions
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.target_update = target_update

#         self.online_net = DQN(input_shape, n_actions).to(device)
#         self.target_net = DQN(input_shape, n_actions).to(device)
#         self.target_net.load_state_dict(self.online_net.state_dict())
#         self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
#         self.memory = deque(maxlen=buffer_size)
#         self.step_count = 0

#     def select_action(self, state, epsilon):
#         if random.random() < epsilon:
#             return random.randrange(self.n_actions)
#         state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#         with torch.no_grad():
#             q_values = self.online_net(state_t)
#         return q_values.argmax().item()

#     def store_transition(self, transition):
#         self.memory.append(transition)

#     def update(self):
#         if len(self.memory) < self.batch_size:
#             return
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)

#         states = torch.from_numpy(np.array(states)).float().to(device)
#         actions = torch.tensor(actions, device=device)
#         rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
#         next_states = torch.from_numpy(np.array(next_states)).float().to(device)
#         dones = torch.tensor(dones, dtype=torch.float32, device=device)

#         q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#         next_actions = self.online_net(next_states).argmax(1)
#         next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
#         targets = rewards + self.gamma * next_q_values * (1 - dones)

#         loss = nn.MSELoss()(q_values, targets.detach())
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.step_count += 1
#         if self.step_count % self.target_update == 0:
#             self.target_net.load_state_dict(self.online_net.state_dict())
# class MultiAgentGridEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

#     def __init__(self, render_mode=None, size=15, total_episodes=5000):
#         self.size = size
#         self.window_size = 540
#         self.render_mode = render_mode
#         self.total_episodes = total_episodes
#         self.window = None
#         self.clock = None
        
#         self.episode_count = 0
#         self.current_epsilon = 1.0
#         self.battery_bar_height = 15
        
#         # Battery + cost parameters
#         self.max_battery = 1.0
#         self.battery_decay_rate = 0.0015
#         self.action_cost = 0.001
#         self.k_wind = 0.01
#         self.task_cost = 0.0  # dynamically add if task needed

#         self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))
#         self.observation_space = spaces.Tuple([
#             spaces.Box(low=0, high=1, shape=(7, size, size), dtype=np.float32),
#             spaces.Box(low=0, high=1, shape=(7, size, size), dtype=np.float32)
#         ])

#         self._action_to_direction = {
#             0: np.array([1, 0]),
#             1: np.array([0, 1]),
#             2: np.array([-1, 0]),
#             3: np.array([0, -1])
#         }

#         self._generate_static_elements()
#         self.total_victims = int(np.sum(self.fixed_victims))
#     def _generate_static_elements(self):
#         total_cells = self.size * self.size
#         num_obstacles = int(0.10 * total_cells)
#         num_victims = int(0.08 * total_cells)

#         self.fixed_obstacles = np.zeros((self.size, self.size))
#         placed = 0
#         while placed < num_obstacles:
#             x, y = np.random.randint(0, self.size, 2)
#             if (x, y) not in [(0,0), (self.size-1, self.size-1)] and self.fixed_obstacles[y,x] == 0:
#                 self.fixed_obstacles[y,x] = 1
#                 placed += 1

#         self.fixed_victims = np.zeros((self.size, self.size))
#         placed = 0
#         while placed < num_victims:
#             x, y = np.random.randint(0, self.size, 2)
#             if (x, y) not in [(0,0), (self.size-1, self.size-1)] and \
#                 self.fixed_obstacles[y,x] == 0 and self.fixed_victims[y,x] == 0:
#                 self.fixed_victims[y,x] = 1
#                 placed += 1

#         directions = list(self._action_to_direction.values())
#         self.fixed_wind = {}
#         placed = 0
#         while placed < 10:
#             x, y = np.random.randint(0, self.size, 2)
#             if (x,y) not in self.fixed_wind and self.fixed_obstacles[y,x] == 0:
#                 self.fixed_wind[(x,y)] = {
#                     "dir": directions[np.random.choice(4)],
#                     "speed": np.random.uniform(0.2, 1.0)
#                 }
#                 placed += 1

#     def _initialize_maps(self):
#         self.agent1_pos = np.array([0, 0])
#         self.agent2_pos = np.array([self.size-1, self.size-1])

#         self.coverage_map_1 = np.zeros((self.size, self.size))
#         self.coverage_map_2 = np.zeros((self.size, self.size))
#         self.victim_map = self.fixed_victims.copy()
#         self.victim_found_map_1 = np.zeros((self.size, self.size))
#         self.victim_found_map_2 = np.zeros((self.size, self.size))
#         self.risk_map = np.random.uniform(0, 1, (self.size, self.size))
#         self.battery_map_1 = -np.ones((self.size, self.size))
#         self.battery_map_2 = -np.ones((self.size, self.size))
#         self.time_map = -np.ones((self.size, self.size))

#         self.obstacle_map = self.fixed_obstacles.copy()
#         self.wind_map = self.fixed_wind.copy()

#         self.battery_1 = self.max_battery
#         self.battery_2 = self.max_battery
#         self.step_count = 0
#         self.total_reward_1 = 0
#         self.total_reward_2 = 0
#         self.victims_found_1 = 0
#         self.victims_found_2 = 0

#     def _move_agent(self, agent_id, action):
#         if (agent_id == 1 and self.battery_1 <= 0) or (agent_id == 2 and self.battery_2 <= 0):
#             return 0
        
#         pos = self.agent1_pos if agent_id == 1 else self.agent2_pos
#         move = self._action_to_direction[action]
#         next_pos = pos + move
#         if np.any(next_pos < 0) or np.any(next_pos >= self.size) or self.obstacle_map[next_pos[1], next_pos[0]] == 1:
#             next_pos = pos

#         wind = self.wind_map.get(tuple(pos), {"dir": np.array([0, 0]), "speed": 0})
#         wind_cost = self.k_wind * wind["speed"]
#         task_cost = self.task_cost  # dynamically set later if rescue happens
#         step_cost = self.battery_decay_rate + self.action_cost + wind_cost + task_cost + np.random.normal(0, 0.001)

#         if agent_id == 1:
#             self.battery_1 -= step_cost
#             self.agent1_pos = next_pos
#             coverage_map = self.coverage_map_1
#             battery_map = self.battery_map_1
#             found_map = self.victim_found_map_1
#         else:
#             self.battery_2 -= step_cost
#             self.agent2_pos = next_pos
#             coverage_map = self.coverage_map_2
#             battery_map = self.battery_map_2
#             found_map = self.victim_found_map_2

#         y, x = next_pos[1], next_pos[0]
#         reward = 0
#         if coverage_map[y, x] == 0:
#             reward += 1
#         if self.victim_map[y, x] == 1 and found_map[y, x] == 0:
#             found_map[y, x] = 1
#             reward += 2
#             if agent_id == 1:
#                 self.victims_found_1 += 1
#             else:
#                 self.victims_found_2 += 1
#             # Apply task cost when rescue happens
#             self.task_cost = 0.0005
#         else:
#             self.task_cost = 0  # reset

#         if self.risk_map[y, x] > 0.7:
#             reward -= 0.02

#         coverage_map[y, x] = 1
#         battery_map[y, x] = self.battery_1 if agent_id == 1 else self.battery_2
#         self.time_map[y, x] = self.step_count
#         if agent_id == 1:
#             self.total_reward_1 += reward
#         else:
#             self.total_reward_2 += reward

#         return reward
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self._initialize_maps()
#         if self.render_mode == "human":
#             self._render_frame()
#         return (
#             self._get_obs_for(self.agent1_pos, self.coverage_map_1, self.battery_map_1),
#             self._get_obs_for(self.agent2_pos, self.coverage_map_2, self.battery_map_2)
#         )
#     def step(self, actions):
#         reward_1 = self._move_agent(1, actions[0])
#         reward_2 = self._move_agent(2, actions[1])
#         self.step_count += 1
#         done = (self.battery_1 <= 0 and self.battery_2 <= 0)
#         if self.render_mode == "human":
#             self._render_frame()
#         return (
#             self._get_obs_for(self.agent1_pos, self.coverage_map_1, self.battery_map_1),
#             self._get_obs_for(self.agent2_pos, self.coverage_map_2, self.battery_map_2)
#         ), (reward_1, reward_2), done, False, {}

#     def _get_obs_for(self, agent_pos, coverage_map, battery_map):
#         agent_map = np.zeros((self.size, self.size))
#         agent_map[agent_pos[1], agent_pos[0]] = 1
#         return np.stack([
#             agent_map,
#             coverage_map,
#             self.victim_map,
#             self.risk_map,
#             self.obstacle_map,
#             np.clip(battery_map, 0, 1),
#             np.clip(self.time_map / 1000, 0, 1)
#         ])
#     def _render_frame(self):
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.window_size + 60))
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()

#         canvas = pygame.Surface((self.window_size, self.window_size + 60))
#         canvas.fill((255, 255, 255))
#         pix = self.window_size // self.size
#         grid_offset = 40

#         pygame.draw.rect(canvas, (230, 230, 230), (0, 0, self.window_size, 40))
#         font = pygame.font.SysFont('Arial', 16)
#         ep_text = font.render(f"Ep: {self.episode_count}/{self.total_episodes}", True, (0, 0, 0))
#         canvas.blit(ep_text, (10, 10))
#         victims_text = font.render(f"Agent 1: {self.victims_found_1} | Agent 2: {self.victims_found_2}", True, (0, 0, 0))
#         canvas.blit(victims_text, (self.window_size // 2 - 70, 10))
#         batt_text = font.render(f"Batt: {self.battery_1 * 100:.0f}% | {self.battery_2 * 100:.0f}%", True, (0, 0, 0))
#         canvas.blit(batt_text, (self.window_size - 180, 10))

#         for y in range(self.size):
#             for x in range(self.size):
#                 rect = pygame.Rect(x * pix, y * pix + grid_offset, pix, pix)
#                 pygame.draw.rect(canvas, (255, 255, 255), rect)
#                 if self.coverage_map_1[y, x] or self.coverage_map_2[y, x]:
#                     pygame.draw.rect(canvas, (200, 200, 200), rect)
#                 if self.victim_map[y, x]:
#                     color = (139, 0, 0) if (self.victim_found_map_1[y, x] or self.victim_found_map_2[y, x]) else (255, 0, 0)
#                     pygame.draw.rect(canvas, color, rect)
#                 if self.obstacle_map[y, x]:
#                     pygame.draw.rect(canvas, (50, 50, 50), rect)
#                 pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

#         for (x, y), wind in self.wind_map.items():
#             center = (x * pix + pix // 2, y * pix // 2 + grid_offset)
#             end = (center[0] + int(wind["dir"][0] * 15), center[1] + int(wind["dir"][1] * 15))
#             green_intensity = 100 + int(wind["speed"] * 155)
#             pygame.draw.line(canvas, (0, green_intensity, 0), center, end, 3)

#         def draw_agent(pos, color):
#             center = (pos[0] * pix + pix // 2, pos[1] * pix // 2 + grid_offset)
#             points = [
#                 (center[0], center[1] - pix // 3),
#                 (center[0] - pix // 4, center[1] + pix // 4),
#                 (center[0] + pix // 4, center[1] + pix // 4)
#             ]
#             pygame.draw.polygon(canvas, color, points)

#         draw_agent(self.agent1_pos, (0, 100, 255))
#         draw_agent(self.agent2_pos, (0, 200, 0))

#         bar_y = self.window_size + grid_offset - 10
#         pygame.draw.rect(canvas, (200, 200, 200), (0, bar_y, self.window_size, self.battery_bar_height))
#         avg_battery = (self.battery_1 + self.battery_2) / 2
#         pygame.draw.rect(canvas, (0, 200, 0), (0, bar_y, self.window_size * avg_battery, self.battery_bar_height))

#         self.window.blit(canvas, (0, 0))
#         pygame.event.pump()
#         pygame.display.update()
#         self.clock.tick(self.metadata["render_fps"])


# def train_and_plot(env, agent1, agent2, total_episodes, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
#     agent1_rewards, agent2_rewards = [], []
#     coverage_1, coverage_2 = [], []
#     battery_usage_1, battery_usage_2 = [], []
#     risk_exposure_1, risk_exposure_2 = [], []

#     for ep in range(total_episodes):
#         obs = env.reset()
#         env.episode_count = ep + 1
#         done = False

#         while not done:
#             state1, state2 = obs
#             action1 = agent1.select_action(state1, epsilon)
#             action2 = agent2.select_action(state2, epsilon)

#             next_obs, rewards, done, _, _ = env.step((action1, action2))

#             agent1.store_transition((state1, action1, rewards[0], next_obs[0], done))
#             agent2.store_transition((state2, action2, rewards[1], next_obs[1], done))

#             agent1.update()
#             agent2.update()

#             obs = next_obs
        
#         epsilon = max(epsilon * epsilon_decay, epsilon_min)
#         agent1_rewards.append(env.total_reward_1)
#         agent2_rewards.append(env.total_reward_2)
#         coverage_1.append(np.sum(env.coverage_map_1) / (env.size**2))
#         coverage_2.append(np.sum(env.coverage_map_2) / (env.size**2))
#         battery_usage_1.append(1 - env.battery_1 / env.max_battery)
#         battery_usage_2.append(1 - env.battery_2 / env.max_battery)
#         risk_exposure_1.append(np.mean(env.risk_map[env.coverage_map_1 == 1]) if np.any(env.coverage_map_1 == 1) else 0)
#         risk_exposure_2.append(np.mean(env.risk_map[env.coverage_map_2 == 1]) if np.any(env.coverage_map_2 == 1) else 0)
        
#         print(f"Episode {ep+1}: Agent 1: Reward={env.total_reward_1:.2f}, Victims={env.victims_found_1} | "
#               f"Agent 2: Reward={env.total_reward_2:.2f}, Victims={env.victims_found_2}")

#     # Save CSV and models
#     df = pd.DataFrame({
#         'episode': range(1, total_episodes + 1),
#         'agent1_reward': agent1_rewards,
#         'agent2_reward': agent2_rewards,
#         'coverage_1': coverage_1,
#         'coverage_2': coverage_2,
#         'battery_usage_1': battery_usage_1,
#         'battery_usage_2': battery_usage_2,
#         'risk_exposure_1': risk_exposure_1,
#         'risk_exposure_2': risk_exposure_2
#     })
#     df.to_csv("training_log.csv", index=False)
#     torch.save(agent1.online_net.state_dict(), "agent1_model.pth")
#     torch.save(agent2.online_net.state_dict(), "agent2_model.pth")

#     # Plot
#     plt.figure(figsize=(12, 8))
#     plt.subplot(2, 2, 1)
#     plt.plot(agent1_rewards, label="Agent 1")
#     plt.plot(agent2_rewards, label="Agent 2")
#     plt.title("Total Reward")
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.legend()

#     plt.subplot(2, 2, 2)
#     plt.plot(coverage_1, label="Agent 1")
#     plt.plot(coverage_2, label="Agent 2")
#     plt.title("Coverage %")
#     plt.xlabel("Episode")
#     plt.ylabel("Coverage")
#     plt.legend()

#     plt.subplot(2, 2, 3)
#     plt.plot(battery_usage_1, label="Agent 1")
#     plt.plot(battery_usage_2, label="Agent 2")
#     plt.title("Battery Usage %")
#     plt.xlabel("Episode")
#     plt.ylabel("Usage")
#     plt.legend()

#     plt.subplot(2, 2, 4)
#     plt.plot(risk_exposure_1, label="Agent 1")
#     plt.plot(risk_exposure_2, label="Agent 2")
#     plt.title("Avg Risk Exposure")
#     plt.xlabel("Episode")
#     plt.ylabel("Risk")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     total_episodes = 5000
#     env = MultiAgentGridEnv(render_mode="human", total_episodes=total_episodes)
#     obs = env.reset()
#     input_shape = obs[0].shape

#     agent1 = DoubleDQNAgent(input_shape, 4)
#     agent2 = DoubleDQNAgent(input_shape, 4)

#     epsilon = 1.0
#     epsilon_decay = 0.995
#     epsilon_min = 0.1

#     train_and_plot(env, agent1, agent2, total_episodes)
#     env.close()






import gym
from gym import spaces
import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- DQN Model ---
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * input_shape[1] * input_shape[2], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# --- Double DQN Agent ---
class DoubleDQNAgent:
    def __init__(self, input_shape, n_actions, lr=1e-4, gamma=0.99, buffer_size=10000, batch_size=64, target_update=1000):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.online_net = DQN(input_shape, n_actions)
        self.target_net = DQN(input_shape, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.step_count = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax().item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.from_numpy(np.array(states)).float()
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.from_numpy(np.array(next_states)).float()
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_actions = self.online_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

# --- Environment + training will follow ---
# Continuation: MultiAgentGridEnv with penalties and battery management
class MultiAgentGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=15, total_episodes=5000):
        self.size = size
        self.window_size = 540
        self.render_mode = render_mode
        self.total_episodes = total_episodes
        self.window = None
        self.clock = None
        
        self.episode_count = 0
        self.current_epsilon = 1.0
        self.battery_bar_height = 15
        
        self.action_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(7, size, size), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(7, size, size), dtype=np.float32)
        ])

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1])
        }

        self.max_battery = 1.0
        self.battery_decay_rate = 0.0015
        self.action_cost = 0.001
        self.k_wind = 0.01
        self.task_cost = 0.0  # dynamically add if task needed

        self._generate_static_elements()
        self.total_victims = int(np.sum(self.fixed_victims))

    def _generate_static_elements(self):
        total_cells = self.size * self.size
        num_obstacles = int(0.10 * total_cells)
        num_victims = int(0.08 * total_cells)

        # Place obstacles
        self.fixed_obstacles = np.zeros((self.size, self.size))
        placed = 0
        while placed < num_obstacles:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) not in [(0, 0), (self.size - 1, self.size - 1)] and self.fixed_obstacles[y, x] == 0:
                self.fixed_obstacles[y, x] = 1
                placed += 1

        # Place victims
        self.fixed_victims = np.zeros((self.size, self.size))
        placed = 0
        while placed < num_victims:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) not in [(0, 0), (self.size - 1, self.size - 1)] and \
               self.fixed_obstacles[y, x] == 0 and self.fixed_victims[y, x] == 0:
                self.fixed_victims[y, x] = 1
                placed += 1

        # Place wind (no overlap with obstacles or victims)
        directions = list(self._action_to_direction.values())
        self.fixed_wind = {}
        placed = 0
        while placed < 10:
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) not in self.fixed_wind and \
               self.fixed_obstacles[y, x] == 0 :
                self.fixed_wind[(x, y)] = {
                    "dir": directions[np.random.choice(4)],
                    "speed": np.random.uniform(0.2, 1.0)
                }
                placed += 1


    # More methods (_initialize_maps, _move_agent, etc.) will follow...
    def _initialize_maps(self):
        self.agent1_pos = np.array([0, 0])
        self.agent2_pos = np.array([self.size - 1, self.size - 1])

        self.coverage_map_1 = np.zeros((self.size, self.size))
        self.coverage_map_2 = np.zeros((self.size, self.size))
        self.victim_map = self.fixed_victims.copy()
        self.victim_found_map_1 = np.zeros((self.size, self.size))
        self.victim_found_map_2 = np.zeros((self.size, self.size))
        self.risk_map = np.random.uniform(0, 1, (self.size, self.size))
        self.battery_map_1 = -np.ones((self.size, self.size))
        self.battery_map_2 = -np.ones((self.size, self.size))
        self.time_map = -np.ones((self.size, self.size))

        self.obstacle_map = self.fixed_obstacles.copy()
        self.wind_map = self.fixed_wind.copy()

        self.battery_1 = self.max_battery
        self.battery_2 = self.max_battery
        self.step_count = 0
        self.total_reward_1 = 0
        self.total_reward_2 = 0
        self.victims_found_1 = 0
        self.victims_found_2 = 0

    def _move_agent(self, agent_id, action):
        if (agent_id == 1 and self.battery_1 <= 0) or (agent_id == 2 and self.battery_2 <= 0):
            return 0
        
        pos = self.agent1_pos if agent_id == 1 else self.agent2_pos
        move = self._action_to_direction[action]
        next_pos = pos + move
        if np.any(next_pos < 0) or np.any(next_pos >= self.size) or self.obstacle_map[next_pos[1], next_pos[0]] == 1:
            next_pos = pos
        
        wind = self.wind_map.get(tuple(pos), {"dir": np.array([0, 0]), "speed": 0})
        wind_cost = self.k_wind * wind["speed"]
        step_cost = self.battery_decay_rate + wind_cost + np.random.normal(0, 0.001)
        
        if agent_id == 1:
            self.battery_1 -= step_cost
            self.agent1_pos = next_pos
            coverage_map = self.coverage_map_1
            battery_map = self.battery_map_1
            found_map = self.victim_found_map_1
        else:
            self.battery_2 -= step_cost
            self.agent2_pos = next_pos
            coverage_map = self.coverage_map_2
            battery_map = self.battery_map_2
            found_map = self.victim_found_map_2
        
        y, x = next_pos[1], next_pos[0]
        reward = 0
        if coverage_map[y, x] == 0:
            reward += 1
        if self.victim_map[y, x] == 1 and found_map[y, x] == 0:
            found_map[y, x] = 1
            reward += 2
            if agent_id == 1:
                self.victims_found_1 += 1
            else:
                self.victims_found_2 += 1
        if self.risk_map[y, x] > 0.7:
            reward -= 0.02

        coverage_map[y, x] = 1
        battery_map[y, x] = self.battery_1 if agent_id == 1 else self.battery_2
        self.time_map[y, x] = self.step_count
        if agent_id == 1:
            self.total_reward_1 += reward
        else:
            self.total_reward_2 += reward
        return reward
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_maps()
        if self.render_mode == "human":
            self._render_frame()
        return (self._get_obs_for(self.agent1_pos, self.coverage_map_1, self.battery_map_1),
                self._get_obs_for(self.agent2_pos, self.coverage_map_2, self.battery_map_2))

    def step(self, actions):
        reward_1 = self._move_agent(1, actions[0])
        reward_2 = self._move_agent(2, actions[1])
        self.step_count += 1
        done = (self.battery_1 <= 0 and self.battery_2 <= 0)
        if self.render_mode == "human":
            self._render_frame()
        return (
            self._get_obs_for(self.agent1_pos, self.coverage_map_1, self.battery_map_1),
            self._get_obs_for(self.agent2_pos, self.coverage_map_2, self.battery_map_2)
        ), (reward_1, reward_2), done, False, {}

    def _get_obs_for(self, agent_pos, coverage_map, battery_map):
        agent_map = np.zeros((self.size, self.size))
        agent_map[agent_pos[1], agent_pos[0]] = 1
        return np.stack([
            agent_map,
            coverage_map,
            self.victim_map,
            self.risk_map,
            self.obstacle_map,
            np.clip(battery_map, 0, 1),
            np.clip(self.time_map / 1000, 0, 1)
        ])
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 60))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size + 60))
        canvas.fill((255, 255, 255))
        pix = self.window_size // self.size
        grid_offset = 40

        pygame.draw.rect(canvas, (230, 230, 230), (0, 0, self.window_size, 40))
        font = pygame.font.SysFont('Arial', 16)

        ep_text = font.render(f"Ep: {self.episode_count}/{self.total_episodes}", True, (0, 0, 0))
        canvas.blit(ep_text, (10, 10))

        victims_text = font.render(f"Agent 1: {self.victims_found_1} | Agent 2: {self.victims_found_2}", True, (0, 0, 0))
        canvas.blit(victims_text, (self.window_size // 2 - 70, 10))

        batt_text = font.render(f"Batt: {self.battery_1 * 100:.0f}% | {self.battery_2 * 100:.0f}%", True, (0, 0, 0))
        canvas.blit(batt_text, (self.window_size - 180, 10))

        for y in range(self.size):
            for x in range(self.size):
                rect = pygame.Rect(x * pix, y * pix + grid_offset, pix, pix)
                pygame.draw.rect(canvas, (255, 255, 255), rect)
                if self.coverage_map_1[y, x] or self.coverage_map_2[y, x]:
                    pygame.draw.rect(canvas, (200, 200, 200), rect)
                if self.victim_map[y, x]:
                    color = (139, 0, 0) if (self.victim_found_map_1[y, x] or self.victim_found_map_2[y, x]) else (255, 0, 0)
                    pygame.draw.rect(canvas, color, rect)
                if self.obstacle_map[y, x]:
                    pygame.draw.rect(canvas, (50, 50, 50), rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        for (x, y), wind in self.wind_map.items():
            center = (x * pix + pix // 2, y * pix + pix // 2 + grid_offset)
            end = (center[0] + int(wind["dir"][0] * 15), center[1] + int(wind["dir"][1] * 15))
            green_intensity = 100 + int(wind["speed"] * 155)
            pygame.draw.line(canvas, (0, green_intensity, 0), center, end, 3)

        def draw_agent(pos, color):
            center = (pos[0] * pix + pix // 2, pos[1] * pix + pix // 2 + grid_offset)
            points = [
                (center[0], center[1] - pix // 3),
                (center[0] - pix // 4, center[1] + pix // 4),
                (center[0] + pix // 4, center[1] + pix // 4)
            ]
            pygame.draw.polygon(canvas, color, points)

        draw_agent(self.agent1_pos, (0, 100, 255))
        draw_agent(self.agent2_pos, (0, 200, 0))

        bar_y = self.window_size + grid_offset - 10
        pygame.draw.rect(canvas, (200, 200, 200), (0, bar_y, self.window_size, self.battery_bar_height))
        avg_battery = (self.battery_1 + self.battery_2) / 2
        pygame.draw.rect(canvas, (0, 200, 0), (0, bar_y, self.window_size * avg_battery, self.battery_bar_height))

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

def train_and_plot(env, agent1, agent2, total_episodes, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
    agent1_rewards, agent2_rewards = [], []
    coverage_1, coverage_2 = [], []
    battery_usage_1, battery_usage_2 = [], []
    risk_exposure_1, risk_exposure_2 = [], []

    for ep in range(total_episodes):
        obs = env.reset()
        env.episode_count = ep + 1
        done = False

        while not done:
            state1, state2 = obs
            action1 = agent1.select_action(state1, epsilon)
            action2 = agent2.select_action(state2, epsilon)

            next_obs, rewards, done, _, _ = env.step((action1, action2))

            agent1.store_transition((state1, action1, rewards[0], next_obs[0], done))
            agent2.store_transition((state2, action2, rewards[1], next_obs[1], done))

            agent1.update()
            agent2.update()

            obs = next_obs
        
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        agent1_rewards.append(env.total_reward_1)
        agent2_rewards.append(env.total_reward_2)
        coverage_1.append(np.sum(env.coverage_map_1) / (env.size**2))
        coverage_2.append(np.sum(env.coverage_map_2) / (env.size**2))
        battery_usage_1.append(1 - env.battery_1 / env.max_battery)
        battery_usage_2.append(1 - env.battery_2 / env.max_battery)
        risk_exposure_1.append(np.mean(env.risk_map[env.coverage_map_1 == 1]) if np.any(env.coverage_map_1 == 1) else 0)
        risk_exposure_2.append(np.mean(env.risk_map[env.coverage_map_2 == 1]) if np.any(env.coverage_map_2 == 1) else 0)
        print(f"Episode {ep+1}: "
      f"Agent 1: Reward={env.total_reward_1:.2f}, Victims={env.victims_found_1} | "
      f"Agent 2: Reward={env.total_reward_2:.2f}, Victims={env.victims_found_2}")
    
    df = pd.DataFrame({
        'episode': range(1, total_episodes+1),
        'agent1_reward': agent1_rewards,
        'agent2_reward': agent2_rewards,
        'coverage_1': coverage_1,
        'coverage_2': coverage_2,
        'battery_usage_1': battery_usage_1,
        'battery_usage_2': battery_usage_2,
        'risk_exposure_1': risk_exposure_1,
        'risk_exposure_2': risk_exposure_2
    })
    df.to_csv("training_log.csv", index=False)
    torch.save(agent1.online_net.state_dict(), "agent1_model.pth")
    torch.save(agent2.online_net.state_dict(), "agent2_model.pth")

    plt.figure(figsize=(12, 8))
    plt.subplot(2,2,1)
    plt.plot(agent1_rewards, label="Agent 1")
    plt.plot(agent2_rewards, label="Agent 2")
    plt.title("Total Reward")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(coverage_1, label="Agent 1")
    plt.plot(coverage_2, label="Agent 2")
    plt.title("Coverage %")
    plt.xlabel('Episode')
    plt.ylabel('Coverage')
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(battery_usage_1, label="Agent 1")
    plt.plot(battery_usage_2, label="Agent 2")
    plt.title("Battery Usage %")
    plt.xlabel('Episode')
    plt.ylabel('Usage %')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(risk_exposure_1, label="Agent 1")
    plt.plot(risk_exposure_2, label="Agent 2")
    plt.title("Avg Risk Exposure")
    plt.xlabel('Episode')
    plt.ylabel('Risk')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    total_episodes = 250
    env = MultiAgentGridEnv(render_mode="human", total_episodes=total_episodes)
    obs = env.reset()
    input_shape = obs[0].shape

    agent1 = DoubleDQNAgent(input_shape, 4)
    agent2 = DoubleDQNAgent(input_shape, 4)

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1

    train_and_plot(env, agent1, agent2, total_episodes)

    env.close()


