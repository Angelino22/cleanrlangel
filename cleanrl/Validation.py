import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
import supersuit as ss
from pettingzoo.atari import foozpong_v3
from torch.utils.tensorboard import SummaryWriter
import wandb

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    # Algorithm specific arguments
    env_id: str = "foozpong_v3"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the replay memory"""
    start_e: float = 0.1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 50000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    save_path: str = ""
    """the path to the model"""

def make_env(env_id, seed, capture_video, run_name):
    def thunk():
        env = foozpong_v3.parallel_env(render_mode="rgb_array")
        env = ss.max_observation_v0(env, 2)
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
        env = ss.agent_indicator_v0(env, type_only=False)
        if args.capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
        return env
    return thunk

class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space(env.possible_agents[0]).n),
        )

    def forward(self, x):
        return self.network(x / 255.0)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def stochastic_policy(env, agent):
    action_space = env.action_space(agent)
    if random.random() < 0.8:
        # Shoot at the enemy goal
        return 1  
    else:
        return action_space.sample()

class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device):
        self.buffer_size = buffer_size
        self.device = device
        self.observations = np.zeros((buffer_size, *observation_space.shape), dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size, *observation_space.shape), dtype=np.uint8)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.infos = [{} for _ in range(buffer_size)]
        self.ptr = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward, done, info):
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.infos[self.ptr] = info
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.observations[indices],
                     next_obs=self.next_observations[indices],
                     actions=self.actions[indices],
                     rewards=self.rewards[indices],
                     dones=self.dones[indices],
                     infos=[self.infos[i] for i in indices])
        return {k: torch.tensor(v, device=self.device, dtype=torch.float32) if k != 'infos' else v for k, v in batch.items()}

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = make_env(args.env_id, args.seed, args.capture_video, run_name)()
    agent_ids = envs.possible_agents
    offense_agent = agent_ids[0]
    defense_agent = agent_ids[2]
    ai1_agent = agent_ids[1]
    ai2_agent = agent_ids[3]
    q_network = QNetwork(envs).to(device)
    q_network.load_state_dict(torch.load(args.save_path))  # Load pre-trained model

    # Initialization
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    episode_length = 0
    episode_reward = np.zeros(len(agent_ids))
    adjusted_episode_reward = np.zeros(len(agent_ids))
    score_rewards_offense = 0  # Track score rewards by offense agent
    score_rewards_defense = 0  # Track score rewards by defense agent
    score_rewards_ai1 = 0  # Track score rewards by ai1 agent
    score_rewards_ai2 = 0  # Track score rewards by ai2 agent
    total_score_rewards_teamtrack = 0  # Score rewards by Team Track (offense + defense)
    total_score_rewards_teamai = 0  # Score rewards by Team AI (other agents)
    total_score_rewards_game = 0  # Total score rewards in the game
    error_count = 0  # Track error count

    game_score = {offense_agent: 0, defense_agent: 0, ai1_agent: 0, ai2_agent: 0}
    team_track_score = 0
    team_ai_score = 0
    game_timer = {offense_agent: time.time(), defense_agent: time.time(), ai1_agent: time.time(), ai2_agent: time.time()}

    def check_for_errors():
        global error_count
        current_time = time.time()
        for agent in agent_ids:
            if current_time - game_timer[agent] > 2:
                error_count += 1
                rewards[agent] -= 1
                game_timer[agent] = current_time

    for global_step in range(args.total_timesteps):
        actions = {}
        for agent in agent_ids:
            if agent == offense_agent and agent in obs:
                obs_tensor = torch.tensor(obs[agent][:, :, :4], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                q_values = q_network(obs_tensor)
                actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                game_timer[agent] = time.time()
            elif agent == defense_agent and agent in obs:
                actions[agent] = stochastic_policy(envs, agent)
                game_timer[agent] = time.time()
            elif agent in obs:
                # For enemy DQN agents
                obs_tensor = torch.tensor(obs[agent][:, :, :4], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                q_values = q_network(obs_tensor)
                actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]
                game_timer[agent] = time.time()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        check_for_errors()

        for agent in agent_ids:
            if agent in rewards:
                episode_reward[agent_ids.index(agent)] += rewards[agent]
                adjusted_episode_reward[agent_ids.index(agent)] += rewards[agent]
            rewards[agent] = np.clip(rewards[agent], -1, 1)

        if offense_agent in rewards:
            if rewards[offense_agent] == 1:
                score_rewards_offense += 1
                total_score_rewards_teamtrack += 1
                game_score[offense_agent] += 1

        if defense_agent in rewards:
            if rewards[defense_agent] == 1:
                score_rewards_defense += 1
                total_score_rewards_teamtrack += 1
                game_score[defense_agent] += 1

        for agent in agent_ids:
            if agent not in [offense_agent, defense_agent] and agent in rewards:
                if rewards[agent] == 1:  
                    total_score_rewards_teamai += 1
                    game_score[agent] += 1

        total_score_rewards_game = total_score_rewards_teamtrack + total_score_rewards_teamai

        if game_score[offense_agent] + game_score[defense_agent] >= 10 or game_score[ai1_agent] + game_score[ai2_agent] >= 10:
            if game_score[offense_agent] + game_score[defense_agent] >= 10:
                team_track_score += 1
            else:
                team_ai_score += 1
            game_score = {offense_agent: 0, defense_agent: 0, ai1_agent: 0, ai2_agent: 0}
            obs, _ = envs.reset(seed=args.seed)

        obs = next_obs

        if global_step % 100 == 0:
            writer.add_scalar(f"charts/episodic_return0", episode_reward[0], global_step)
            writer.add_scalar(f"charts/episodic_return1", episode_reward[1], global_step)
            writer.add_scalar(f"charts/adjusted_episodic_return0", adjusted_episode_reward[0], global_step)
            writer.add_scalar(f"charts/adjusted_episodic_return1", adjusted_episode_reward[1], global_step)
            writer.add_scalar(f"charts/score_rewards_offense", score_rewards_offense, global_step)
            writer.add_scalar(f"charts/score_rewards_defense", score_rewards_defense, global_step)
            writer.add_scalar(f"charts/score_rewards_ai1", score_rewards_ai1, global_step)
            writer.add_scalar(f"charts/score_rewards_ai2", score_rewards_ai2, global_step)
            writer.add_scalar(f"charts/total_score_rewards_teamtrack", total_score_rewards_teamtrack, global_step)
            writer.add_scalar(f"charts/total_score_rewards_teamai", total_score_rewards_teamai, global_step)
            writer.add_scalar(f"charts/total_score_rewards_game", total_score_rewards_game, global_step)
            writer.add_scalar(f"charts/error_rate", error_count, global_step)
            writer.add_scalar(f"charts/team_track_score", team_track_score, global_step)
            writer.add_scalar(f"charts/team_ai_score", team_ai_score, global_step)
            writer.add_scalar(f"charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if args.track:
                wandb.log({
                    "charts/episodic_return0": episode_reward[0],
                    "charts/episodic_return1": episode_reward[1],
                    "charts/adjusted_episodic_return0": adjusted_episode_reward[0],
                    "charts/adjusted_episodic_return1": adjusted_episode_reward[1],
                    "charts/score_rewards_offense": score_rewards_offense,
                    "charts/score_rewards_defense": score_rewards_defense,
                    "charts/score_rewards_ai1": score_rewards_ai1,
                    "charts/score_rewards_ai2": score_rewards_ai2,
                    "charts/total_score_rewards_teamtrack": total_score_rewards_teamtrack,
                    "charts/total_score_rewards_teamai": total_score_rewards_teamai,
                    "charts/total_score_rewards_game": total_score_rewards_game,
                    "charts/error_rate": error_count,
                    "charts/team_track_score": team_track_score,
                    "charts/team_ai_score": team_ai_score,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "global_step": global_step
                })

        if infos:
            episode_length = global_step
            episode_reward = np.zeros(len(agent_ids))
            adjusted_episode_reward = np.zeros(len(agent_ids))

        obs = next_obs

    envs.close()
    writer.close()
