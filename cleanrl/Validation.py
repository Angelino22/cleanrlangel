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
    model_path: str = "runs/your_pretrained_model.pth"
    """the path to the pre-trained model"""
    # Algorithm specific arguments
    env_id: str = "foozpong_v3"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
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
    defense_agent = agent_ids[1]
    q_network = QNetwork(envs).to(device)
    q_network.load_state_dict(torch.load(args.model_path))  # Load pre-trained model

    # Initialization
    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    episode_length = 0
    episode_reward = np.zeros(len(agent_ids))
    adjusted_episode_reward = np.zeros(len(agent_ids))
    goals_scored_offense = 0  # Track goals scored by offense agent
    goals_scored_defense = 0  # Track goals scored by defense agent
    total_goals_teamtrack = 0
    total_goals_teamai = 0

    for global_step in range(args.total_timesteps):
        actions = {}
        for agent in agent_ids:
            if agent == offense_agent and agent in obs:
                obs_tensor = torch.tensor(obs[agent][:, :, :4], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                q_values = q_network(obs_tensor)
                actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            elif agent == defense_agent and agent in obs:
                actions[agent] = stochastic_policy(envs, agent)
            elif agent in obs:
                # For enemy DQN agents
                obs_tensor = torch.tensor(obs[agent][:, :, :4], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                q_values = q_network(obs_tensor)
                actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        for agent in agent_ids:
            if agent in rewards:
                episode_reward[agent_ids.index(agent)] += rewards[agent]
                adjusted_episode_reward[agent_ids.index(agent)] += rewards[agent]

        obs = next_obs

        if global_step % 100 == 0:
            writer.add_scalar(f"charts/episodic_return0", episode_reward[0], global_step)
            writer.add_scalar(f"charts/episodic_return1", episode_reward[1], global_step)
            writer.add_scalar(f"charts/adjusted_episodic_return0", adjusted_episode_reward[0], global_step)
            writer.add_scalar(f"charts/adjusted_episodic_return1", adjusted_episode_reward[1], global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if args.track:
                wandb.log({
                    "charts/episodic_return0": episode_reward[0],
                    "charts/episodic_return1": episode_reward[1],
                    "charts/adjusted_episodic_return0": adjusted_episode_reward[0],
                    "charts/adjusted_episodic_return1": adjusted_episode_reward[1],
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "charts/episodic_length": global_step - episode_length,
                    "charts/goals_scored_offense": goals_scored_offense,
                    "charts/goals_scored_defense": goals_scored_defense,
                    "charts/total_goals_teamtrack": total_goals_teamtrack,
                    "charts/total_goals_teamai": total_goals_teamai,
                    "charts/total_goals_game": total_goals_teamtrack + total_goals_teamai,
                    "global_step": global_step
                })

        if infos:
            if args.track:
                wandb.log({
                    "charts/episodic_return0": episode_reward[0],
                    "charts/episodic_return1": episode_reward[1],
                    "charts/adjusted_episodic_return0": adjusted_episode_reward[0],
                    "charts/adjusted_episodic_return1": adjusted_episode_reward[1],
                    "charts/episodic_length": global_step - episode_length,
                    "charts/goals_scored_offense": goals_scored_offense,
                    "charts/goals_scored_defense": goals_scored_defense,
                    "charts/total_goals_teamtrack": total_goals_teamtrack,
                    "charts/total_goals_teamai": total_goals_teamai,
                    "global_step": global_step
                })
            episode_length = global_step
            episode_reward = np.zeros(len(agent_ids))
            adjusted_episode_reward = np.zeros(len(agent_ids))

        obs = next_obs

    envs.close()
    writer.close()
