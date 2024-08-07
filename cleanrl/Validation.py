import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import supersuit as ss
from pettingzoo.atari import foozpong_v3
from stable_baselines3.common.buffers import ReplayBuffer
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
    capture_video: bool = False
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
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
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
    save_path: str = "models/q_network.pth"

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
    target_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network.load_state_dict(q_network.state_dict())

    observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        envs.action_space(offense_agent),
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

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
                epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
                if random.random() < epsilon:
                    actions[agent] = envs.action_space(agent).sample()
                else:
                    obs_tensor = torch.tensor(obs[agent][:, :, :4], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                    q_values = q_network(obs_tensor)
                    actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]
            elif agent == defense_agent and agent in obs:
                actions[agent] = stochastic_policy(envs, agent)
            elif agent in obs:
                # For enemy DQN agents
                epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
                if random.random() < epsilon:
                    actions[agent] = envs.action_space(agent).sample()
                else:
                    obs_tensor = torch.tensor(obs[agent][:, :, :4], dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                    q_values = q_network(obs_tensor)
                    actions[agent] = torch.argmax(q_values, dim=1).cpu().numpy()[0]

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        for agent in agent_ids:
            if agent in rewards:
                episode_reward[agent_ids.index(agent)] += rewards[agent]
                adjusted_episode_reward[agent_ids.index(agent)] += rewards[agent]

        if offense_agent in rewards:
            real_next_obs = next_obs[offense_agent].copy()
            if truncations[offense_agent]:
                real_next_obs = infos[offense_agent].get("final_observation", real_next_obs)
            rb.add(
                np.array(obs[offense_agent][:, :, :4]).transpose(2, 0, 1),
                np.array(real_next_obs[:, :, :4]).transpose(2, 0, 1),
                np.array([actions[offense_agent]]),
                np.array([rewards[offense_agent]]),
                np.array([terminations[offense_agent]]),
                infos.get(offense_agent, {})
            )
            if rewards[offense_agent] == 1:
                goals_scored_offense += 1
                total_goals_teamtrack += 1

        if defense_agent in rewards:
            if rewards[defense_agent] == 1:
                goals_scored_defense += 1
                total_goals_teamtrack += 1

        for agent in agent_ids:
            if agent not in [offense_agent, defense_agent] and agent in rewards:
                if rewards[agent] == 1:  
                    total_goals_teamai += 1

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                if rb.size() >= args.batch_size:
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        next_obs_tensor = torch.tensor(data.next_observations, dtype=torch.float32).to(device)
                        target_max, _ = target_network(next_obs_tensor.permute((0, 1, 2, 3))).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    obs_tensor = torch.tensor(data.observations, dtype=torch.float32).to(device)
                    old_val = q_network(obs_tensor.permute((0, 1, 2, 3))).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar(f"losses/td_loss_{offense_agent}", loss, global_step)
                        writer.add_scalar(f"losses/q_values_{offense_agent}", old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                        if args.track:
                            wandb.log({
                                f"losses/td_loss_{offense_agent}": loss.item(),
                                f"losses/q_values_{offense_agent}": old_val.mean().item(),
                                "charts/SPS": int(global_step / (time.time() - start_time)),
                                "charts/episodic_return0": episode_reward[0],
                                "charts/episodic_return1": episode_reward[1],
                                "charts/adjusted_episodic_return0": adjusted_episode_reward[0],
                                "charts/adjusted_episodic_return1": adjusted_episode_reward[1],
                                "charts/episodic_length": global_step - episode_length,
                                "charts/goals_scored_offense": goals_scored_offense,
                                "charts/goals_scored_defense": goals_scored_defense,
                                "charts/total_goals_teamtrack": total_goals_teamtrack,
                                "charts/total_goals_teamai": total_goals_teamai,
                                "charts/learning_rate": args.learning_rate,
                                "charts/min_epsilon": args.end_e,
                                "charts/epsilon_decay": args.exploration_fraction,
                                "global_step": global_step
                            })

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        if global_step % args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )

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
                    "charts/learning_rate": args.learning_rate,
                    "charts/min_epsilon": args.end_e,
                    "charts/epsilon_decay": args.exploration_fraction,
                    "global_step": global_step
                })
            episode_length = global_step
            episode_reward = np.zeros(len(agent_ids))
            adjusted_episode_reward = np.zeros(len(agent_ids))

        obs = next_obs

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}_{offense_agent}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
