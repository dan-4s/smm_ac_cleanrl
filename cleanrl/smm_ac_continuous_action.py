# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import signal
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import tyro

from cleanrl_utils.buffers import ReplayBuffer
from smm_ac_utils.shared_functions import get_steps_per_env, write_and_dump


@dataclass
class Args:
    wandb_run_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    output_filename: str = "test_results"
    "the name of the results file where we store run data"
    seed: int = int.from_bytes(os.urandom(4), "little")
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "SMM-AC"
    """the wandb's project name"""
    wandb_entity: str = "gauthier-gidel"
    """the entity (team) of wandb's project"""
    wandb_group: str = "SMM"
    """The group of an individual run, indexed by the algorithm and subcategories therein."""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    value_est: str = "explicit_regulariser" # Can also be "empirical_expectation"
    """The value estimation method for SMM."""
    num_val_est_samples: int = 1
    """The number of samples collected for the value estimate, when empirical_expectation"""
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    pi_ref_lr: float = 3e-5
    """learning rate of the reference policy"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    ref_policy_frequency: int = 2
    """the frequency of training the reference policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 1
    """Distribution spikeyness hyperparameter."""
    omega: float = 5.0
    """Temperature hyperparameter."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
def weighted_log_sum_exp(value, weights, dim=None):
        eps = 1e-20
        m, idx = torch.max(value, dim=dim)
        return m.squeeze(dim) + torch.log(torch.sum(torch.exp(value-m)*(weights),
                                       dim=dim) + eps)

class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, x_t
    
    # TODO: Verify that the below method is correct...
    def get_log_prob(self, x, x_t):
        """The naming convention here is fucked, but I'll stick with it for now..."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = args.wandb_run_name
    checkpoint_filename = args.output_filename + ".pt"
    if args.track:
        import wandb
        wandb_id = os.path.basename(args.output_filename)
        wandb.init(
            project=args.wandb_project_name,
            # mode="offline", # TEMPORARY until I get back my wandb access...
            entity=args.wandb_entity,
            id=wandb_id, # Just the run_id, without the folder path.
            resume="allow",
            sync_tensorboard=False, # Avoid DDOSing the cluster.
            group=args.wandb_group,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # For data logging.
    run_data = {
        "episodic_return": [],
        "episodic_length": [],
        "episodic_step": [],
    }
    schema = pa.schema([
        ("episodic_return", pa.float64()), # List of floats
        ("episodic_length", pa.int64()),   # List of ints
        ("episodic_step", pa.int64()),     # The global step
    ])
    restart_idx = os.getenv("SLURM_RESTART_COUNT", "0")
    current_parquet_path = f"{args.output_filename}_restart{restart_idx}.parquet"
    parquet_writer = pq.ParquetWriter(current_parquet_path, schema)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    pi_ref = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    pi_ref_optimizer = optim.Adam(list(pi_ref.parameters()), lr=args.pi_ref_lr) # Want pi_ref to move very slowly!

    # Automatic entropy tuning
    alpha = args.alpha
    if args.autotune:
        # For SMM-AC we can only tune omega in the same way that SAC tunes
        # their alpha. Our alpha is a choice about the minimum entropy we want
        # a policy to have -> its minimum shape is like a softmax. As such, we
        # will not explore learning our alpha for now.
        target_KL = 0.05 # Like TRPO, PPO, etc., keep the divergence small.
        
        # Setting an inverse temperature of 5 is generally a good starting
        # point.
        log_lambda = torch.tensor([-1.6094], requires_grad=True, device=device)
        omega = 1 / (log_lambda.exp().item())
        lambda_optimizer = optim.Adam([log_lambda], lr=args.q_lr)

        # target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        # log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # alpha = 1 / log_alpha.exp().item()
        # a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        omega = args.omega

    # Pre-emption recovery: register termination and time-out signals.
    start_step = 0
    if os.path.exists(checkpoint_filename):
        print(f"Resuming from checkpoint: {checkpoint_filename}")
        ckpt = torch.load(checkpoint_filename, map_location=device)
        actor.load_state_dict(ckpt['actor_state_dict'])
        pi_ref.load_state_dict(ckpt['pi_ref_state_dict'])
        qf1.load_state_dict(ckpt['qf1_state_dict'])
        qf2.load_state_dict(ckpt['qf2_state_dict'])
        q_optimizer.load_state_dict(ckpt['q_optimizer_state_dict'])
        actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        pi_ref_optimizer.load_state_dict(ckpt['pi_ref_optimizer_state_dict'])
        if args.autotune:
            log_lambda.data = ckpt['log_lambda']
            lambda_optimizer.load_state_dict(ckpt['lambda_optimizer_state_dict'])
        start_step = ckpt['global_step'] + 1
        qf1_target.load_state_dict(ckpt['qf1_target_state_dict'])
        qf2_target.load_state_dict(ckpt['qf2_target_state_dict'])

    # Signal handler
    global_step = start_step
    def save_checkpoint_and_exit(signum, frame):
        print(f"\nSignal {signum} received. Saving and exiting...")
        write_and_dump(parquet_writer, run_data)
        parquet_writer.close()
        ckpt = {
            'global_step': global_step,
            'actor_state_dict': actor.state_dict(),
            'pi_ref_state_dict': pi_ref.state_dict(),
            'qf1_state_dict': qf1.state_dict(),
            'qf1_target_state_dict': qf1_target.state_dict(),
            'qf2_state_dict': qf2.state_dict(),
            'qf2_target_state_dict': qf2_target.state_dict(),
            'q_optimizer_state_dict': q_optimizer.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'pi_ref_optimizer_state_dict': pi_ref_optimizer.state_dict(),
            'log_lambda': log_lambda if args.autotune else None,
            'lambda_optimizer_state_dict': lambda_optimizer.state_dict() if args.autotune else None,
        }
        torch.save(ckpt, checkpoint_filename)
        print(f"Checkpoint saved to {checkpoint_filename}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, save_checkpoint_and_exit)
    signal.signal(signal.SIGUSR1, save_checkpoint_and_exit)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    total_timesteps = get_steps_per_env(args.env_id)
    effective_learning_starts = args.learning_starts if start_step == 0 else (start_step + 5000)
    lr_scheduler = CosineAnnealingLR(
        optimizer=pi_ref_optimizer,
        T_max=total_timesteps // args.ref_policy_frequency,
        eta_min=1e-6,
    ) # TODO: add the learning rate and scheduler state to the checkpoint!
    for global_step in range(start_step, total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # Always sample directly from agent if we're past the
            # learning_starts mark! The data will be better!
            actions, _, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    if args.track:
                        wandb.log({
                            "charts/episodic_return": info["episode"]["r"],
                            "charts/episodic_length": info["episode"]["l"],
                        }, step=global_step)
                    run_data["episodic_return"].append(float(info["episode"]["r"][0]))
                    run_data["episodic_length"].append(int(info["episode"]["l"][0]))
                    run_data["episodic_step"].append(int(global_step))
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > effective_learning_starts and rb.size() > args.batch_size:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                # Estimate the value of the next state.
                if(args.value_est == "explicit_regulariser"):
                    # Similarly to SAC, generate the value estimate by taking:
                    # Q(s, a) - temp * regulariser.
                    # NOTE: We need the random_sample because it is the action
                    # estimate generated from the actor's normal distribution. To
                    # then get pi_ref's log_prob for that action, we need to have
                    # the sample on the domain of the normal distribution.
                    qs = []
                    # import pdb
                    # pdb.set_trace()
                    N = args.num_val_est_samples
                    for _ in range(N):
                        next_state_actions, next_state_log_pi, _, random_sample = actor.get_action(data.next_observations)
                        next_state_log_pi_ref = pi_ref.get_log_prob(data.next_observations, random_sample)
                        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                        qs.append(
                            torch.min(qf1_next_target, qf2_next_target) -
                            (1/omega) * (next_state_log_pi - next_state_log_pi_ref) )
                    qs = torch.stack(qs).squeeze(0)
                    # Avoid the mean if single-sample estimator.
                    min_qf_next_target = torch.mean(qs, dim=0) if(N > 1) else qs
                elif(args.value_est == "empirical_expectation"):
                    # TODO: LMFAO JUST USE LOG-SUM-EXP!!!!! Weight the sum with 1/n obvs.
                    # Estimate the value by taking:
                    # temp * log( empirical_expectation[ exp(1/temp * Q(s,a)) ] )
                    # Start out with a single sample, add more if unstable.
                    # Averaging is quite unstable, as it turns out haha.
                    qs = []
                    # import pdb
                    # pdb.set_trace()
                    N = args.num_val_est_samples
                    for _ in range(N):
                        pi_ref_action, log_pi_ref, _, _ = pi_ref.get_action(data.next_observations)
                        qf1_next_target = qf1_target(data.next_observations, pi_ref_action)
                        qf2_next_target = qf2_target(data.next_observations, pi_ref_action)
                        q_est = torch.min(qf1_next_target, qf2_next_target)
                        qs.append(omega * q_est)
                    qs = torch.stack(qs).squeeze(0)
                    min_qf_next_target = (1/omega) * weighted_log_sum_exp(qs, 1/N, dim=0)
                    # min_qf_next_target = min_qf_next_target
                    # pi_ref_action, log_pi_ref, _, _ = pi_ref.get_action(data.next_observations)
                    # qf1_next_target = qf1_target(data.next_observations, pi_ref_action)
                    # qf2_next_target = qf2_target(data.next_observations, pi_ref_action)
                    # q_est = torch.min(qf1_next_target, qf2_next_target)
                    # min_qf_next_target = q_est
                else:
                    raise ValueError(f"Value estimator {args.value_est} is not recognised.")
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                # Update the policy.
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    act, log_pi, _, random_sample = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, act)
                    qf2_pi = qf2(data.observations, act)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    with torch.no_grad():
                        # log_pi_ref must not provide gradients to the Actor
                        log_pi_ref = pi_ref.get_log_prob(data.observations, random_sample) 

                    # KL-regularized loss
                    # We minimize: KL(pi || pi_ref) - omega * Q
                    # Rewritten: (1/omega) * (log_pi - log_pi_ref) - min_qf_pi
                    actor_loss = ((1.0 / omega) * (log_pi - log_pi_ref) - min_qf_pi).mean()
                    # actor_loss = (((alpha + omega) * log_pi) - min_qf_pi).mean() -> old method, didn't explicitly separate pi_ref and actor.

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _, random_sample = actor.get_action(data.observations)
                            log_pi_ref = pi_ref.get_log_prob(data.observations, random_sample)
                        lambda_loss = (log_lambda.exp() * (-log_pi_ref + log_pi + target_KL)).mean()

                        lambda_optimizer.zero_grad()
                        lambda_loss.backward()
                        lambda_optimizer.step()
                        omega = 1 / (log_lambda.exp().item())
            
            if global_step % args.ref_policy_frequency == 0:  # TD 3 Delayed update support
                # Update the reference policy.
                for _ in range(
                    args.ref_policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    # Try training pi_ref with the data sampled from pi.
                    act_ref, log_pi_ref, _, _ = pi_ref.get_action(data.observations)
                    # with torch.no_grad():
                    # The Q-values shouldn't provide gradients to pi_ref.
                    qf1_pi = qf1(data.observations, act_ref)
                    qf2_pi = qf2(data.observations, act_ref)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    # Try training pi_ref with the data sampled from pi.
                    pi_ref_loss = ((log_pi_ref) - alpha * min_qf_pi).mean()

                    pi_ref_optimizer.zero_grad()
                    pi_ref_loss.backward()
                    pi_ref_optimizer.step()
                    lr_scheduler.step()

                    # if args.autotune:
                    #     with torch.no_grad():
                    #         _, log_pi, _, _ = actor.get_action(data.observations)
                    #     alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    #     a_optimizer.zero_grad()
                    #     alpha_loss.backward()
                    #     a_optimizer.step()
                    #     alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                sps = int((global_step - start_step) / (time.time() - start_time))
                print("SPS:", sps)
                if args.track:
                    log_dict = {
                        "losses/qf1_values": qf1_a_values.mean().item(),
                        "losses/qf2_values": qf2_a_values.mean().item(),
                        "losses/qf1_loss": qf1_loss.item(),
                        "losses/qf2_loss": qf2_loss.item(),
                        "losses/qf_loss": qf_loss.item() / 2.0,
                        "losses/actor_loss": actor_loss.item(),
                        "losses/pi_ref_loss": pi_ref_loss.item(),
                        "hyperparams/alpha": alpha,
                        "hyperparams/omega": omega,
                        "hyperparams/pi_ref_lr": lr_scheduler.get_last_lr()[0],
                        "charts/SPS": sps,
                    }
                    if args.autotune:
                        # log_dict["losses/alpha_loss"] = alpha_loss.item()
                        log_dict["losses/lambda_loss"] = lambda_loss.item()
                    # ONE network call instead of many disk calls
                    wandb.log(log_dict, step=global_step)
            
            if(global_step % 10_000 == 0):
                write_and_dump(writer=parquet_writer, run_data=run_data)

    # Unregister the signals to avoid double-write in case of kill at the end.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)

    # Save a checkpoint:
    ckpt = {
        'global_step': global_step,
        'actor_state_dict': actor.state_dict(),
        'pi_ref_state_dict': pi_ref.state_dict(),
        'qf1_state_dict': qf1.state_dict(),
        'qf1_target_state_dict': qf1_target.state_dict(),
        'qf2_state_dict': qf2.state_dict(),
        'qf2_target_state_dict': qf2_target.state_dict(),
        'q_optimizer_state_dict': q_optimizer.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'pi_ref_optimizer_state_dict': pi_ref_optimizer.state_dict(),
    }
    torch.save(ckpt, checkpoint_filename)

    # Dump data and close parquet writer.
    write_and_dump(writer=parquet_writer, run_data=run_data)
    parquet_writer.close()
    envs.close()
