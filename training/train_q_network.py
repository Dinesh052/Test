"""
Train the DialogXpert Q-network via TD-learning on real environment episodes.

Runs epsilon-greedy episodes against CrisisNegotiatorEnvironment, collects
(obs, action, reward, next_obs, done) transitions into a replay buffer,
and trains the 384→64→10 MLP with batched TD(0) updates.

Usage:
    python train_q_network.py [--episodes 500] [--batch-size 64] [--lr 3e-4]
"""
import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from server.environment import CrisisNegotiatorEnvironment
from server.q_network import ACTION_TYPES, ACTION_TO_IDX
from models import NegotiatorAction

# ── Content templates per action type (so actions are meaningful) ──
ACTION_CONTENT = {
    "emotional_label": "It sounds like you're feeling overwhelmed right now.",
    "mirror": "You said nobody listens — I hear that.",
    "open_question": "Help me understand what happened from your side.",
    "acknowledge_demand": "I hear what you're asking for. Let me work on that.",
    "offer_concession": "Here's what I can do right now to help.",
    "buy_time": "Give me a moment — I'm working on something for you.",
    "push_back_commander": "Hold position — I'm making progress here.",
    "speak": "I'm here and I'm listening to you.",
    "request_demand": "Tell me what you need most right now.",
    "ask_proof_of_life": "Can you let me know everyone in there is okay?",
}


def build_obs_text(obs) -> str:
    """Convert observation to text for sentence-transformer encoding."""
    parts = []
    parts.append(f"Phase: {getattr(obs, 'phase', 'unknown')}")
    parts.append(f"Time left: {getattr(obs, 'time_remaining', 0)}")
    parts.append(f"Commander: {getattr(obs, 'commander_patience', 'patient')}")
    ht_msg = getattr(obs, 'last_ht_message', '')
    if ht_msg:
        parts.append(f"HT: {ht_msg[:150]}")
    cues = getattr(obs, 'last_ht_cues', [])
    if cues:
        parts.append(f"Cues: {', '.join(cues)}")
    demands = getattr(obs, 'stated_demands', [])
    if demands:
        d_strs = [d.get('text', '')[:40] for d in demands[:3]]
        parts.append(f"Demands: {'; '.join(d_strs)}")
    flags = getattr(obs, 'supervisor_flags', [])
    if flags:
        parts.append(f"Flags: {len(flags)}")
    traj = getattr(obs, 'agitation_trajectory', [])
    if traj:
        parts.append(f"Agitation trend: {traj}")
    return " | ".join(parts)


def make_action(action_type: str) -> NegotiatorAction:
    """Create a NegotiatorAction with appropriate content."""
    content = ACTION_CONTENT.get(action_type, "I'm listening.")
    target = "commander" if action_type == "push_back_commander" else "hostage_taker"
    return NegotiatorAction(
        action_type=action_type, content=content,
        reasoning="q_network_policy", target=target,
    )


class ReplayBuffer:
    def __init__(self, maxlen=50_000):
        self.buf = deque(maxlen=maxlen)

    def push(self, obs_emb, action_idx, reward, next_emb, done):
        self.buf.append((obs_emb, action_idx, reward, next_emb, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        obs = np.stack([b[0] for b in batch])
        acts = np.array([b[1] for b in batch])
        rews = np.array([b[2] for b in batch], dtype=np.float32)
        nobs = np.stack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return obs, acts, rews, nobs, dones

    def __len__(self):
        return len(self.buf)


def train(args):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    # Build Q-network
    q_net = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, len(ACTION_TYPES)))
    target_net = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, len(ACTION_TYPES)))
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    replay = ReplayBuffer(maxlen=args.buffer_size)
    env = CrisisNegotiatorEnvironment()
    difficulties = ["easy", "medium", "hard"]
    log = []

    eps_start, eps_end, eps_decay = 1.0, 0.05, args.episodes * 0.7
    total_steps = 0
    t0 = time.time()

    for ep in range(args.episodes):
        diff = difficulties[ep % 3]
        obs = env.reset(task_id=f"generate:{diff}", seed=ep)
        obs_text = build_obs_text(obs)
        obs_emb = encoder.encode([obs_text])[0]

        ep_reward = 0.0
        ep_steps = 0
        epsilon = eps_end + (eps_start - eps_end) * max(0, 1 - ep / eps_decay)

        while not getattr(obs, 'done', False) and ep_steps < 25:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randint(0, len(ACTION_TYPES) - 1)
            else:
                with torch.no_grad():
                    qv = q_net(torch.tensor(obs_emb, dtype=torch.float32))
                    action_idx = qv.argmax().item()

            action_type = ACTION_TYPES[action_idx]
            action = make_action(action_type)
            obs = env.step(action)
            ep_steps += 1
            total_steps += 1

            reward = float(getattr(obs, 'reward', 0.0))
            done = bool(getattr(obs, 'done', False))
            next_text = build_obs_text(obs)
            next_emb = encoder.encode([next_text])[0]

            replay.push(obs_emb, action_idx, reward, next_emb, done)
            obs_emb = next_emb
            ep_reward += reward

            # Train if enough samples
            if len(replay) >= args.batch_size:
                b_obs, b_acts, b_rews, b_nobs, b_dones = replay.sample(args.batch_size)
                b_obs_t = torch.tensor(b_obs, dtype=torch.float32)
                b_nobs_t = torch.tensor(b_nobs, dtype=torch.float32)
                b_rews_t = torch.tensor(b_rews, dtype=torch.float32)
                b_dones_t = torch.tensor(b_dones, dtype=torch.float32)
                b_acts_t = torch.tensor(b_acts, dtype=torch.long)

                # Current Q-values
                q_values = q_net(b_obs_t)
                q_sa = q_values.gather(1, b_acts_t.unsqueeze(1)).squeeze(1)

                # Target Q-values (from target network)
                with torch.no_grad():
                    next_q = target_net(b_nobs_t).max(dim=1).values
                    targets = b_rews_t + args.gamma * next_q * (1 - b_dones_t)

                loss = F.mse_loss(q_sa, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network periodically
        if ep % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        outcome = ""
        msg = getattr(obs, 'message', '') or ''
        if 'surrender' in msg.lower() or 'released' in msg.lower():
            outcome = "success"
        elif 'harm' in msg.lower():
            outcome = "harm"
        else:
            outcome = "other"

        final_reward = float(getattr(obs, 'reward', 0.0))
        log.append({"episode": ep, "difficulty": diff, "steps": ep_steps,
                     "cumulative_reward": round(ep_reward, 4),
                     "final_reward": round(final_reward, 4),
                     "outcome": outcome, "epsilon": round(epsilon, 3)})

        if ep % 25 == 0 or ep == args.episodes - 1:
            recent = log[-25:]
            avg_r = sum(e["final_reward"] for e in recent) / len(recent)
            succ = sum(1 for e in recent if e["outcome"] == "success") / len(recent)
            elapsed = time.time() - t0
            print(f"[{ep:4d}/{args.episodes}] avg_reward={avg_r:.3f} success={succ:.0%} "
                  f"eps={epsilon:.2f} buf={len(replay)} elapsed={elapsed:.0f}s")

    # Save weights
    torch.save(q_net.state_dict(), args.output)
    print(f"\n✓ Saved Q-network weights to {args.output}")

    # Save training log
    log_path = args.output.replace('.pt', '_log.json')
    Path(log_path).write_text(json.dumps(log, indent=2))
    print(f"✓ Saved training log to {log_path}")

    # Print final stats
    last_50 = log[-50:]
    avg = sum(e["final_reward"] for e in last_50) / len(last_50)
    succ = sum(1 for e in last_50 if e["outcome"] == "success") / len(last_50)
    print(f"\nFinal 50 episodes: avg_reward={avg:.3f} success_rate={succ:.0%}")

    # Plot epsilon-decay + reward curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Q-Network Training (DialogXpert TD-Learning)")

        episodes = [e["episode"] for e in log]
        rewards = [e["final_reward"] for e in log]
        epsilons = [e["epsilon"] for e in log]

        # Left: reward curve
        win = min(25, len(rewards))
        rolling = [sum(rewards[max(0,i-win+1):i+1])/min(i+1,win) for i in range(len(rewards))]
        ax1.scatter(episodes, rewards, alpha=0.2, s=8, c='steelblue')
        ax1.plot(episodes, rolling, 'k-', lw=2, label=f'rolling mean (w={win})')
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Final Reward")
        ax1.set_title("Episode Reward"); ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1.05)

        # Right: epsilon-decay curve
        ax2.plot(episodes, epsilons, 'r-', lw=2, label='epsilon (exploration)')
        ax2.fill_between(episodes, 0, epsilons, alpha=0.15, color='red')
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Epsilon")
        ax2.set_title("Epsilon-Greedy Decay (1.0 → 0.05)")
        ax2.legend(fontsize=8); ax2.set_ylim(0, 1.05)
        ax2.annotate(f'ε={epsilons[0]:.2f}\n(random)', xy=(0, epsilons[0]),
                     fontsize=7, ha='left')
        ax2.annotate(f'ε={epsilons[-1]:.2f}\n(greedy)', xy=(len(epsilons)-1, epsilons[-1]),
                     fontsize=7, ha='right')

        plt.tight_layout()
        plt.savefig("q_network_training.png", dpi=150)
        plt.close()
        print(f"✓ Saved q_network_training.png")
    except ImportError:
        pass

    # Verify: rank actions for a sample observation
    print("\n=== Action Rankings (sample high-agitation obs) ===")
    test_obs = "Phase: negotiation | Time left: 12 | Commander: restless | HT: Nobody cares about me! | Cues: voice raised, pacing | Demands: Talk to my kids"
    test_emb = torch.tensor(encoder.encode([test_obs])[0], dtype=torch.float32)
    with torch.no_grad():
        qv = q_net(test_emb)
    ranked = sorted([(ACTION_TYPES[i], qv[i].item()) for i in range(len(ACTION_TYPES))],
                    key=lambda x: x[1], reverse=True)
    for action, val in ranked:
        print(f"  {action:25s} Q={val:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--buffer-size", type=int, default=50000)
    p.add_argument("--target-update", type=int, default=10)
    p.add_argument("--output", default="q_network.pt")
    train(p.parse_args())
