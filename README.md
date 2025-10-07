# PyBullet Simple Tag 3D

This repository provides a PyBullet-powered 3D reinterpretation of the PettingZoo Multi-Agent Particle Environment (MPE) [Simple Tag](https://pettingzoo.farama.org/environments/mpe/simple_tag/) scenario.

The project mirrors the structure of the original 2D environment while extending it to three dimensions. Agents and landmarks are represented as spherical rigid bodies inside a bounded PyBullet simulation. Adversarial agents attempt to tag the good agent, who receives penalties for collisions and for leaving the arena, exactly as in the classic MPE setup.

## Environment features

* **Agents** – configurable number of adversaries (red) and a good agent (green) that move freely across the arena.
* **Obstacles** – immovable dark landmarks that obstruct motion.
* **PyBullet simulation** – all bodies are simulated inside PyBullet with optional GUI-based rendering.
* **PettingZoo-inspired API** – the `SimpleTag3DBulletEnv` class exposes `reset()` and `step()` functions returning dictionaries of observations, rewards, terminations, truncations, and infos for each agent. A convenience `make_env()` helper mirrors the PettingZoo factory style.

## Installation

Install the dependencies (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

## Usage

```python
from pybullet_simple_tag import make_env

env = make_env(gui=True)  # set gui=False to run headless
obs, infos = env.reset()
for _ in range(50):
    actions = env.sample_actions()
    obs, rewards, terminations, truncations, infos = env.step(actions)
    if not env.agents:
        break
env.close()
```

### Sampling actions

The environment expects discrete integer actions in the range `[0, 4]` for each agent:

| Action | Effect             |
|--------|--------------------|
| 0      | No-op              |
| 1      | Move left (-X)     |
| 2      | Move right (+X)    |
| 3      | Move backward (-Y) |
| 4      | Move forward (+Y)  |

`SimpleTag3DBulletEnv` provides helpers `action_space_sample()` (single action) and `sample_actions()` (dictionary for every active agent) implemented via `numpy.random.Generator.integers`.

## Closing the environment

Always call `env.close()` when finished to ensure that the underlying PyBullet physics client is properly disconnected.

## License

This project is distributed under the same license as the original PettingZoo MPE environments (MIT).
