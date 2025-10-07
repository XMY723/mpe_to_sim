# PyBullet Simple Tag Simulation

This repository contains a PyBullet-based recreation of the
[`pettingzoo.mpe.simple_tag`](https://pettingzoo.farama.org/environments/mpe/simple_tag/)
scenario. The simulation keeps every entity constrained to the XY plane (no Z
axis actions) while exposing an interface similar to a multi-agent environment.

## Quickstart

```python
from mpe_to_sim import SimpleTagBulletSim

sim = SimpleTagBulletSim(gui=False)
obs = sim.reset()
for _ in range(100):
    actions = {agent: sim.sample_action() for agent in sim.agents}
    obs, rewards, dones, infos = sim.step(actions)
    if dones["__all__"]:
        break
sim.close()
```

Set `gui=True` when constructing `SimpleTagBulletSim` to visualise the
PyBullet simulation.
