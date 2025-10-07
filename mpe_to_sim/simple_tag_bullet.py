"""PyBullet simulation that mirrors the PettingZoo ``simple_tag`` scenario.

The original PettingZoo environment (``pettingzoo.mpe.simple_tag``) models a
2-D continuous world where a team of taggers attempt to catch a single runner
while navigating around fixed landmarks. This module offers a light-weight
PyBullet implementation of that scenario where all motion happens on the
``x``/``y`` plane (no agent can move along the ``z`` axis).

Example
-------
>>> from mpe_to_sim import SimpleTagBulletSim
>>> sim = SimpleTagBulletSim(gui=False)
>>> observations = sim.reset()
>>> for _ in range(100):
...     actions = {agent: sim.sample_action() for agent in sim.agents}
...     observations, rewards, dones, infos = sim.step(actions)
...     if dones["__all__"]:
...         break
>>> sim.close()

The environment is intentionally deterministic given an RNG seed and mirrors the
reward structure used by PettingZoo:

* Taggers receive dense negative rewards proportional to their distance to the
  runner and a sparse +1 reward when the runner is tagged.
* The runner receives the opposite dense reward and a -1 penalty when tagged.

Observation vectors are made of the agent's velocity, absolute position, the
relative positions of landmarks, and the relative positions of other agents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data


Vector2 = Tuple[float, float]


@dataclass
class Body:
    """Container for simulated bodies (agents or landmarks)."""

    name: str
    body_id: int
    radius: float
    is_runner: bool = False


class SimpleTagBulletSim:
    """PyBullet based re-implementation of the PettingZoo ``simple_tag`` task."""

    def __init__(
        self,
        num_taggers: int = 3,
        num_runners: int = 1,
        num_landmarks: int = 2,
        max_speed: float = 1.3,
        arena_radius: float = 1.2,
        tag_radius: float = 0.1,
        time_step: float = 1.0 / 60.0,
        frame_skip: int = 4,
        gui: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.num_taggers = num_taggers
        self.num_runners = num_runners
        self.num_landmarks = num_landmarks
        self.max_speed = max_speed
        self.arena_radius = arena_radius
        self.tag_radius = tag_radius
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.gui = gui
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(time_step)
        p.setGravity(0.0, 0.0, 0.0)

        self._z_height = 0.05
        self._plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self._plane_id, -1, lateralFriction=0.0)

        self.taggers: List[Body] = []
        self.runners: List[Body] = []
        self.landmarks: List[Body] = []
        self.agents: List[str] = []

        self._build_world()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Disconnects from the PyBullet client."""

        if p.isConnected(self._client):
            p.disconnect(self._client)

    # ------------------------------------------------------------------
    def reset(self) -> Mapping[str, np.ndarray]:
        """Reset the simulation and return initial observations."""

        self._reset_entities()
        return {agent.name: self._observe(agent) for agent in self._iter_agents()}

    # ------------------------------------------------------------------
    def step(
        self, actions: Mapping[str, Iterable[float]]
    ) -> Tuple[Mapping[str, np.ndarray], Mapping[str, float], Mapping[str, bool], Mapping[str, dict]]:
        """Advance the simulation using per-agent 2-D actions.

        Parameters
        ----------
        actions:
            Mapping from agent name to a 2-D action (desired velocity on ``x``
            and ``y``). The values are clipped to ``[-1, 1]`` and scaled by
            ``max_speed``.
        """

        for agent in self._iter_agents():
            ax, ay = self._extract_action(actions.get(agent.name))
            p.resetBaseVelocity(agent.body_id, [ax, ay, 0.0], [0.0, 0.0, 0.0])

        for _ in range(self.frame_skip):
            p.stepSimulation()
            self._enforce_planar_motion()

        observations = {agent.name: self._observe(agent) for agent in self._iter_agents()}
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        infos: Dict[str, dict] = {agent.name: {} for agent in self._iter_agents()}
        infos["__all__"] = {"tagged": dones.get("__all__", False)}
        return observations, rewards, dones, infos

    # ------------------------------------------------------------------
    def sample_action(self) -> np.ndarray:
        """Draw a random action from a uniform distribution in [-1, 1]."""

        return self._rng.uniform(low=-1.0, high=1.0, size=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_world(self) -> None:
        """Create all rigid bodies used in the simulation."""

        sphere_collision = lambda radius: p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        def create_body(name: str, radius: float, mass: float, color: Tuple[float, float, float, float]) -> int:
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            collision = sphere_collision(radius)
            body = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[0.0, 0.0, self._z_height],
            )
            p.changeDynamics(body, -1, linearDamping=2.0, angularDamping=1.0, lateralFriction=0.0)
            return body

        self.taggers = []
        self.runners = []
        self.landmarks = []

        for idx in range(self.num_taggers):
            body_id = create_body(f"tagger_{idx}", radius=0.05, mass=1.0, color=(0.8, 0.1, 0.1, 1.0))
            self.taggers.append(Body(name=f"tagger_{idx}", body_id=body_id, radius=0.05, is_runner=False))

        for idx in range(self.num_runners):
            body_id = create_body(f"runner_{idx}", radius=0.055, mass=0.8, color=(0.1, 0.8, 0.2, 1.0))
            self.runners.append(Body(name=f"runner_{idx}", body_id=body_id, radius=0.055, is_runner=True))

        for idx in range(self.num_landmarks):
            body_id = create_body(f"landmark_{idx}", radius=0.07, mass=0.0, color=(0.2, 0.2, 0.8, 1.0))
            self.landmarks.append(Body(name=f"landmark_{idx}", body_id=body_id, radius=0.07))
            p.changeDynamics(body_id, -1, linearDamping=0.0)

        self.agents = [body.name for body in (*self.taggers, *self.runners)]

        self._reset_entities()

    # ------------------------------------------------------------------
    def _reset_entities(self) -> None:
        """Position all entities randomly inside the arena."""

        for body in self.landmarks:
            position = self._sample_position(radius_margin=body.radius)
            p.resetBasePositionAndOrientation(body.body_id, [*position, self._z_height], [0.0, 0.0, 0.0, 1.0])
            p.resetBaseVelocity(body.body_id, [0.0, 0.0, 0.0])

        for body in (*self.taggers, *self.runners):
            position = self._sample_position(radius_margin=body.radius)
            p.resetBasePositionAndOrientation(body.body_id, [*position, self._z_height], [0.0, 0.0, 0.0, 1.0])
            p.resetBaseVelocity(body.body_id, [0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    def _sample_position(self, radius_margin: float) -> Vector2:
        radius = self._rng.uniform(low=0.0, high=max(self.arena_radius - radius_margin, 0.01))
        angle = self._rng.uniform(low=0.0, high=2 * math.pi)
        return radius * math.cos(angle), radius * math.sin(angle)

    # ------------------------------------------------------------------
    def _extract_action(self, action: Optional[Iterable[float]]) -> Vector2:
        if action is None:
            ax, ay = 0.0, 0.0
        else:
            ax, ay = action
        ax = float(np.clip(ax, -1.0, 1.0)) * self.max_speed
        ay = float(np.clip(ay, -1.0, 1.0)) * self.max_speed
        return ax, ay

    # ------------------------------------------------------------------
    def _observe(self, body: Body) -> np.ndarray:
        pos = np.array(p.getBasePositionAndOrientation(body.body_id)[0][:2], dtype=np.float32)
        vel = np.array(p.getBaseVelocity(body.body_id)[0][:2], dtype=np.float32)

        other_agents: List[np.ndarray] = []
        other_vels: List[np.ndarray] = []
        for other in self._iter_agents():
            if other.body_id == body.body_id:
                continue
            other_pos = np.array(p.getBasePositionAndOrientation(other.body_id)[0][:2], dtype=np.float32)
            other_vel = np.array(p.getBaseVelocity(other.body_id)[0][:2], dtype=np.float32)
            other_agents.append(other_pos - pos)
            other_vels.append(other_vel)

        landmark_vectors: List[np.ndarray] = []
        for landmark in self.landmarks:
            landmark_pos = np.array(p.getBasePositionAndOrientation(landmark.body_id)[0][:2], dtype=np.float32)
            landmark_vectors.append(landmark_pos - pos)

        obs = [vel, pos]
        if landmark_vectors:
            obs.append(np.concatenate(landmark_vectors))
        if other_agents:
            obs.append(np.concatenate(other_agents))
            obs.append(np.concatenate(other_vels))
        return np.concatenate(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    def _compute_rewards(self) -> Mapping[str, float]:
        rewards: Dict[str, float] = {}
        runner_positions = [self._body_position(runner) for runner in self.runners]
        tagger_positions = [self._body_position(tagger) for tagger in self.taggers]

        for tagger, tagger_pos in zip(self.taggers, tagger_positions):
            closest_distance = min(
                np.linalg.norm(tagger_pos - runner_pos)
                for runner_pos in runner_positions
            )
            reward = -closest_distance
            if self._is_tagger_success(tagger_pos, runner_positions):
                reward += 1.0
            rewards[tagger.name] = reward

        for runner, runner_pos in zip(self.runners, runner_positions):
            closest_distance = min(
                np.linalg.norm(runner_pos - tagger_pos)
                for tagger_pos in tagger_positions
            )
            reward = closest_distance
            if self._is_runner_caught(runner_pos, tagger_positions):
                reward -= 1.0
            rewards[runner.name] = reward

        return rewards

    # ------------------------------------------------------------------
    def _compute_dones(self) -> Mapping[str, bool]:
        dones: Dict[str, bool] = {agent.name: False for agent in self._iter_agents()}
        tagged = any(
            self._is_runner_caught(self._body_position(runner), [self._body_position(tagger) for tagger in self.taggers])
            for runner in self.runners
        )
        if tagged:
            for agent in dones:
                dones[agent] = True
        dones["__all__"] = tagged
        return dones

    # ------------------------------------------------------------------
    def _body_position(self, body: Body) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(body.body_id)[0][:2], dtype=np.float32)

    # ------------------------------------------------------------------
    def _is_runner_caught(self, runner_pos: np.ndarray, tagger_positions: Iterable[np.ndarray]) -> bool:
        for tagger_pos in tagger_positions:
            if np.linalg.norm(tagger_pos - runner_pos) <= self.tag_radius:
                return True
        return False

    # ------------------------------------------------------------------
    def _is_tagger_success(self, tagger_pos: np.ndarray, runner_positions: Iterable[np.ndarray]) -> bool:
        for runner_pos in runner_positions:
            if np.linalg.norm(tagger_pos - runner_pos) <= self.tag_radius:
                return True
        return False

    # ------------------------------------------------------------------
    def _iter_agents(self) -> Iterable[Body]:
        yield from self.taggers
        yield from self.runners

    # ------------------------------------------------------------------
    def _enforce_planar_motion(self) -> None:
        for body in self._iter_agents():
            pos, orn = p.getBasePositionAndOrientation(body.body_id)
            x, y = pos[:2]
            p.resetBasePositionAndOrientation(body.body_id, [x, y, self._z_height], orn)
            vel_lin, vel_ang = p.getBaseVelocity(body.body_id)
            vx, vy = vel_lin[:2]
            p.resetBaseVelocity(body.body_id, [vx, vy, 0.0], [0.0, 0.0, 0.0])


__all__ = ["SimpleTagBulletSim"]
