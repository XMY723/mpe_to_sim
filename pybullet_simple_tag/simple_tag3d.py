"""PyBullet powered 3D rendition of the PettingZoo MPE Simple Tag scenario."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class EntityState:
    p_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    p_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    c: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


@dataclass
class Entity:
    name: str = ""
    collide: bool = False
    movable: bool = True
    size: float = 0.1
    color: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    state: EntityState = field(default_factory=EntityState)
    body_id: Optional[int] = None


@dataclass
class Agent(Entity):
    adversary: bool = False
    max_speed: float = 1.0
    accel: float = 3.0
    silent: bool = True


@dataclass
class Landmark(Entity):
    boundary: bool = False


class BulletWorld:
    """Container for PyBullet state."""

    def __init__(self, gui: bool = False, time_step: float = 1.0 / 30.0) -> None:
        self.dim_c = 3
        self.dim_p = 3
        self.gui = gui
        self.time_step = time_step
        self.client: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.agents: List[Agent] = []
        self.landmarks: List[Landmark] = []
        self.arena_radius = 1.2

    def connect(self) -> None:
        if self.client is not None:
            return
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

    def disconnect(self) -> None:
        if self.client is None:
            return
        p.disconnect(physicsClientId=self.client)
        self.client = None
        self.plane_id = None

    def reset(self) -> None:
        if self.client is None:
            raise RuntimeError("BulletWorld.reset() called before connect().")
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

    def spawn_entity(self, entity: Entity, color: Tuple[float, float, float]) -> None:
        if self.client is None:
            raise RuntimeError("BulletWorld.spawn_entity() called before connect().")
        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=entity.size,
            physicsClientId=self.client,
        )
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=entity.size,
            rgbaColor=(*color, 1.0),
            physicsClientId=self.client,
        )
        mass = 0.0 if not entity.movable else 1.0
        entity.body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=entity.state.p_pos,
            physicsClientId=self.client,
        )
        if not entity.movable:
            p.changeDynamics(entity.body_id, -1, lateralFriction=1.0, physicsClientId=self.client)

    def remove_entity(self, entity: Entity) -> None:
        if self.client is None or entity.body_id is None:
            return
        p.removeBody(entity.body_id, physicsClientId=self.client)
        entity.body_id = None

    def step(self) -> None:
        if self.client is None:
            raise RuntimeError("BulletWorld.step() called before connect().")
        p.stepSimulation(physicsClientId=self.client)
        for entity in self.agents + self.landmarks:
            if entity.body_id is None:
                continue
            pos, _ = p.getBasePositionAndOrientation(entity.body_id, physicsClientId=self.client)
            vel, _ = p.getBaseVelocity(entity.body_id, physicsClientId=self.client)
            entity.state.p_pos = np.asarray(pos, dtype=np.float32)
            entity.state.p_vel = np.asarray(vel, dtype=np.float32)


class Scenario:
    """3D Simple Tag scenario executed within PyBullet."""

    def make_world(
        self,
        num_good: int = 1,
        num_adversaries: int = 3,
        num_obstacles: int = 2,
        gui: bool = False,
        time_step: float = 1.0 / 30.0,
    ) -> BulletWorld:
        world = BulletWorld(gui=gui, time_step=time_step)
        world.connect()

        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles

        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = i < num_adversaries
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.size = 0.12 if agent.adversary else 0.08
            agent.accel = 3.0 if agent.adversary else 4.5
            agent.max_speed = 1.0 if agent.adversary else 1.4

        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.25
            landmark.boundary = False

        return world

    def reset_world(self, world: BulletWorld, rng: np.random.Generator) -> None:
        world.reset()
        for agent in world.agents:
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            agent.state = EntityState()
        for landmark in world.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state = EntityState()

        for agent in world.agents:
            pos = rng.uniform(-1.0, 1.0, size=3)
            pos[2] = rng.uniform(0.1, 0.5)
            agent.state.p_pos = pos.astype(np.float32)
            agent.state.p_vel = np.zeros(3, dtype=np.float32)
            agent.state.c = np.zeros(3, dtype=np.float32)
            world.spawn_entity(agent, tuple(agent.color))
            self._sync_pose(agent, world)

        for landmark in world.landmarks:
            pos = rng.uniform(-0.9, 0.9, size=3)
            pos[2] = rng.uniform(0.1, 0.6)
            landmark.state.p_pos = pos.astype(np.float32)
            landmark.state.p_vel = np.zeros(3, dtype=np.float32)
            world.spawn_entity(landmark, tuple(landmark.color))
            self._sync_pose(landmark, world)

    def _sync_pose(self, entity: Entity, world: BulletWorld) -> None:
        if world.client is None or entity.body_id is None:
            return
        p.resetBasePositionAndOrientation(
            entity.body_id,
            entity.state.p_pos,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=world.client,
        )
        p.resetBaseVelocity(
            entity.body_id,
            linearVelocity=entity.state.p_vel,
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=world.client,
        )

    def good_agents(self, world: BulletWorld) -> List[Agent]:
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world: BulletWorld) -> List[Agent]:
        return [agent for agent in world.agents if agent.adversary]

    def is_collision(self, entity_a: Entity, entity_b: Entity) -> bool:
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = entity_a.size + entity_b.size
        return dist < dist_min

    def reward(self, agent: Agent, world: BulletWorld) -> float:
        if agent.adversary:
            return self.adversary_reward(agent, world)
        return self.agent_reward(agent, world)

    def agent_reward(self, agent: Agent, world: BulletWorld) -> float:
        reward = 0.0
        if agent.collide:
            for adversary in self.adversaries(world):
                if self.is_collision(agent, adversary):
                    reward -= 10.0

        def bound(x: float) -> float:
            if x < 0.9:
                return 0.0
            if x < 1.0:
                return (x - 0.9) * 10.0
            return min(np.exp(2.0 * x - 2.0), 10.0)

        for axis in range(world.dim_p):
            reward -= bound(abs(float(agent.state.p_pos[axis])))
        return reward

    def adversary_reward(self, agent: Agent, world: BulletWorld) -> float:
        reward = 0.0
        agents = self.good_agents(world)
        if agent.collide:
            for other_agent in agents:
                if self.is_collision(other_agent, agent):
                    reward += 10.0
        return reward

    def observation(self, agent: Agent, world: BulletWorld) -> np.ndarray:
        entity_pos = [landmark.state.p_pos - agent.state.p_pos for landmark in world.landmarks]
        other_pos: List[np.ndarray] = []
        other_vel: List[np.ndarray] = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        components: List[np.ndarray] = [
            agent.state.p_vel,
            agent.state.p_pos,
        ]
        if entity_pos:
            components.extend(entity_pos)
        if other_pos:
            components.extend(other_pos)
        if other_vel:
            components.extend(other_vel)
        if not components:
            return np.array([], dtype=np.float32)
        return np.concatenate(components).astype(np.float32)


class SimpleTag3DBulletEnv:
    """Parallel-style multi-agent environment for the 3D Simple Tag scenario."""

    metadata = {"name": "simple_tag_3d_bullet", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(
        self,
        num_good: int = 1,
        num_adversaries: int = 3,
        num_obstacles: int = 2,
        max_cycles: int = 50,
        gui: bool = False,
        time_step: float = 1.0 / 30.0,
        seed: Optional[int] = None,
    ) -> None:
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.num_obstacles = num_obstacles
        self.max_cycles = max_cycles
        self.gui = gui
        self.time_step = time_step
        self.scenario = Scenario()
        self.world = self.scenario.make_world(
            num_good,
            num_adversaries,
            num_obstacles,
            gui=gui,
            time_step=time_step,
        )
        self._rng = np.random.default_rng(seed)
        self._elapsed_steps = 0
        self.possible_agents = [agent.name for agent in self.world.agents]
        self.agents = self.possible_agents.copy()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._elapsed_steps = 0
        for entity in self.world.agents + self.world.landmarks:
            if entity.body_id is not None:
                self.world.remove_entity(entity)
        self.scenario.reset_world(self.world, self._rng)
        observations = {agent.name: self.scenario.observation(agent, self.world) for agent in self.world.agents}
        infos = {agent.name: {} for agent in self.world.agents}
        self.agents = self.possible_agents.copy()
        return observations, infos

    def close(self) -> None:
        self.world.disconnect()

    def _action_to_velocity(self, action: int, agent: Agent) -> np.ndarray:
        directions = {
            0: np.zeros(3, dtype=np.float32),
            1: np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            3: np.array([0.0, -1.0, 0.0], dtype=np.float32),
            4: np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        vec = directions.get(int(action), np.zeros(3, dtype=np.float32))
        speed = agent.max_speed
        return vec * speed

    def step(
        self, actions: Mapping[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, float]],
    ]:
        if not self.agents:
            raise RuntimeError("step() called after all agents are done.")
        for agent in self.world.agents:
            action = actions.get(agent.name, 0)
            velocity = self._action_to_velocity(action, agent)
            self._apply_velocity(agent, velocity)
        self.world.step()
        self._enforce_bounds()
        observations = {agent.name: self.scenario.observation(agent, self.world) for agent in self.world.agents}
        rewards = {agent.name: self.scenario.reward(agent, self.world) for agent in self.world.agents}
        terminations = {agent.name: False for agent in self.world.agents}
        self._elapsed_steps += 1
        truncation = self._elapsed_steps >= self.max_cycles
        truncations = {agent.name: truncation for agent in self.world.agents}
        infos = {agent.name: {} for agent in self.world.agents}
        if truncation:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def action_space_sample(self) -> int:
        """Returns a uniformly sampled discrete action in the valid range."""

        return int(self._rng.integers(0, 5))

    def sample_actions(self) -> Dict[str, int]:
        """Convenience helper that samples an action for every active agent."""

        return {agent: self.action_space_sample() for agent in self.agents}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _apply_velocity(self, agent: Agent, velocity: np.ndarray) -> None:
        if self.world.client is None or agent.body_id is None:
            return
        clipped = np.clip(velocity, -agent.max_speed, agent.max_speed)
        p.resetBaseVelocity(
            agent.body_id,
            linearVelocity=clipped,
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self.world.client,
        )

    def _enforce_bounds(self) -> None:
        if self.world.client is None:
            return
        bounds = self.world.arena_radius
        for agent in self.world.agents:
            if agent.body_id is None:
                continue
            pos = agent.state.p_pos.copy()
            clipped = np.clip(pos, -bounds, bounds)
            if not np.allclose(clipped, pos):
                agent.state.p_pos = clipped
                agent.state.p_vel = np.zeros(3, dtype=np.float32)
                p.resetBasePositionAndOrientation(
                    agent.body_id,
                    clipped,
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=self.world.client,
                )
                p.resetBaseVelocity(
                    agent.body_id,
                    linearVelocity=[0.0, 0.0, 0.0],
                    angularVelocity=[0.0, 0.0, 0.0],
                    physicsClientId=self.world.client,
                )


def make_env(**kwargs: object) -> SimpleTag3DBulletEnv:
    """Factory helper mirroring PettingZoo's API."""

    return SimpleTag3DBulletEnv(**kwargs)
