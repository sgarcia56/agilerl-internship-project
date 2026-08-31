"""End-to-end smoke test for MF-PBT wiring through ``LocalTrainer``.

Exercises the full path — manifest spec -> ``build_mf_pbt_from_spec`` ->
``create_population_from_spec`` subpopulation tagging -> ``MFPBT.evolve_population``
running on *real* PPO agents (clone / mutation / migration), without a heavy
training loop. Complements the fake-agent unit tests in
``tests/test_hpo/test_mf_pbt.py``.
"""

from __future__ import annotations

import warnings
from collections import Counter
from unittest.mock import patch

from gymnasium.spaces import Box, Discrete

from agilerl.algorithms.core.base import EvolvableAlgorithm
from agilerl.hpo.mf_pbt import MFPBT
from agilerl.hpo.mutation import Mutations
from agilerl.models import PPOSpec
from agilerl.models.env import GymEnvSpec
from agilerl.models.hpo import (
    MFPBTSpec,
    MutationProbabilities,
    MutationSpec,
    RLHyperparameter,
)
from agilerl.models.networks import MlpSpec, StochasticActorSpec
from agilerl.models.training import TrainingSpec
from agilerl.training.trainer import LocalTrainer


class _DummyEnv:
    """Minimal CartPole-like vector env exposing the spaces the trainer needs."""

    def __init__(self) -> None:
        self.name = "CartPole-v1"
        self.observation_space = Box(low=-1, high=1, shape=(4,))
        self.action_space = Discrete(2)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_envs = 1

    def close(self):
        pass


def _make_trainer() -> LocalTrainer:
    mf_pbt = MFPBTSpec(
        n_subpopulations=2,
        n_individuals_per_subpopulation=4,
        evolution_frequency_ratios=[1, 2],
        n_winners=1,
        n_survivors=1,
        n_open_for_migration=1,
        n_losers=1,
    )
    mutation = MutationSpec(
        probabilities=MutationProbabilities(no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2),
        rl_hp_selection={
            "lr": RLHyperparameter(min=1e-5, max=1e-2),
            # learn_step is mutable in the real configs and resizes the rollout buffer.
            "learn_step": RLHyperparameter(min=64, max=2048),
        },
    )
    ppo = PPOSpec(
        learn_step=128,
        net_config=StochasticActorSpec(
            encoder_config=MlpSpec(hidden_size=[16]),
            head_config=MlpSpec(hidden_size=[16]),
        ),
    )
    with patch.object(LocalTrainer, "_make_env", return_value=_DummyEnv()):
        return LocalTrainer(
            algorithm=ppo,
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
            training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=8),
            mutation=mutation,
            mf_pbt=mf_pbt,
        )


def test_localtrainer_builds_mfpbt_and_tags_subpopulations():
    trainer = _make_trainer()

    assert isinstance(trainer.mf_pbt, MFPBT)
    assert trainer.tournament_selection is None  # MF-PBT replaces tournament
    subpops = sorted(a.subpopulation for a in trainer.population)
    assert subpops == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len({a.index for a in trainer.population}) == 8


def test_mfpbt_trainer_to_manifest_round_trips():
    # Regression: trainer.train() calls to_manifest(), which rebuilds a
    # TrainingManifest from the live specs where training.pop_size is now the
    # concrete derived value. The manifest validator must not reject its own
    # derived pop_size on this round-trip.
    trainer = _make_trainer()

    manifest = trainer.to_manifest()  # must not raise

    assert manifest["training"]["pop_size"] == 8  # 2 subpops x 4 individuals
    assert "mf_pbt" in manifest
    assert "tournament_selection" not in manifest


def test_migration_reset_hp_resizes_rollout_buffer():
    # A slow-to-fast migrant clones the external agent's networks but takes the studied
    # subpop elite's mutable HPs, including learn_step. PPO sizes its rollout buffer
    # from learn_step via a mutation hook, so the migrant's buffer (sized to the
    # external agent's learn_step at clone time) MUST be rebuilt to match the elite's
    # learn_step, or rollout collection overflows it.
    trainer = _make_trainer()
    agents = list(trainer.population)
    external, elite = agents[4], agents[0]  # different subpopulations

    external.learn_step = 256
    external.mutation_hook()  # buffer sized to 256
    elite.learn_step = 512
    elite.mutation_hook()  # buffer sized to 512

    trainer.mf_pbt._sync_index(agents)  # migration() does this before _migrate_*
    migrant = trainer.mf_pbt._migrate_reset_hp(external, elite, subpop=1)

    assert migrant.learn_step == 512  # took the elite's learn_step
    # Buffer capacity must follow the new learn_step (ceil(learn_step / num_envs)).
    assert migrant.rollout_buffer.capacity == -(512 // -migrant.num_envs)


def test_mfpbt_evolves_real_population_across_cycles():
    trainer = _make_trainer()
    agents = list(trainer.population)

    # Run several evolution cycles. delta=[1,2] => subpop 0 evolves every cycle and
    # subpop 1 every other cycle, so both the evolution and (full-clone and
    # slow-to-fast) migration branches are exercised on real PPO agents.
    for cycle in range(3):
        for offset, agent in enumerate(agents):
            agent.fitness = [float((cycle + offset) % 8)]
        agents = trainer.mf_pbt.evolve_population(
            agents, mutation=trainer.mutations, env_name="CartPole-v1", algo="PPO"
        )

        assert len(agents) == 8
        counts = Counter(a.subpopulation for a in agents)
        assert counts[0] == 4 and counts[1] == 4
        assert len({a.index for a in agents}) == 8  # indices stay unique
        assert all(isinstance(a, EvolvableAlgorithm) for a in agents)


def _make_regrama_trainer() -> LocalTrainer:
    """A trainer whose parameter mutation is ReGraMa, always fired."""
    mf_pbt = MFPBTSpec(
        n_subpopulations=2,
        n_individuals_per_subpopulation=4,
        evolution_frequency_ratios=[1, 2],
        n_winners=1,
        n_survivors=1,
        n_open_for_migration=1,
        n_losers=1,
    )
    mutation = MutationSpec(
        # Force the parameter mutation so every winner-clone takes the ReGraMa path.
        probabilities=MutationProbabilities(
            no_mut=0.0, arch_mut=0.0, new_layer=0.0, params_mut=1.0, rl_hp_mut=0.0
        ),
        regrama_param_mut=True,
    )
    ppo = PPOSpec(
        learn_step=128,
        net_config=StochasticActorSpec(
            encoder_config=MlpSpec(hidden_size=[16]),
            head_config=MlpSpec(hidden_size=[16]),
        ),
    )
    with patch.object(LocalTrainer, "_make_env", return_value=_DummyEnv()):
        return LocalTrainer(
            algorithm=ppo,
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
            training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=8),
            mutation=mutation,
            mf_pbt=mf_pbt,
        )


def test_mfpbt_routes_the_grama_side_table_to_regrama():
    # MF-PBT clones winners over losers and perturbs the clones. Without the
    # parent tag and the side table reaching Mutations, ReGraMa cannot resolve a
    # clone's snapshot and silently degrades to the Gaussian operator.
    trainer = _make_regrama_trainer()
    agents = list(trainer.population)
    for offset, agent in enumerate(agents):
        agent.fitness = [float(offset)]

    # Sentinel snapshots: truthy, and distinguishable per parent.
    side_table = {agent.index: [[agent.index]] for agent in agents}

    seen = []

    def _spy(self, individual, grama_scores):
        seen.append((individual._parent_index, grama_scores))
        return individual

    with (
        patch.object(Mutations, "regrama_parameter_mutation", _spy),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        trainer.mf_pbt.evolve_population(
            agents,
            mutation=trainer.mutations,
            env_name="CartPole-v1",
            algo="PPO",
            grama_scores=side_table,
        )

    # ReGraMa ran, and each clone was scored against its own parent's snapshot.
    assert seen
    assert all(scores == side_table[parent] for parent, scores in seen)
    # No agent degraded to the Gaussian operator.
    assert not [w for w in caught if "falling back to the Gaussian" in str(w.message)]


def test_mfpbt_without_a_side_table_falls_back_loudly():
    # The contract the wiring protects: an unthreaded side table is a silent
    # downgrade to Gaussian noise, so it must warn rather than pass unnoticed.
    trainer = _make_regrama_trainer()
    agents = list(trainer.population)
    for offset, agent in enumerate(agents):
        agent.fitness = [float(offset)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.mf_pbt.evolve_population(
            agents,
            mutation=trainer.mutations,
            env_name="CartPole-v1",
            algo="PPO",
        )

    assert [w for w in caught if "regrama_param_mut is set" in str(w.message)]
