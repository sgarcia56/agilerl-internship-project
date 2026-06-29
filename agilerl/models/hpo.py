from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class RLHyperparameter(BaseModel):
    """Min/max range and mutation factors for a single RL hyperparameter.

    :param min: Minimum value of the hyperparameter.
    :type min: float
    :param max: Maximum value of the hyperparameter.
    :type max: float
    :param grow_factor: Factor by which the hyperparameter will be grown during mutation.
    :type grow_factor: float
    :param shrink_factor: Factor by which the hyperparameter will be shrunk during mutation.
    :type shrink_factor: float
    """

    min: float
    max: float
    grow_factor: float = Field(default=1.2, ge=1.0)
    shrink_factor: float = Field(default=0.8, ge=0.0, le=1.0)


class MutationProbabilities(BaseModel):
    """Mutation probability distribution.

    :param no_mut: Probability of no mutation.
    :type no_mut: float
    :param arch_mut: Probability of architecture mutation.
    :type arch_mut: float
    :param new_layer: Probability of new layer mutation.
    :type new_layer: float
    :param params_mut: Probability of parameters mutation.
    :type params_mut: float
    :param act_mut: Probability of activation mutation.
    :type act_mut: float
    :param rl_hp_mut: Probability of RL hyperparameter mutation.
    :type rl_hp_mut: float
    """

    no_mut: float = Field(default=0.4, ge=0.0, le=1.0)
    arch_mut: float = Field(default=0.2, ge=0.0, le=1.0)
    new_layer: float = Field(default=0.2, ge=0.0, le=1.0)
    params_mut: float = Field(default=0.2, ge=0.0, le=1.0)
    act_mut: float = Field(default=0.0, ge=0.0, le=1.0)
    rl_hp_mut: float = Field(default=0.2, ge=0.0, le=1.0)


class MutationSpec(BaseModel):
    """Pydantic model for Mutations object.

    :param probabilities: Probability distribution for the mutations.
    :type probabilities: MutationProbabilities
    :param rl_hp_selection: RL hyperparameters to mutate.
    :type rl_hp_selection: dict[str, RLHyperparameter]
    :param mutation_sd: Standard deviation of the mutation.
    :type mutation_sd: float
    :param rand_seed: Random seed for repeatability.
    :type rand_seed: int
    :param mutate_elite: Whether the elite member of the population is itself mutated.
    :type mutate_elite: bool
    :param random_reset_param_mut: Whether the Gaussian parameter mutation keeps its
        random-reset band, which *replaces* a selected weight with a fresh
        ``N(0, 1)`` draw rather than perturbing it. Disabling it leaves that band's
        weights at their trained values; the band's probability mass is **not**
        redistributed to the other bands.
    :type random_reset_param_mut: bool
    :param amplified_gauss_param_mut: Whether the Gaussian parameter mutation keeps
        its amplified ("super") noise band, whose standard deviation is 10x the
        weight's magnitude. Disabling it leaves that band's weights at their trained
        values; the band's probability mass is **not** redistributed.
    :type amplified_gauss_param_mut: bool
    :param regrama_param_mut: Whether the parameter mutation resets dormant neurons
        (ReGraMa, "Measure gradients, not activations!") before the Gaussian pass,
        instead of relying on Gaussian noise alone. ``mutation_sd`` still scales the
        Gaussian pass's ordinary noise either way.
    :type regrama_param_mut: bool
    :param dormant_tau: ReGraMa dormancy threshold (a neuron with normalised score
        ``<= dormant_tau`` is dormant and is reset). Independent of the diagnostic
        ``training.dormant_tau``; only used when ``regrama_param_mut`` is set.
    :type dormant_tau: float
    :param regrama_out_scale: ReGraMa revival strength. A Xavier-reset neuron's
        outgoing weights are re-seeded at this fraction of the consumer layer's
        live column scale instead of being zeroed, so the revived neuron has a
        non-zero gradient (both for its own score and for its incoming weights).
        A scale below ``dormant_tau`` risks the revived neuron being re-flagged as
        dormant before it learns anything; the default trades that against the size
        of the perturbation, and was picked from a PPO/Hopper-v4 sweep (one seed) in
        which 0.02 beat the whole 0.05--0.25 range by a wide margin.
        ``0.0`` restores the zeroed-outgoing behaviour.
        Only used when ``regrama_param_mut`` is set.
    :type regrama_out_scale: float
    :param arch_mut_type: Architecture-mutation strategy: ``"original"`` (AgileRL's
        default add/remove node/channel/layer) or ``"func_preserving"``
        (function-preserving Net2Net-style *additions* -- new units are added with
        zero outgoing weights and new head layers are identity-initialised).
        Removals are the original random-count positional operator under both
        settings, so the two differ only in how capacity is added.
    :type arch_mut_type: Literal["original", "func_preserving"]
    :param arch_fp_noise: Symmetry-breaking noise scale for function-preserving
        additions (read *only* when ``arch_mut_type == "func_preserving"`` and an
        architecture add fires; completely inert otherwise, so its value is
        irrelevant for ``"original"`` runs). A relative factor ``alpha``: the new
        units' outgoing weights are seeded with ``randn * (alpha * sigma)`` where
        ``sigma`` is the std of the existing outgoing weights in that consuming
        layer. The default ``0.1`` breaks the new units' symmetry so they receive
        incoming-weight gradient and the added capacity is recruitable, at a
        negligible (~1%) function-preservation cost; set ``0.0`` for exact-zero,
        byte-identical preservation.
    :type arch_fp_noise: float
    :param arch_encoder_layer_mut: Whether ``add_layer`` / ``remove_layer`` are
        enabled on the *encoder* as well as the head. AgileRL disables encoder
        layer mutations by default because restructuring the encoder resets the
        representation feeding every head, which adds a lot of variance; a
        function-preserving deepening injects no such shock, so the default here
        is ``None`` -> ``arch_mut_type == "func_preserving"``. Set it explicitly
        to compare arms on an equal search space (an ``"original"`` baseline
        needs ``true`` to match a ``"func_preserving"`` arm). Only takes effect
        for **MLP** encoders; see :class:`EvolvableNetwork
        <agilerl.networks.base.EvolvableNetwork>`.
    :type arch_encoder_layer_mut: bool | None
    """

    model_config = ConfigDict(extra="forbid")

    probabilities: MutationProbabilities = Field(default_factory=MutationProbabilities)
    rl_hp_selection: dict[str, RLHyperparameter] = Field(default_factory=dict)
    mutation_sd: float = Field(default=0.1, ge=0.0)
    rand_seed: int = Field(default=42, ge=0)
    mutate_elite: bool = False
    random_reset_param_mut: bool = True
    amplified_gauss_param_mut: bool = True
    regrama_param_mut: bool = False
    dormant_tau: float = Field(default=0.1, gt=0.0)
    regrama_out_scale: float = Field(default=0.02, ge=0.0)
    arch_mut_type: Literal["original", "func_preserving"] = "original"
    arch_fp_noise: float = Field(default=0.1, ge=0.0)
    arch_encoder_layer_mut: bool | None = None

    def encoder_layer_mutations_enabled(self) -> bool:
        """Resolve whether encoder layer mutations should be enabled.

        ``arch_encoder_layer_mut`` is tri-state: an explicit ``True``/``False``
        wins, while ``None`` derives the value from the mutation strategy so that
        function-preserving runs get encoder deepening without a second knob.

        :return: Whether to enable encoder ``add_layer`` / ``remove_layer``.
        :rtype: bool
        """
        if self.arch_encoder_layer_mut is not None:
            return self.arch_encoder_layer_mut

        return self.arch_mut_type == "func_preserving"


class TournamentSelectionSpec(BaseModel):
    """Pydantic model for TournamentSelection object.

    :param tournament_size: Size of the tournament.
    :type tournament_size: int
    :param elitism: Whether elitism is enabled.
    :type elitism: bool
    """

    tournament_size: int = Field(default=2, ge=2)
    elitism: bool = True


class MFPBTSpec(BaseModel):
    """Pydantic model for the MF-PBT (Multiple-Frequencies PBT) evolution regime.

    MF-PBT replaces tournament-selection + mutation. The population is split into
    ``n_subpopulations`` subpopulations of ``n_individuals_per_subpopulation``
    agents each (so ``pop_size`` is derived, not configured). Each subpopulation
    ``i`` evolves every ``evolution_frequency_ratios[i]`` cycles, and is partitioned
    by fitness rank into four brackets whose sizes must sum to the per-subpopulation
    individual count. Following the paper, ``n_open_for_migration`` must not exceed
    ``n_winners + n_survivors`` so migration fills no more slots than the
    subpopulation preserves natively.

    :param n_subpopulations: Number of subpopulations.
    :type n_subpopulations: int
    :param n_individuals_per_subpopulation: Agents in each subpopulation.
    :type n_individuals_per_subpopulation: int
    :param evolution_frequency_ratios: Per-subpopulation evolution-frequency ratios
        (strictly increasing integers, each ``>= 1``; one per subpopulation).
    :type evolution_frequency_ratios: list[int]
    :param n_winners: Agents in the winners bracket.
    :type n_winners: int
    :param n_survivors: Agents in the survivors bracket.
    :type n_survivors: int
    :param n_open_for_migration: Agents in the open-for-migration bracket.
    :type n_open_for_migration: int
    :param n_losers: Agents in the losers bracket.
    :type n_losers: int
    :param rand_seed: Random seed for reproducible winner-clone selection.
    :type rand_seed: int
    """

    model_config = ConfigDict(extra="forbid")

    n_subpopulations: int = Field(default=4, ge=1)
    n_individuals_per_subpopulation: int = Field(default=4, ge=1)
    evolution_frequency_ratios: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    n_winners: int = Field(default=1, ge=0)
    n_survivors: int = Field(default=1, ge=0)
    n_open_for_migration: int = Field(default=1, ge=0)
    n_losers: int = Field(default=1, ge=0)
    rand_seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _validate_mf_pbt(self) -> Self:
        ratios = self.evolution_frequency_ratios
        if len(ratios) != self.n_subpopulations:
            msg = (
                f"evolution_frequency_ratios must have length n_subpopulations "
                f"({self.n_subpopulations}), got {len(ratios)}."
            )
            raise ValueError(msg)
        if any(r < 1 for r in ratios):
            msg = "Each evolution_frequency_ratio must be >= 1."
            raise ValueError(msg)
        if any(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1)):
            msg = "evolution_frequency_ratios must be strictly increasing."
            raise ValueError(msg)
        bracket_sum = (
            self.n_winners
            + self.n_survivors
            + self.n_open_for_migration
            + self.n_losers
        )
        if bracket_sum != self.n_individuals_per_subpopulation:
            msg = (
                f"n_winners + n_survivors + n_open_for_migration + n_losers "
                f"({bracket_sum}) must equal n_individuals_per_subpopulation "
                f"({self.n_individuals_per_subpopulation})."
            )
            raise ValueError(msg)
        # The winners bracket sources the replacement clones (evolution) and supplies
        # the subpopulation elite (migration), so it cannot be empty when either fires.
        if self.n_winners < 1 and (self.n_losers > 0 or self.n_open_for_migration > 0):
            msg = (
                "n_winners must be >= 1 when n_losers > 0 (winner-clones replace "
                "losers) or n_open_for_migration > 0 (the top winner is the "
                "subpopulation elite used for migration)."
            )
            raise ValueError(msg)
        # Migration may fill at most as many slots as the subpopulation preserves
        # natively (winners + survivors), so cross-frequency migration cannot replace
        # more of a subpopulation than it keeps -- the bracketing of the MF-PBT paper.
        if self.n_open_for_migration > self.n_winners + self.n_survivors:
            msg = (
                f"n_open_for_migration ({self.n_open_for_migration}) must be <= "
                f"n_winners + n_survivors ({self.n_winners} + {self.n_survivors} = "
                f"{self.n_winners + self.n_survivors}) so that migration fills no more "
                f"slots than the subpopulation preserves natively (MF-PBT paper)."
            )
            raise ValueError(msg)
        return self
