"""coevolution_scheduler.py
===========================
CoevolutionScheduler — lightweight evolutionary algorithm that tunes
plugin parameters over time.

No GL dependencies.  Pure Python + NumPy so it can run in either the main
thread or the physics worker.

Design
------
* Population: a list of N parameter dicts, one per individual.
* Fitness:    a float per individual (higher = better), accumulated over a
              sliding window of *fitness_window* evaluations.
* Per epoch:  elitist selection → single-point crossover → Gaussian mutation.

Config keys (from configs/duty_stations/simulator/coevolution.yaml)
------------------------------------------------------------------
population_size  int     number of individuals                 (default 8)
n_generations    int     max generations before stopping       (default 32)
mutation_rate    float   P(gene mutated)                       (default 0.12)
mutation_sigma   float   perturbation σ as fraction of range   (default 0.15)
crossover_rate   float   P(crossover between parent pair)      (default 0.55)
param_clamp      bool    clamp mutated values to [min, max]    (default True)
selection        str     "tournament" | "roulette" | "elitist" (default "tournament")
tournament_k     int     tournament group size                 (default 3)
elite_frac       float   fraction of top carried forward       (default 0.25)
fitness_window   int     frames averaged for fitness score     (default 8)

schedule.epoch_frames    int  sim frames per generation epoch  (default 64)
schedule.warmup_frames   int  frames before evolution starts   (default 16)
schedule.cooldown_frames int  frames between select & mutate   (default 8)
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Callable, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Individual helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_individual(param_specs: list[dict], rng: random.Random) -> dict:
    """Draw a random parameter dict within [min, max] for each spec."""
    ind: dict = {}
    for spec in param_specs:
        lo  = float(spec.get("min",     0.0))
        hi  = float(spec.get("max",     1.0))
        val = rng.uniform(lo, hi)
        ind[spec["name"]] = val
    return ind


def _crossover(a: dict, b: dict, rng: random.Random) -> tuple[dict, dict]:
    """Single-point crossover over the sorted key list."""
    keys = sorted(a.keys())
    n    = len(keys)
    if n < 2:
        return dict(a), dict(b)
    pt = rng.randint(1, n - 1)
    child_a = {k: (a[k] if i < pt else b[k]) for i, k in enumerate(keys)}
    child_b = {k: (b[k] if i < pt else a[k]) for i, k in enumerate(keys)}
    return child_a, child_b


def _mutate(
    ind: dict,
    param_specs: list[dict],
    rate: float,
    sigma: float,
    do_clamp: bool,
    rng: random.Random,
) -> dict:
    """Gaussian mutation with optional range clamping."""
    spec_map = {s["name"]: s for s in param_specs}
    result   = {}
    for k, v in ind.items():
        if rng.random() < rate:
            spec = spec_map.get(k, {})
            lo   = float(spec.get("min", 0.0))
            hi   = float(spec.get("max", 1.0))
            rng_w = hi - lo if hi > lo else 1.0
            v    += rng.gauss(0.0, sigma * rng_w)
            if do_clamp:
                v = max(lo, min(hi, v))
        result[k] = v
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class CoevolutionScheduler:
    """Evolutionary scheduler for plugin parameter tuning.

    Parameters
    ----------
    cfg:
        Configuration dict (from coevolution.yaml).
    param_specs:
        List of ``{name, min, max, default, ...}`` dicts drawn from the
        active plugin's ``PARAM_SPECS``.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        cfg: dict,
        param_specs: list[dict] | None = None,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # ── Configuration ─────────────────────────────────────────────────────
        self.population_size = int(cfg.get("population_size", 8))
        self.n_generations   = int(cfg.get("n_generations",   32))
        self.mutation_rate   = float(cfg.get("mutation_rate",   0.12))
        self.mutation_sigma  = float(cfg.get("mutation_sigma",  0.15))
        self.crossover_rate  = float(cfg.get("crossover_rate",  0.55))
        self.param_clamp     = bool(cfg.get("param_clamp",      True))
        self.elite_frac      = float(cfg.get("elite_frac",      0.25))
        self.fitness_window  = int(cfg.get("fitness_window",    8))
        self.selection_mode  = str(cfg.get("selection",         "tournament"))
        self.tournament_k    = int(cfg.get("tournament_k",      3))

        schedule = cfg.get("schedule", {})
        self.epoch_frames    = int(schedule.get("epoch_frames",    64))
        self.warmup_frames   = int(schedule.get("warmup_frames",   16))
        self.cooldown_frames = int(schedule.get("cooldown_frames",  8))

        self.param_specs: list[dict] = list(param_specs or [])

        # ── Runtime state ─────────────────────────────────────────────────────
        self.generation    = 0
        self.frame_counter = 0
        self._phase        = "warmup"   # warmup | running | cooldown | done
        self._cooldown_remaining = 0

        # Population: list of dicts; fitness_buf: deque per individual
        self.population: list[dict] = [
            _sample_individual(self.param_specs, self._rng)
            for _ in range(self.population_size)
        ]
        self._fitness_bufs: list[deque] = [
            deque(maxlen=self.fitness_window)
            for _ in range(self.population_size)
        ]
        self._current_idx = 0   # which individual is currently being evaluated

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active_params(self) -> dict:
        """Parameter dict for the individual currently under evaluation."""
        return self.population[self._current_idx]

    @property
    def is_done(self) -> bool:
        return self._phase == "done"

    def push_fitness(self, score: float) -> None:
        """Register one fitness observation for the current individual."""
        self._fitness_bufs[self._current_idx].append(float(score))

    def tick(self) -> None:
        """Advance the scheduler by one simulation frame.

        Call this once per physics step.  The scheduler manages its own state
        machine (warmup → running → cooldown → running → … → done).
        """
        self.frame_counter += 1

        if self._phase == "done":
            return

        if self._phase == "warmup":
            if self.frame_counter >= self.warmup_frames:
                self._phase = "running"
                self.frame_counter = 0
            return

        if self._phase == "cooldown":
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self._phase = "running"
                self.frame_counter = 0
            return

        # ── running ───────────────────────────────────────────────────────────
        # Advance the active individual pointer every epoch_frames frames.
        frames_per_ind = max(1, self.epoch_frames // self.population_size)
        if self.frame_counter >= frames_per_ind:
            self.frame_counter = 0
            self._current_idx = (self._current_idx + 1) % self.population_size
            if self._current_idx == 0:
                # Completed one pass over the whole population → evolve
                self._evolve()

    def step_epoch(
        self,
        fitness_fn: Callable[[dict], float] | None = None,
    ) -> list[dict]:
        """Manually run one full evolutionary epoch and return the new population.

        *fitness_fn*, if provided, scores each individual in the population.
        Otherwise uses the buffered scores collected by :meth:`push_fitness`.

        Returns the new population list (also stored in ``self.population``).
        """
        if fitness_fn is not None:
            for i, ind in enumerate(self.population):
                s = float(fitness_fn(ind))
                self._fitness_bufs[i].append(s)
        self._evolve()
        return self.population

    def get_stats(self) -> dict:
        """Return a snapshot of the current scheduler state."""
        scores = self._mean_scores()
        best_i = int(np.argmax(scores)) if scores else 0
        return {
            "generation":    self.generation,
            "phase":         self._phase,
            "frame_counter": self.frame_counter,
            "current_idx":   self._current_idx,
            "pop_size":      self.population_size,
            "mean_fitness":  float(np.mean(scores)) if scores else 0.0,
            "best_fitness":  float(max(scores)) if scores else 0.0,
            "best_params":   dict(self.population[best_i]) if self.population else {},
            "is_done":       self.is_done,
        }

    def reset(self, param_specs: list[dict] | None = None) -> None:
        """Reset the scheduler with (optionally new) param specs."""
        if param_specs is not None:
            self.param_specs = list(param_specs)
        self.generation    = 0
        self.frame_counter = 0
        self._phase        = "warmup"
        self._cooldown_remaining = 0
        self._current_idx  = 0
        self.population = [
            _sample_individual(self.param_specs, self._rng)
            for _ in range(self.population_size)
        ]
        self._fitness_bufs = [
            deque(maxlen=self.fitness_window)
            for _ in range(self.population_size)
        ]

    # ── Private ───────────────────────────────────────────────────────────────

    def _mean_scores(self) -> list[float]:
        return [
            float(np.mean(buf)) if buf else 0.0
            for buf in self._fitness_bufs
        ]

    def _select(self, scores: list[float]) -> dict:
        """Return one selected individual (does not remove from population)."""
        mode = self.selection_mode
        if mode == "roulette":
            return self._roulette_select(scores)
        if mode == "elitist":
            idx = int(np.argmax(scores))
            return dict(self.population[idx])
        # default: tournament
        return self._tournament_select(scores)

    def _tournament_select(self, scores: list[float]) -> dict:
        k    = min(self.tournament_k, len(self.population))
        idxs = self._rng.sample(range(len(self.population)), k)
        best = max(idxs, key=lambda i: scores[i])
        return dict(self.population[best])

    def _roulette_select(self, scores: list[float]) -> dict:
        arr   = np.array(scores, float)
        arr  -= arr.min()
        total = arr.sum()
        if total < 1e-12:
            return dict(self.population[self._rng.randrange(len(self.population))])
        probs = arr / total
        idx   = int(self._np_rng.choice(len(self.population), p=probs))
        return dict(self.population[idx])

    def _evolve(self) -> None:
        if self.generation >= self.n_generations:
            self._phase = "done"
            return

        scores    = self._mean_scores()
        n_elite   = max(1, int(math.ceil(self.elite_frac * self.population_size)))
        elite_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_elite]
        elites    = [dict(self.population[i]) for i in elite_idx]

        new_pop   = list(elites)
        while len(new_pop) < self.population_size:
            p1 = self._select(scores)
            p2 = self._select(scores)
            if self._rng.random() < self.crossover_rate and len(self.param_specs) >= 2:
                c1, c2 = _crossover(p1, p2, self._rng)
            else:
                c1, c2 = dict(p1), dict(p2)
            c1 = _mutate(c1, self.param_specs, self.mutation_rate,
                         self.mutation_sigma, self.param_clamp, self._rng)
            c2 = _mutate(c2, self.param_specs, self.mutation_rate,
                         self.mutation_sigma, self.param_clamp, self._rng)
            new_pop.append(c1)
            if len(new_pop) < self.population_size:
                new_pop.append(c2)

        self.population      = new_pop[:self.population_size]
        self._fitness_bufs   = [deque(maxlen=self.fitness_window)
                                 for _ in range(self.population_size)]
        self._current_idx    = 0
        self.generation     += 1
        self._phase          = "cooldown"
        self._cooldown_remaining = self.cooldown_frames

        if self.generation >= self.n_generations:
            self._phase = "done"
