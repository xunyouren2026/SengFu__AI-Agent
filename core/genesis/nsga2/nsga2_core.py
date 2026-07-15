"""
NSGA-II / NSGA-III / MOEA/D -- Complete Implementation
=======================================================
Pure Python implementation (math + random only).
No external dependencies beyond the standard library.

References:
  [1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
      A fast and elitist multiobjective genetic algorithm: NSGA-II.
      IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
  [2] Deb, K., & Jain, H. (2014).
      An evolutionary many-objective optimization algorithm using
      reference-point-based nondominated sorting approach, Part I.
      IEEE Transactions on Evolutionary Computation, 18(4), 577-601.
  [3] Zhang, Q., & Li, H. (2007).
      MOEA/D: A multiobjective evolutionary algorithm based on decomposition.
      IEEE Transactions on Evolutionary Computation, 11(6), 712-731.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Individual
# ---------------------------------------------------------------------------

class Individual:
    """A single solution in the population."""

    __slots__ = (
        "genes",
        "objectives",
        "constraints",
        "rank",
        "crowding_distance",
        "domination_count",
        "dominated_set",
        "constraint_violation",
    )

    def __init__(
        self,
        genes: Optional[List[float]] = None,
        objectives: Optional[List[float]] = None,
        constraints: Optional[List[float]] = None,
    ) -> None:
        self.genes: List[float] = genes if genes is not None else []
        self.objectives: List[float] = objectives if objectives is not None else []
        self.constraints: List[float] = constraints if constraints is not None else []
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.domination_count: int = 0
        self.dominated_set: List[int] = []  # indices into the population list
        self.constraint_violation: float = 0.0

    def compute_constraint_violation(self) -> None:
        """Sum of positive constraint violations (<=0 means satisfied)."""
        self.constraint_violation = sum(max(0.0, c) for c in self.constraints)

    def dominates(self, other: "Individual") -> bool:
        """
        Pareto domination check (constraint-free version).
        self dominates other iff self is no worse in every objective
        and strictly better in at least one.
        """
        at_least_one_better = False
        for a, b in zip(self.objectives, other.objectives):
            if a > b:  # assuming minimisation
                return False
            if a < b:
                at_least_one_better = True
        return at_least_one_better

    def constrained_dominates(self, other: "Individual") -> bool:
        """
        Constrained domination principle (Deb et al., 2002).
        1. Feasible solution always dominates infeasible.
        2. Among infeasible, lower total constraint violation wins.
        3. Among feasible, standard Pareto domination applies.
        """
        cv_self = self.constraint_violation
        cv_other = other.constraint_violation

        if cv_self == 0.0 and cv_other > 0.0:
            return True
        if cv_self > 0.0 and cv_other == 0.0:
            return False
        if cv_self > 0.0 and cv_other > 0.0:
            return cv_self < cv_other
        return self.dominates(other)

    def __lt__(self, other: "Individual") -> bool:
        """Compare by (rank, -crowding_distance) for sorting."""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.crowding_distance > other.crowding_distance

    def copy(self) -> "Individual":
        ind = Individual(
            genes=list(self.genes),
            objectives=list(self.objectives),
            constraints=list(self.constraints),
        )
        ind.rank = self.rank
        ind.crowding_distance = self.crowding_distance
        ind.domination_count = self.domination_count
        ind.dominated_set = list(self.dominated_set)
        ind.constraint_violation = self.constraint_violation
        return ind


# ---------------------------------------------------------------------------
# 2. NSGA2Config
# ---------------------------------------------------------------------------

@dataclass
class NSGA2Config:
    """Configuration for NSGA-II / NSGA-III / MOEA/D."""

    population_size: int = 100
    num_generations: int = 200
    crossover_prob: float = 0.9
    mutation_prob: float = 1.0 / 30.0
    num_objectives: int = 2
    num_variables: int = 30
    variable_bounds: List[Tuple[float, float]] = field(default_factory=list)
    crossover_type: str = "sbx"       # "sbx" or "uniform"
    mutation_type: str = "polynomial"  # "polynomial" or "gaussian"
    sbx_eta: float = 20.0             # distribution index for SBX
    pm_eta: float = 20.0              # distribution index for polynomial mutation
    gaussian_sigma: float = 0.1       # initial sigma for Gaussian mutation
    tournament_size: int = 2
    seed: Optional[int] = None

    # NSGA-III specific
    num_reference_points: int = 12
    reference_point_divisions: int = 12

    # MOEA/D specific
    moead_neighborhood_size: int = 20
    moead_max_replacements: int = 2

    def __post_init__(self) -> None:
        if not self.variable_bounds:
            self.variable_bounds = [(0.0, 1.0)] * self.num_variables
        if self.seed is not None:
            random.seed(self.seed)


# ---------------------------------------------------------------------------
# 3. SBXCrossover -- Simulated Binary Crossover
# ---------------------------------------------------------------------------

class SBXCrossover:
    """
    Simulated Binary Crossover (SBX).

    For each gene i:
        beta_q is sampled from the polynomial distribution:
            p(beta) = 0.5*(eta+1)*beta^eta            if beta <= 1
            p(beta) = 0.5*(eta+1)/beta^(eta+2)        if beta > 1

        Children:
            c1_i = 0.5 * [(1 + beta_q) * p1_i + (1 - beta_q) * p2_i]
            c2_i = 0.5 * [(1 - beta_q) * p1_i + (1 + beta_q) * p2_i]
    """

    def __init__(self, eta: float = 20.0, prob: float = 0.9) -> None:
        self.eta = eta
        self.prob = prob

    def _beta_q(self) -> float:
        u = random.random()
        eta = self.eta
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        return beta

    def cross(
        self, parent1: Individual, parent2: Individual, bounds: List[Tuple[float, float]]
    ) -> Tuple[Individual, Individual]:
        child1_genes = list(parent1.genes)
        child2_genes = list(parent2.genes)
        for i in range(len(parent1.genes)):
            if random.random() > self.prob:
                continue
            if abs(parent1.genes[i] - parent2.genes[i]) < 1e-14:
                continue
            beta = self._beta_q()
            c1 = 0.5 * ((1.0 + beta) * parent1.genes[i] + (1.0 - beta) * parent2.genes[i])
            c2 = 0.5 * ((1.0 - beta) * parent1.genes[i] + (1.0 + beta) * parent2.genes[i])
            lo, hi = bounds[i]
            child1_genes[i] = max(lo, min(hi, c1))
            child2_genes[i] = max(lo, min(hi, c2))
        c1_ind = Individual(genes=child1_genes, constraints=list(parent1.constraints))
        c2_ind = Individual(genes=child2_genes, constraints=list(parent2.constraints))
        return c1_ind, c2_ind


# ---------------------------------------------------------------------------
# 4. PolynomialMutation
# ---------------------------------------------------------------------------

class PolynomialMutation:
    """
    Polynomial Mutation.

    For each gene i, with probability p_m:
        delta_q is sampled from:
            if u < 0.5:
                delta = (2*u)^(1/(eta+1)) - 1
            else:
                delta = 1 - (2*(1-u))^(1/(eta+1))

        child_i = parent_i + delta * (upper_i - lower_i)
    """

    def __init__(self, eta: float = 20.0, prob: float = 0.067) -> None:
        self.eta = eta
        self.prob = prob

    def mutate(self, individual: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        genes = list(individual.genes)
        eta = self.eta
        for i in range(len(genes)):
            if random.random() > self.prob:
                continue
            lo, hi = bounds[i]
            y = genes[i]
            delta_lo = y - lo
            delta_hi = hi - y
            if delta_lo < 1e-14 and delta_hi < 1e-14:
                continue
            u = random.random()
            if u < 0.5:
                delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            y_new = y + delta * (hi - lo)
            genes[i] = max(lo, min(hi, y_new))
        return Individual(genes=genes, constraints=list(individual.constraints))


# ---------------------------------------------------------------------------
# 5. UniformCrossover
# ---------------------------------------------------------------------------

class UniformCrossover:
    """Standard uniform crossover -- each gene swapped with probability 0.5."""

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def cross(
        self, parent1: Individual, parent2: Individual, bounds: List[Tuple[float, float]]
    ) -> Tuple[Individual, Individual]:
        g1, g2 = list(parent1.genes), list(parent2.genes)
        for i in range(len(g1)):
            if random.random() < self.prob:
                g1[i], g2[i] = g2[i], g1[i]
        c1 = Individual(genes=g1, constraints=list(parent1.constraints))
        c2 = Individual(genes=g2, constraints=list(parent2.constraints))
        return c1, c2


# ---------------------------------------------------------------------------
# 6. GaussianMutation
# ---------------------------------------------------------------------------

class GaussianMutation:
    """
    Gaussian noise mutation with adaptive sigma.

    child_i = parent_i + N(0, sigma * (upper_i - lower_i))
    """

    def __init__(self, sigma: float = 0.1, prob: float = 0.067) -> None:
        self.sigma = sigma
        self.prob = prob

    def mutate(self, individual: Individual, bounds: List[Tuple[float, float]]) -> Individual:
        genes = list(individual.genes)
        for i in range(len(genes)):
            if random.random() > self.prob:
                continue
            lo, hi = bounds[i]
            width = hi - lo
            noise = random.gauss(0.0, self.sigma * width)
            genes[i] = max(lo, min(hi, genes[i] + noise))
        return Individual(genes=genes, constraints=list(individual.constraints))


# ---------------------------------------------------------------------------
# 7. NonDominatedSort -- Fast non-dominated sorting O(MN^2)
# ---------------------------------------------------------------------------

def non_dominated_sort(
    population: List[Individual],
    use_constraints: bool = True,
) -> List[List[int]]:
    """
    Fast non-dominated sorting.

    Returns a list of fronts, where each front is a list of indices
    into *population*.

    Complexity: O(M * N^2) where M = #objectives, N = pop size.
    """
    n = len(population)
    domination_func = (
        (lambda a, b: a.constrained_dominates(b))
        if use_constraints
        else (lambda a, b: a.dominates(b))
    )

    for ind in population:
        ind.domination_count = 0
        ind.dominated_set = []

    # Build domination lists -- O(M * N^2)
    for i in range(n):
        for j in range(i + 1, n):
            if domination_func(population[i], population[j]):
                population[i].dominated_set.append(j)
                population[j].domination_count += 1
            elif domination_func(population[j], population[i]):
                population[j].dominated_set.append(i)
                population[i].domination_count += 1

    fronts: List[List[int]] = []
    # First front
    front0 = [i for i in range(n) if population[i].domination_count == 0]
    fronts.append(front0)

    current_front = front0
    while current_front:
        next_front: List[int] = []
        for i in current_front:
            for j in population[i].dominated_set:
                population[j].domination_count -= 1
                if population[j].domination_count == 0:
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        current_front = next_front

    # Assign ranks
    for rank, front in enumerate(fronts):
        for idx in front:
            population[idx].rank = rank

    return fronts


# ---------------------------------------------------------------------------
# 8. CrowdingDistance
# ---------------------------------------------------------------------------

def crowding_distance_assignment(front: List[Individual]) -> None:
    """
    Assign crowding distance to every individual in *front*.

    For each objective m:
        Sort front by objective m.
        Boundary individuals get infinite distance.
        For interior individuals:
            cd_i += (f_{m+1} - f_{m-1}) / (f_max - f_min)
    """
    n = len(front)
    if n <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    num_obj = len(front[0].objectives)
    for ind in front:
        ind.crowding_distance = 0.0

    for m in range(num_obj):
        front.sort(key=lambda ind: ind.objectives[m])
        f_min = front[0].objectives[m]
        f_max = front[-1].objectives[m]
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        if f_max - f_min < 1e-14:
            continue
        for i in range(1, n - 1):
            front[i].crowding_distance += (
                front[i + 1].objectives[m] - front[i - 1].objectives[m]
            ) / (f_max - f_min)


# ---------------------------------------------------------------------------
# 9. TournamentSelection
# ---------------------------------------------------------------------------

class TournamentSelection:
    """
    Binary tournament selection using the crowded-comparison operator.

    crowded_comparison(a, b):
        if a.rank < b.rank  ->  a wins
        if a.rank > b.rank  ->  b wins
        if a.cd > b.cd      ->  a wins
        else                ->  b wins
    """

    def __init__(self, tournament_size: int = 2) -> None:
        self.tournament_size = tournament_size

    @staticmethod
    def crowded_comparison(a: Individual, b: Individual) -> bool:
        """Return True if *a* is better than *b*."""
        if a.rank < b.rank:
            return True
        if a.rank > b.rank:
            return False
        return a.crowding_distance > b.crowding_distance

    def select(self, population: List[Individual]) -> Individual:
        candidates = random.sample(population, min(self.tournament_size, len(population)))
        best = candidates[0]
        for c in candidates[1:]:
            if self.crowded_comparison(c, best):
                best = c
        return best


# ---------------------------------------------------------------------------
# 10. ConstraintHandling
# ---------------------------------------------------------------------------

class ConstraintHandling:
    """
    Constraint handling utilities.

    Uses the constrained domination principle as defined in Individual.
    """

    @staticmethod
    def is_feasible(individual: Individual) -> bool:
        return individual.constraint_violation <= 0.0

    @staticmethod
    def total_violation(individual: Individual) -> float:
        return individual.constraint_violation

    @staticmethod
    def compute_violations(population: List[Individual]) -> None:
        for ind in population:
            ind.compute_constraint_violation()

    @staticmethod
    def epsilon_feasibility(individual: Individual, epsilon: float = 1e-6) -> bool:
        return individual.constraint_violation <= epsilon


# ---------------------------------------------------------------------------
# 11. NSGA2 -- Main algorithm
# ---------------------------------------------------------------------------

class NSGA2:
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II).

    Usage::

        config = NSGA2Config(num_objectives=2, num_variables=30, ...)
        nsga2 = NSGA2(config, objective_func)
        pareto_front = nsga2.run()
    """

    def __init__(
        self,
        config: NSGA2Config,
        objective_func: Callable[[List[float]], Tuple[List[float], List[float]]],
    ) -> None:
        """
        Parameters
        ----------
        config : NSGA2Config
            Algorithm configuration.
        objective_func : callable(genes) -> (objectives, constraints)
            Must return a tuple of (list of objective values, list of constraint values).
            Constraints are of the form g(x) <= 0.
        """
        self.config = config
        self.objective_func = objective_func
        self.population: List[Individual] = []
        self.generation = 0
        self._rng = random.Random(config.seed)

        # Build crossover / mutation operators
        self.crossover = self._build_crossover()
        self.mutation = self._build_mutation()
        self.selector = TournamentSelection(config.tournament_size)

    # -- operator factories --------------------------------------------------

    def _build_crossover(self):
        cfg = self.config
        if cfg.crossover_type == "sbx":
            return SBXCrossover(eta=cfg.sbx_eta, prob=cfg.crossover_prob)
        elif cfg.crossover_type == "uniform":
            return UniformCrossover(prob=cfg.crossover_prob)
        raise ValueError(f"Unknown crossover type: {cfg.crossover_type}")

    def _build_mutation(self):
        cfg = self.config
        if cfg.mutation_type == "polynomial":
            return PolynomialMutation(eta=cfg.pm_eta, prob=cfg.mutation_prob)
        elif cfg.mutation_type == "gaussian":
            return GaussianMutation(sigma=cfg.gaussian_sigma, prob=cfg.mutation_prob)
        raise ValueError(f"Unknown mutation type: {cfg.mutation_type}")

    # -- initialisation ------------------------------------------------------

    def _random_genes(self) -> List[float]:
        genes = []
        for lo, hi in self.config.variable_bounds:
            genes.append(lo + self._rng.random() * (hi - lo))
        return genes

    def initialize_population(self) -> None:
        """Create initial random population and evaluate."""
        self.population = []
        for _ in range(self.config.population_size):
            genes = self._random_genes()
            ind = Individual(genes=genes)
            self.population.append(ind)
        self._evaluate_population(self.population)

    # -- evaluation ----------------------------------------------------------

    def _evaluate_population(self, pop: List[Individual]) -> None:
        for ind in pop:
            objs, cons = self.objective_func(ind.genes)
            ind.objectives = list(objs)
            ind.constraints = list(cons)
            ind.compute_constraint_violation()

    # -- selection -----------------------------------------------------------

    def selection(self) -> List[Individual]:
        """Select parents via tournament."""
        return [self.selector.select(self.population) for _ in range(self.config.population_size)]

    # -- offspring creation --------------------------------------------------

    def create_offspring(self, parents: List[Individual]) -> List[Individual]:
        """Create offspring population from selected parents."""
        offspring: List[Individual] = []
        bounds = self.config.variable_bounds
        i = 0
        while i + 1 < len(parents):
            p1, p2 = parents[i], parents[i + 1]
            c1, c2 = self.crossover.cross(p1, p2, bounds)
            c1 = self.mutation.mutate(c1, bounds)
            c2 = self.mutation.mutate(c2, bounds)
            offspring.append(c1)
            offspring.append(c2)
            i += 2
        # If odd number of parents, handle the last one
        if i < len(parents):
            c = parents[i].copy()
            c = self.mutation.mutate(c, bounds)
            offspring.append(c)
        return offspring

    # -- environmental selection (elitist merging) ---------------------------

    def environmental_selection(
        self, combined: List[Individual]
    ) -> List[Individual]:
        """
        Select next generation from R_t = P_t union Q_t.

        1. Non-dominated sort of R_t.
        2. Fill new population front by front.
        3. For the last front that would overflow, use crowding distance.
        """
        use_constraints = any(
            ind.constraints and any(c != 0.0 for c in ind.constraints)
            for ind in combined
        )
        fronts = non_dominated_sort(combined, use_constraints=use_constraints)

        new_pop: List[Individual] = []
        target = self.config.population_size

        for front_indices in fronts:
            front = [combined[i] for i in front_indices]
            if len(new_pop) + len(front) <= target:
                crowding_distance_assignment(front)
                new_pop.extend(front)
            else:
                # Need to partially fill from this front
                crowding_distance_assignment(front)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                remaining = target - len(new_pop)
                new_pop.extend(front[:remaining])
                break

        return new_pop

    # -- main loop -----------------------------------------------------------

    def run(self) -> List[Individual]:
        """
        Execute the full NSGA-II evolutionary loop.

        Returns the final Pareto front (rank 0 individuals).
        """
        self.initialize_population()

        for gen in range(self.config.num_generations):
            self.generation = gen

            # Selection
            parents = self.selection()

            # Variation
            offspring = self.create_offspring(parents)
            self._evaluate_population(offspring)

            # Environmental selection
            combined = self.population + offspring
            self.population = self.environmental_selection(combined)

        return self.get_pareto_front()

    # -- post-processing -----------------------------------------------------

    def get_pareto_front(self) -> List[Individual]:
        """Return non-dominated solutions from the final population."""
        use_constraints = any(
            ind.constraints and any(c != 0.0 for c in ind.constraints)
            for ind in self.population
        )
        fronts = non_dominated_sort(self.population, use_constraints=use_constraints)
        if not fronts:
            return []
        return [self.population[i] for i in fronts[0]]


# ---------------------------------------------------------------------------
# 12. NSGA-III -- Reference-point based NSGA
# ---------------------------------------------------------------------------

def generate_reference_points(num_objectives: int, divisions: int) -> List[List[float]]:
    """
    Generate Das-Dennis systematic reference points on the unit simplex.

    Each reference point is a vector of length num_objectives that sums to 1.

    Uses recursive construction:
        For M objectives and p divisions, we generate C(p+M-1, M-1) points.
    """
    points: List[List[float]] = []

    def _recurse(point: List[float], remaining: int, idx: int) -> None:
        if idx == num_objectives - 1:
            point.append(float(remaining))
            points.append(list(point))
            point.pop()
            return
        for v in range(remaining + 1):
            point.append(float(v) / float(divisions))
            _recurse(point, remaining - v, idx + 1)
            point.pop()

    _recurse([], divisions, 0)
    return points


def normalize_objectives(population: List[Individual], num_obj: int) -> Tuple[List[List[float]], List[float], List[float]]:
    """
    Normalize objectives to [0, 1] range using min-max scaling
    with ideal point and nadir point estimation.
    """
    ideal = [float("inf")] * num_obj
    nadir = [float("-inf")] * num_obj
    for ind in population:
        for m in range(num_obj):
            if ind.objectives[m] < ideal[m]:
                ideal[m] = ind.objectives[m]
            if ind.objectives[m] > nadir[m]:
                nadir[m] = ind.objectives[m]

    # Handle degenerate case where ideal == nadir
    for m in range(num_obj):
        if nadir[m] - ideal[m] < 1e-14:
            nadir[m] = ideal[m] + 1.0

    normalized = []
    for ind in population:
        norm_obj = [
            (ind.objectives[m] - ideal[m]) / (nadir[m] - ideal[m])
            for m in range(num_obj)
        ]
        normalized.append(norm_obj)

    return normalized, ideal, nadir


def perpendicular_distance(point: List[float], ref: List[float]) -> float:
    """Compute the perpendicular distance from *point* to the line from origin through *ref*."""
    dot = sum(p * r for p, r in zip(point, ref))
    ref_norm_sq = sum(r * r for r in ref)
    if ref_norm_sq < 1e-14:
        return math.sqrt(sum(p * p for p in point))
    proj_scalar = dot / ref_norm_sq
    proj = [proj_scalar * r for r in ref]
    dist = math.sqrt(sum((p - pr) ** 2 for p, pr in zip(point, proj)))
    return dist


class NSGA3:
    """
    NSGA-III: Many-objective optimisation using reference points.

    Extends NSGA-II with reference-point-based non-dominated sorting
    and niche-preservation to handle many-objective problems (M > 3).

    Reference:
        Deb, K., & Jain, H. (2014). IEEE TEVC, 18(4), 577-601.
    """

    def __init__(
        self,
        config: NSGA2Config,
        objective_func: Callable[[List[float]], Tuple[List[float], List[float]]],
    ) -> None:
        self.config = config
        self.objective_func = objective_func
        self.population: List[Individual] = []
        self.generation = 0

        self.crossover = self._build_crossover()
        self.mutation = self._build_mutation()
        self.selector = TournamentSelection(config.tournament_size)

        # Generate reference points
        self.reference_points = generate_reference_points(
            config.num_objectives, config.reference_point_divisions
        )
        # Niche counts
        self.niche_counts = [0] * len(self.reference_points)

    def _build_crossover(self):
        cfg = self.config
        if cfg.crossover_type == "sbx":
            return SBXCrossover(eta=cfg.sbx_eta, prob=cfg.crossover_prob)
        return UniformCrossover(prob=cfg.crossover_prob)

    def _build_mutation(self):
        cfg = self.config
        if cfg.mutation_type == "polynomial":
            return PolynomialMutation(eta=cfg.pm_eta, prob=cfg.mutation_prob)
        return GaussianMutation(sigma=cfg.gaussian_sigma, prob=cfg.mutation_prob)

    def _random_genes(self) -> List[float]:
        genes = []
        for lo, hi in self.config.variable_bounds:
            genes.append(lo + random.random() * (hi - lo))
        return genes

    def initialize_population(self) -> None:
        self.population = []
        for _ in range(self.config.population_size):
            genes = self._random_genes()
            ind = Individual(genes=genes)
            self.population.append(ind)
        self._evaluate_population(self.population)

    def _evaluate_population(self, pop: List[Individual]) -> None:
        for ind in pop:
            objs, cons = self.objective_func(ind.genes)
            ind.objectives = list(objs)
            ind.constraints = list(cons)
            ind.compute_constraint_violation()

    def _associate_to_reference_points(
        self,
        pop: List[Individual],
        normalized: List[List[float]],
    ) -> Tuple[List[int], List[float]]:
        """
        Associate each individual to its closest reference point.
        Returns (association list, distance list).
        """
        associations = [0] * len(pop)
        distances = [0.0] * len(pop)
        for i, norm_obj in enumerate(normalized):
            min_dist = float("inf")
            min_idx = 0
            for j, ref in enumerate(self.reference_points):
                d = perpendicular_distance(norm_obj, ref)
                if d < min_dist:
                    min_dist = d
                    min_idx = j
            associations[i] = min_idx
            distances[i] = min_dist
        return associations, distances

    def _niching_selection(
        self,
        candidates: List[Individual],
        candidate_normalized: List[List[float]],
        associations: List[int],
        distances: List[float],
        n_select: int,
    ) -> List[Individual]:
        """
        Select *n_select* individuals using niche preservation.
        """
        selected: List[Individual] = []
        niche_counts = [0] * len(self.reference_points)

        # Build association map: ref_idx -> list of candidate indices
        assoc_map: List[List[int]] = [[] for _ in range(len(self.reference_points))]
        for i, ref_idx in enumerate(associations):
            assoc_map[ref_idx].append(i)

        # Shuffle candidates for fairness
        indices = list(range(len(candidates)))
        random.shuffle(indices)

        selected_set = set()

        # First pass: fill one per reference point that has candidates
        for j in range(len(self.reference_points)):
            if len(selected) >= n_select:
                break
            if not assoc_map[j]:
                continue
            # Pick the closest candidate to this reference point
            best_idx = min(assoc_map[j], key=lambda i: distances[i])
            if best_idx not in selected_set:
                selected.append(candidates[best_idx])
                selected_set.add(best_idx)
                niche_counts[j] += 1

        # Second pass: fill remaining from reference points with smallest niche count
        remaining_indices = [i for i in indices if i not in selected_set]
        random.shuffle(remaining_indices)

        for i in remaining_indices:
            if len(selected) >= n_select:
                break
            ref_idx = associations[i]
            # Find reference point with minimum niche count among those
            # associated with this candidate
            min_niche = niche_counts[ref_idx]
            selected.append(candidates[i])
            selected_set.add(i)
            niche_counts[ref_idx] += 1

        return selected[:n_select]

    def _reference_point_based_selection(
        self,
        combined: List[Individual],
        st_front_indices: List[int],
    ) -> List[Individual]:
        """
        Perform NSGA-III reference-point-based selection for the last front.
        """
        target = self.config.population_size
        num_obj = self.config.num_objectives

        # Normalize objectives using all individuals in previous fronts + st_front
        # First, figure out how many we already have from earlier fronts
        # This is handled externally; here we assume we need to select
        # from st_front_indices to fill the remaining slots.

        # Normalize using the entire combined population
        normalized_all, ideal, nadir = normalize_objectives(combined, num_obj)

        st_front = [combined[i] for i in st_front_indices]
        st_normalized = [normalized_all[i] for i in st_front_indices]

        # Associate
        associations, distances = self._associate_to_reference_points(
            st_front, st_normalized
        )

        n_needed = target  # will be trimmed externally
        return self._niching_selection(
            st_front, st_normalized, associations, distances, n_needed
        )

    def run(self) -> List[Individual]:
        """Execute NSGA-III."""
        self.initialize_population()
        target = self.config.population_size

        for gen in range(self.config.num_generations):
            self.generation = gen

            # Selection
            parents = [self.selector.select(self.population) for _ in range(target)]

            # Variation
            offspring = []
            bounds = self.config.variable_bounds
            i = 0
            while i + 1 < len(parents):
                p1, p2 = parents[i], parents[i + 1]
                c1, c2 = self.crossover.cross(p1, p2, bounds)
                c1 = self.mutation.mutate(c1, bounds)
                c2 = self.mutation.mutate(c2, bounds)
                offspring.append(c1)
                offspring.append(c2)
                i += 2
            self._evaluate_population(offspring)

            # Combine
            combined = self.population + offspring
            use_constraints = any(
                ind.constraints and any(c != 0.0 for c in ind.constraints)
                for ind in combined
            )
            fronts = non_dominated_sort(combined, use_constraints=use_constraints)

            # Fill new population front by front
            new_pop: List[Individual] = []
            last_front_idx = -1
            for fi, front_indices in enumerate(fronts):
                if len(new_pop) + len(front_indices) <= target:
                    front = [combined[i] for i in front_indices]
                    crowding_distance_assignment(front)
                    new_pop.extend(front)
                else:
                    last_front_idx = fi
                    break

            if last_front_idx >= 0:
                # Need NSGA-III niching for the last front
                remaining = target - len(new_pop)
                st_front_indices = fronts[last_front_idx]
                selected = self._reference_point_based_selection(combined, st_front_indices)
                new_pop.extend(selected[:remaining])

            self.population = new_pop[:target]

        return self.get_pareto_front()

    def get_pareto_front(self) -> List[Individual]:
        use_constraints = any(
            ind.constraints and any(c != 0.0 for c in ind.constraints)
            for ind in self.population
        )
        fronts = non_dominated_sort(self.population, use_constraints=use_constraints)
        if not fronts:
            return []
        return [self.population[i] for i in fronts[0]]


# ---------------------------------------------------------------------------
# 13. MOEA/D -- Multi-Objective Evolutionary Algorithm based on Decomposition
# ---------------------------------------------------------------------------

class MOEAD:
    """
    MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.

    Decomposes a multi-objective problem into N single-objective
    subproblems using weight vectors and the Tchebycheff approach.

    Reference:
        Zhang, Q., & Li, H. (2007). IEEE TEVC, 11(6), 712-731.
    """

    def __init__(
        self,
        config: NSGA2Config,
        objective_func: Callable[[List[float]], Tuple[List[float], List[float]]],
    ) -> None:
        self.config = config
        self.objective_func = objective_func
        self.population: List[Individual] = []
        self.generation = 0
        self.num_objectives = config.num_objectives
        self.pop_size = config.population_size

        # Weight vectors
        self.weight_vectors = self._generate_weight_vectors()

        # Neighborhood
        self.neighborhood_size = config.moead_neighborhood_size
        self.neighborhoods = self._compute_neighborhoods()

        # Reference point (ideal point)
        self.ideal_point = [float("inf")] * self.num_objectives

        # Operators
        self.crossover = self._build_crossover()
        self.mutation = self._build_mutation()

    def _build_crossover(self):
        cfg = self.config
        if cfg.crossover_type == "sbx":
            return SBXCrossover(eta=cfg.sbx_eta, prob=cfg.crossover_prob)
        return UniformCrossover(prob=cfg.crossover_prob)

    def _build_mutation(self):
        cfg = self.config
        if cfg.mutation_type == "polynomial":
            return PolynomialMutation(eta=cfg.pm_eta, prob=cfg.mutation_prob)
        return GaussianMutation(sigma=cfg.gaussian_sigma, prob=cfg.mutation_prob)

    def _generate_weight_vectors(self) -> List[List[float]]:
        """
        Generate uniform weight vectors using the simplex-lattice design.
        For M objectives and H divisions, generates C(H+M-1, M-1) vectors.
        """
        M = self.num_objectives
        H = max(1, self.pop_size - 1)  # approximate
        # Use a simple recursive method
        vectors: List[List[float]] = []

        def _recurse(vec: List[float], remaining: int, idx: int) -> None:
            if idx == M - 1:
                vec.append(float(remaining) / float(H))
                vectors.append(list(vec))
                vec.pop()
                return
            for v in range(remaining + 1):
                vec.append(float(v) / float(H))
                _recurse(vec, remaining - v, idx + 1)
                vec.pop()

        _recurse([], H, 0)

        # If we have too many, sample down to pop_size
        if len(vectors) > self.pop_size:
            random.shuffle(vectors)
            vectors = vectors[: self.pop_size]
        # If too few, duplicate some
        while len(vectors) < self.pop_size:
            vectors.append(list(vectors[len(vectors) % max(1, len(vectors) - 1)]))

        return vectors[: self.pop_size]

    def _compute_neighborhoods(self) -> List[List[int]]:
        """Compute T nearest weight vectors for each subproblem."""
        T = self.neighborhood_size
        neighborhoods: List[List[int]] = []
        for i in range(self.pop_size):
            distances = []
            for j in range(self.pop_size):
                if i == j:
                    continue
                d = sum(
                    (a - b) ** 2
                    for a, b in zip(self.weight_vectors[i], self.weight_vectors[j])
                )
                distances.append((d, j))
            distances.sort()
            neighborhoods.append([j for _, j in distances[:T]])
        return neighborhoods

    def _tchebycheff(self, f: List[float], weight: List[float], z_ideal: List[float]) -> float:
        """
        Tchebycheff scalarising function:
            g(x | lambda, z*) = max_{m=1..M} { lambda_m * |f_m(x) - z*_m| }
        """
        max_val = -float("inf")
        for m in range(self.num_objectives):
            val = weight[m] * abs(f[m] - z_ideal[m])
            if val > max_val:
                max_val = val
        return max_val

    def _random_genes(self) -> List[float]:
        genes = []
        for lo, hi in self.config.variable_bounds:
            genes.append(lo + random.random() * (hi - lo))
        return genes

    def initialize_population(self) -> None:
        self.population = []
        for _ in range(self.pop_size):
            genes = self._random_genes()
            ind = Individual(genes=genes)
            self.population.append(ind)
        self._evaluate_population(self.population)
        self._update_ideal_point()

    def _evaluate_population(self, pop: List[Individual]) -> None:
        for ind in pop:
            objs, cons = self.objective_func(ind.genes)
            ind.objectives = list(objs)
            ind.constraints = list(cons)
            ind.compute_constraint_violation()

    def _update_ideal_point(self) -> None:
        for ind in self.population:
            for m in range(self.num_objectives):
                if ind.objectives[m] < self.ideal_point[m]:
                    self.ideal_point[m] = ind.objectives[m]

    def run(self) -> List[Individual]:
        """Execute MOEA/D."""
        self.initialize_population()
        bounds = self.config.variable_bounds
        max_replacements = self.config.moead_max_replacements

        for gen in range(self.config.num_generations):
            self.generation = gen

            # Permutation of subproblem indices
            perm = list(range(self.pop_size))
            random.shuffle(perm)

            for i in perm:
                # Select parents from neighborhood
                neighbors = self.neighborhoods[i]
                parent_indices = random.sample(
                    neighbors, min(2, len(neighbors))
                )
                p1 = self.population[parent_indices[0]]
                p2 = self.population[parent_indices[1]]

                # Crossover
                c1, c2 = self.crossover.cross(p1, p2, bounds)

                # Mutation
                c1 = self.mutation.mutate(c1, bounds)
                c2 = self.mutation.mutate(c2, bounds)

                # Evaluate children
                for child in (c1, c2):
                    self._evaluate_population([child])

                    # Update ideal point
                    for m in range(self.num_objectives):
                        if child.objectives[m] < self.ideal_point[m]:
                            self.ideal_point[m] = child.objectives[m]

                    # Update neighboring subproblems
                    replacements = 0
                    for j in neighbors:
                        if replacements >= max_replacements:
                            break
                        g_current = self._tchebycheff(
                            self.population[j].objectives,
                            self.weight_vectors[j],
                            self.ideal_point,
                        )
                        g_child = self._tchebycheff(
                            child.objectives,
                            self.weight_vectors[j],
                            self.ideal_point,
                        )
                        if g_child <= g_current:
                            self.population[j] = child.copy()
                            replacements += 1

        return self.get_pareto_front()

    def get_pareto_front(self) -> List[Individual]:
        """Return non-dominated solutions from the final population."""
        use_constraints = any(
            ind.constraints and any(c != 0.0 for c in ind.constraints)
            for ind in self.population
        )
        fronts = non_dominated_sort(self.population, use_constraints=use_constraints)
        if not fronts:
            return []
        return [self.population[i] for i in fronts[0]]


# ---------------------------------------------------------------------------
# 14. Hypervolume -- Quality indicator
# ---------------------------------------------------------------------------

class Hypervolume:
    """
    Hypervolume (S-metric) indicator for Pareto front quality assessment.

    Computes the volume of objective space dominated by the Pareto front
    and bounded by a reference point.

    Supports 2D and 3D exact computation.
    """

    def __init__(self, reference_point: List[float]) -> None:
        """
        Parameters
        ----------
        reference_point : list of float
            The anti-optimal (nadir) reference point.
            Must be dominated by no solution in the Pareto front
            (i.e., worse than every solution in every objective).
        """
        self.reference_point = list(reference_point)
        self.dim = len(reference_point)

    def compute_2d(self, front: List[Individual]) -> float:
        """
        Compute 2D hypervolume using the sweep-line algorithm.

        Sort by first objective, then accumulate rectangles.
        """
        if not front:
            return 0.0

        points = [(ind.objectives[0], ind.objectives[1]) for ind in front]
        points.sort()

        ref_x, ref_y = self.reference_point[0], self.reference_point[1]
        hv = 0.0
        prev_x = points[0][0]
        prev_y = ref_y

        for x, y in points:
            if x >= ref_x or y >= ref_y:
                continue
            hv += (x - prev_x) * (ref_y - prev_y)
            prev_x = x
            prev_y = y

        # Last segment
        hv += (ref_x - prev_x) * (ref_y - prev_y)
        return hv

    def compute_3d(self, front: List[Individual]) -> float:
        """
        Compute 3D hypervolume using the recursive dimension-sweep algorithm.

        Algorithm:
            1. Sort points by first objective.
            2. For each consecutive pair, compute the volume slice
               as width * hypervolume_2d of the remaining objectives.
        """
        if not front:
            return 0.0

        points = [
            (ind.objectives[0], ind.objectives[1], ind.objectives[2])
            for ind in front
        ]
        # Filter out points outside reference
        ref = self.reference_point
        points = [
            p for p in points
            if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]
        ]
        if not points:
            return 0.0

        points.sort()

        hv = 0.0
        n = len(points)

        for i in range(n):
            # Width in the first dimension
            if i == 0:
                width = points[0][0] - 0.0  # assume origin as lower bound
            else:
                width = points[i][0] - points[i - 1][0]

            # Collect points that are not dominated in the remaining dimensions
            # For the 2D hypervolume of (f2, f3), we need the Pareto front
            # in those dimensions considering only points from i..n-1
            remaining = [(points[j][1], points[j][2]) for j in range(i, n)]
            # Sort by f2, compute 2D hypervolume
            remaining.sort()
            slice_hv = 0.0
            prev_y = remaining[0][0]
            prev_z = ref[2]
            for y, z in remaining:
                if y >= ref[1] or z >= ref[2]:
                    continue
                slice_hv += (y - prev_y) * (ref[2] - prev_z)
                prev_y = y
                prev_z = z
            slice_hv += (ref[1] - prev_y) * (ref[2] - prev_z)

            hv += width * slice_hv

        return hv

    def compute(self, front: List[Individual]) -> float:
        """Compute hypervolume, dispatching to the correct dimension."""
        if self.dim == 2:
            return self.compute_2d(front)
        elif self.dim == 3:
            return self.compute_3d(front)
        else:
            # General dimension hypervolume using Monte Carlo estimation
            return self._compute_monte_carlo(front)

    def _compute_monte_carlo(self, front: List[Individual], n_samples: int = 100000) -> float:
        """
        Monte Carlo estimation of hypervolume for arbitrary dimensions.

        Samples uniformly from the hyper-rectangle defined by the reference
        point and the Pareto-optimal ideal point, then counts how many
        samples are dominated by at least one solution in the front.
        """
        if not front:
            return 0.0

        dim = self.dim
        ref = self.reference_point

        # Compute ideal point (lower bound of dominated region)
        ideal = list(ref)
        for ind in front:
            for m in range(dim):
                if ind.objectives[m] < ideal[m]:
                    ideal[m] = ind.objectives[m]

        # Volume of the sampling region
        region_volume = 1.0
        for m in range(dim):
            width = ref[m] - ideal[m]
            if width <= 0:
                return 0.0
            region_volume *= width

        # Monte Carlo sampling
        dominated_count = 0
        for _ in range(n_samples):
            # Generate random point in [ideal, ref]
            point = [ideal[m] + random.random() * (ref[m] - ideal[m]) for m in range(dim)]

            # Check if point is dominated by any solution in the front
            for ind in front:
                dominated = True
                for m in range(dim):
                    if ind.objectives[m] > point[m]:
                        dominated = False
                        break
                if dominated:
                    dominated_count += 1
                    break

        return (dominated_count / n_samples) * region_volume


# ---------------------------------------------------------------------------
# Utility: Non-dominated sorting returning Individual objects
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(
    population: List[Individual], use_constraints: bool = True
) -> List[List[Individual]]:
    """
    Convenience wrapper that returns fronts as lists of Individual objects
    (rather than indices).
    """
    fronts_indices = non_dominated_sort(population, use_constraints=use_constraints)
    return [[population[i] for i in front] for front in fronts_indices]


# ---------------------------------------------------------------------------
# Utility: Crowding distance for a full population (all fronts)
# ---------------------------------------------------------------------------

def assign_crowding_distances(population: List[Individual]) -> None:
    """Assign crowding distances to all individuals, grouped by front."""
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        crowding_distance_assignment(front)


# ---------------------------------------------------------------------------
# Utility: Validate Pareto front (check for dominance within the front)
# ---------------------------------------------------------------------------

def validate_pareto_front(front: List[Individual]) -> Tuple[bool, int]:
    """
    Validate that no individual in the front dominates another.

    Returns (is_valid, num_violations).
    """
    violations = 0
    n = len(front)
    for i in range(n):
        for j in range(i + 1, n):
            if front[i].dominates(front[j]) or front[j].dominates(front[i]):
                violations += 1
    return violations == 0, violations


# ---------------------------------------------------------------------------
# Utility: Inverted generational distance (IGD)
# ---------------------------------------------------------------------------

def inverted_generational_distance(
    obtained: List[Individual],
    true_pareto: List[List[float]],
) -> float:
    """
    Compute the Inverted Generational Distance (IGD).

    IGD = (1/|P*|) * sum_{p* in P*} min_{p in Q} ||p - p*||

    Lower is better; 0 means the obtained front perfectly covers the true front.
    """
    if not true_pareto or not obtained:
        return float("inf")

    total = 0.0
    for true_point in true_pareto:
        min_dist = float("inf")
        for ind in obtained:
            d = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(ind.objectives, true_point))
            )
            if d < min_dist:
                min_dist = d
        total += min_dist

    return total / len(true_pareto)


# ---------------------------------------------------------------------------
# Utility: Spread (diversity metric)
# ---------------------------------------------------------------------------

def spread(population: List[Individual]) -> float:
    """
    Compute the Spread metric (also called Delta).

    Measures the relative extent of spread achieved among the non-dominated
    solutions. A value of 1 is ideal; lower values indicate poor distribution.

    Delta = (d_f + d_l + sum|d_i - d_bar|) / (d_f + d_l + (N-1)*d_bar)
    """
    fronts = fast_non_dominated_sort(population, use_constraints=False)
    if not fronts or not fronts[0]:
        return float("inf")

    front = fronts[0]
    n = len(front)
    M = len(front[0].objectives)

    if n < 2:
        return 0.0

    # For simplicity, use the first objective for distance computation
    front.sort(key=lambda ind: ind.objectives[0])

    distances = []
    for i in range(1, n):
        d = math.sqrt(
            sum(
                (front[i].objectives[m] - front[i - 1].objectives[m]) ** 2
                for m in range(M)
            )
        )
        distances.append(d)

    d_bar = sum(distances) / len(distances) if distances else 0.0

    # Extreme distances
    d_f = math.sqrt(
        sum(front[0].objectives[m] ** 2 for m in range(M))
    )
    d_l = math.sqrt(
        sum(front[-1].objectives[m] ** 2 for m in range(M))
    )

    if d_f + d_l + (n - 1) * d_bar < 1e-14:
        return 0.0

    delta = (d_f + d_l + sum(abs(d - d_bar) for d in distances)) / (
        d_f + d_l + (n - 1) * d_bar
    )
    return delta


# ---------------------------------------------------------------------------
# Utility: Generate initial population with Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def latin_hypercube_sampling(
    num_samples: int,
    bounds: List[Tuple[float, float]],
) -> List[List[float]]:
    """
    Generate samples using Latin Hypercube Sampling (LHS).

    Divides each dimension into num_samples equal intervals and
    places exactly one sample in each interval.
    """
    num_vars = len(bounds)
    samples: List[List[float]] = []

    for i in range(num_samples):
        sample: List[float] = []
        for j in range(num_vars):
            lo, hi = bounds[j]
            # Stratum for this variable
            stratum_width = (hi - lo) / num_samples
            # Random position within the stratum
            val = lo + (i + random.random()) * stratum_width
            sample.append(val)
        samples.append(sample)

    # Shuffle each column independently
    for j in range(num_vars):
        column = [samples[i][j] for i in range(num_samples)]
        random.shuffle(column)
        for i in range(num_samples):
            samples[i][j] = column[i]

    return samples


# ---------------------------------------------------------------------------
# Utility: Adaptive operator selection (simple fitness-rate-based)
# ---------------------------------------------------------------------------

class AdaptiveOperatorSelection:
    """
    Simple adaptive operator selection based on success rates.

    Tracks the success of each operator and adjusts selection probabilities.
    """

    def __init__(self, operator_names: List[str], initial_probs: Optional[List[float]] = None) -> None:
        self.names = operator_names
        n = len(operator_names)
        self.probs = initial_probs if initial_probs else [1.0 / n] * n
        self.successes = [0] * n
        self.uses = [0] * n
        self.learning_rate = 0.1

    def select_operator(self) -> int:
        """Select an operator index based on current probabilities."""
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(self.probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(self.probs) - 1

    def reward(self, operator_idx: int, success: bool) -> None:
        """Update probabilities based on operator success/failure."""
        self.uses[operator_idx] += 1
        if success:
            self.successes[operator_idx] += 1

        # Update using sliding window
        for i in range(len(self.probs)):
            if self.uses[i] > 0:
                rate = self.successes[i] / self.uses[i]
                self.probs[i] += self.learning_rate * (rate - self.probs[i])

        # Normalize
        total = sum(self.probs)
        if total > 0:
            self.probs = [p / total for p in self.probs]

    def get_probabilities(self) -> List[float]:
        return list(self.probs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Run a quick self-test with ZDT1."""
    print("NSGA-II self-test (ZDT1, 30 vars, 2 objs, 50 gen)...")

    def zdt1(genes: List[float]) -> Tuple[List[float], List[float]]:
        n = len(genes)
        f1 = genes[0]
        g = 1.0 + 9.0 / (n - 1) * sum(genes[1:])
        h = 1.0 - math.sqrt(f1 / g)
        f2 = g * h
        return [f1, f2], []

    config = NSGA2Config(
        population_size=100,
        num_generations=50,
        num_objectives=2,
        num_variables=30,
        crossover_type="sbx",
        mutation_type="polynomial",
        seed=42,
    )
    nsga2 = NSGA2(config, zdt1)
    front = nsga2.run()

    print(f"  Pareto front size: {len(front)}")
    if front:
        f1_vals = [ind.objectives[0] for ind in front]
        f2_vals = [ind.objectives[1] for ind in front]
        print(f"  f1 range: [{min(f1_vals):.6f}, {max(f1_vals):.6f}]")
        print(f"  f2 range: [{min(f2_vals):.6f}, {max(f2_vals):.6f}]")

        # Hypervolume
        hv = Hypervolume(reference_point=[1.1, 1.1])
        vol = hv.compute(front)
        print(f"  Hypervolume (2D): {vol:.6f}")

        # Validate
        valid, violations = validate_pareto_front(front)
        print(f"  Pareto valid: {valid}, violations: {violations}")

    # Test NSGA-III
    print("\nNSGA-III self-test (DTLZ1, 3 objs, 7 vars, 30 gen)...")
    def dtlz1(genes: List[float]) -> Tuple[List[float], List[float]]:
        M = 3
        k = len(genes) - M + 1
        g = 100.0 * (k + sum((x - 0.5) ** 2 - math.cos(20 * math.pi * (x - 0.5)) for x in genes[M - 1:]))
        f = []
        for i in range(M):
            prod = 1.0
            for j in range(M - 1 - i):
                prod *= genes[j]
            if i > 0:
                prod *= 1.0 - genes[M - 1 - i]
            f.append(0.5 * prod * (1.0 + g))
        return f, []

    config3 = NSGA2Config(
        population_size=92,
        num_generations=30,
        num_objectives=3,
        num_variables=7,
        crossover_type="sbx",
        mutation_type="polynomial",
        reference_point_divisions=5,
        seed=42,
    )
    nsga3 = NSGA3(config3, dtlz1)
    front3 = nsga3.run()
    print(f"  Pareto front size: {len(front3)}")

    # Test MOEA/D
    print("\nMOEA/D self-test (ZDT1, 30 vars, 2 objs, 50 gen)...")
    config_md = NSGA2Config(
        population_size=100,
        num_generations=50,
        num_objectives=2,
        num_variables=30,
        crossover_type="sbx",
        mutation_type="polynomial",
        moead_neighborhood_size=20,
        seed=42,
    )
    moead = MOEAD(config_md, zdt1)
    front_md = moead.run()
    print(f"  Pareto front size: {len(front_md)}")
    if front_md:
        f1_vals = [ind.objectives[0] for ind in front_md]
        f2_vals = [ind.objectives[1] for ind in front_md]
        print(f"  f1 range: [{min(f1_vals):.6f}, {max(f1_vals):.6f}]")
        print(f"  f2 range: [{min(f2_vals):.6f}, {max(f2_vals):.6f}]")

    # Test constrained problem (SRN)
    print("\nConstrained problem self-test (SRN)...")
    def srn(genes: List[float]) -> Tuple[List[float], List[float]]:
        x1, x2 = genes[0], genes[1]
        f1 = 2.0 + (x1 - 2.0) ** 2 + (x2 - 1.0) ** 2
        f2 = 9.0 * x1 - (x2 - 1.0) ** 2
        c1 = x1 ** 2 + x2 ** 2 - 225.0
        c2 = x1 - 3.0 * x2 + 10.0
        return [f1, f2], [c1, c2]

    config_c = NSGA2Config(
        population_size=100,
        num_generations=50,
        num_objectives=2,
        num_variables=2,
        variable_bounds=[(-20.0, 20.0), (-20.0, 20.0)],
        crossover_type="sbx",
        mutation_type="polynomial",
        seed=42,
    )
    nsga2_c = NSGA2(config_c, srn)
    front_c = nsga2_c.run()
    feasible = [ind for ind in front_c if ind.constraint_violation <= 0.0]
    print(f"  Pareto front size: {len(front_c)}, feasible: {len(feasible)}")

    print("\nAll self-tests completed.")


if __name__ == "__main__":
    _self_test()
