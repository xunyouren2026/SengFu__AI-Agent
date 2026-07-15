"""
Genetic Programming Core Implementation
=======================================
A complete, production-quality Genetic Programming framework in pure Python.
Implements standard GP, strongly-typed GP, and auto-constructive GP with
all standard operators, selection methods, and bloat control strategies.
"""

import math
import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable, Set


# =============================================================================
# 1. GPNode - Tree Node for Expression Trees
# =============================================================================

class GPNode:
    """A node in a genetic programming expression tree."""

    __slots__ = ['node_type', 'value', 'children', 'parent']

    def __init__(self, node_type: str, value: str, children: Optional[List['GPNode']] = None):
        if node_type not in ('function', 'terminal'):
            raise ValueError(f"node_type must be 'function' or 'terminal', got '{node_type}'")
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []
        self.parent = None
        for child in self.children:
            child.parent = self

    def depth(self) -> int:
        """Return the depth of the subtree rooted at this node."""
        if not self.children:
            return 0
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        """Return the number of nodes in the subtree rooted at this node."""
        return 1 + sum(child.size() for child in self.children)

    def evaluate(self, inputs: Optional[Dict[str, float]] = None,
                 func_impl: Optional[Dict[str, Callable]] = None) -> float:
        """Evaluate the expression tree, returning a numeric result."""
        if inputs is None:
            inputs = {}

        if self.node_type == 'terminal':
            if self.value in inputs:
                return float(inputs[self.value])
            if self.value == 'pi':
                return math.pi
            if self.value == 'e':
                return math.e
            try:
                return float(self.value)
            except ValueError:
                raise ValueError(f"Unknown terminal: '{self.value}'")

        if func_impl is not None and self.value in func_impl:
            args = [child.evaluate(inputs, func_impl) for child in self.children]
            return func_impl[self.value](*args)

        args = [child.evaluate(inputs, func_impl) for child in self.children]
        return self._builtin_eval(self.value, args)

    @staticmethod
    def _builtin_eval(name: str, args: List[float]) -> float:
        """Evaluate a built-in function with given arguments."""
        if name == 'add':
            return args[0] + args[1]
        elif name == 'sub':
            return args[0] - args[1]
        elif name == 'mul':
            return args[0] * args[1]
        elif name == 'div':
            return args[0] / args[1] if abs(args[1]) > 1e-10 else 1.0
        elif name == 'neg':
            return -args[0]
        elif name == 'abs':
            return abs(args[0])
        elif name == 'sin':
            return math.sin(args[0])
        elif name == 'cos':
            return math.cos(args[0])
        elif name == 'tan':
            val = math.tan(args[0])
            return max(-1e6, min(1e6, val))
        elif name == 'exp':
            val = math.exp(max(-700, min(700, args[0])))
            return max(-1e6, min(1e6, val))
        elif name == 'log':
            return math.log(abs(args[0])) if abs(args[0]) > 1e-10 else 0.0
        elif name == 'sqrt':
            return math.sqrt(abs(args[0]))
        elif name == 'lt':
            return 1.0 if args[0] < args[1] else 0.0
        elif name == 'gt':
            return 1.0 if args[0] > args[1] else 0.0
        elif name == 'eq':
            return 1.0 if abs(args[0] - args[1]) < 1e-10 else 0.0
        elif name == 'and_':
            return 1.0 if (args[0] > 0.5 and args[1] > 0.5) else 0.0
        elif name == 'or_':
            return 1.0 if (args[0] > 0.5 or args[1] > 0.5) else 0.0
        elif name == 'not_':
            return 1.0 if args[0] <= 0.5 else 0.0
        elif name == 'if_then_else':
            return args[1] if args[0] > 0.5 else args[2]
        else:
            raise ValueError(f"Unknown function: '{name}'")

    def copy(self) -> 'GPNode':
        """Return a deep copy of this node and its subtree."""
        new_children = [child.copy() for child in self.children]
        new_node = GPNode(self.node_type, self.value, new_children)
        return new_node

    def to_string(self) -> str:
        """Return a human-readable string representation of the subtree."""
        if self.node_type == 'terminal':
            return self.value
        child_strs = [child.to_string() for child in self.children]
        display_name = self.value
        if self.value == 'add':
            display_name = '+'
        elif self.value == 'sub':
            display_name = '-'
        elif self.value == 'mul':
            display_name = '*'
        elif self.value == 'div':
            display_name = '/'
        elif self.value == 'neg':
            display_name = '~'
        elif self.value == 'and_':
            display_name = 'AND'
        elif self.value == 'or_':
            display_name = 'OR'
        elif self.value == 'not_':
            display_name = 'NOT'
        elif self.value == 'if_then_else':
            return f"(IF {child_strs[0]} THEN {child_strs[1]} ELSE {child_strs[2]})"
        if len(child_strs) == 1:
            return f"({display_name} {child_strs[0]})"
        return f"({display_name} {' '.join(child_strs)})"

    def get_all_nodes(self) -> List['GPNode']:
        """Return a flat list of all nodes in the subtree (pre-order)."""
        result = [self]
        for child in self.children:
            result.extend(child.get_all_nodes())
        return result

    def get_nodes_at_depth(self, target_depth: int) -> List['GPNode']:
        """Return all nodes at a specific depth (root is depth 0)."""
        if target_depth == 0:
            return [self]
        result = []
        for child in self.children:
            result.extend(child.get_nodes_at_depth(target_depth - 1))
        return result

    def replace_child(self, old_child: 'GPNode', new_child: 'GPNode') -> bool:
        """Replace a child node with a new one. Returns True if found."""
        for i, child in enumerate(self.children):
            if child is old_child:
                self.children[i] = new_child
                new_child.parent = self
                return True
        return False


# =============================================================================
# 2. FunctionSet - Available Functions
# =============================================================================

@dataclass
class FunctionDefinition:
    """Definition of a single GP function."""
    name: str
    arity: int
    implementation: Callable
    return_type: str = 'float'

    def __call__(self, *args):
        return self.implementation(*args)


class FunctionSet:
    """Collection of available functions for GP expression trees."""

    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all default built-in functions."""
        # Arithmetic
        self.add('add', 2, lambda a, b: a + b)
        self.add('sub', 2, lambda a, b: a - b)
        self.add('mul', 2, lambda a, b: a * b)
        self.add('div', 2, lambda a, b: a / b if abs(b) > 1e-10 else 1.0)
        self.add('neg', 1, lambda a: -a)
        self.add('abs', 1, lambda a: abs(a))

        # Mathematical
        self.add('sin', 1, lambda a: math.sin(a))
        self.add('cos', 1, lambda a: math.cos(a))
        self.add('tan', 1, lambda a: max(-1e6, min(1e6, math.tan(a))))
        self.add('exp', 1, lambda a: max(-1e6, min(1e6, math.exp(max(-700, min(700, a))))))
        self.add('log', 1, lambda a: math.log(abs(a)) if abs(a) > 1e-10 else 0.0)
        self.add('sqrt', 1, lambda a: math.sqrt(abs(a)))

        # Comparison
        self.add('lt', 2, lambda a, b: 1.0 if a < b else 0.0)
        self.add('gt', 2, lambda a, b: 1.0 if a > b else 0.0)
        self.add('eq', 2, lambda a, b: 1.0 if abs(a - b) < 1e-10 else 0.0)
        self.add('and_', 2, lambda a, b: 1.0 if (a > 0.5 and b > 0.5) else 0.0)
        self.add('or_', 2, lambda a, b: 1.0 if (a > 0.5 or b > 0.5) else 0.0)
        self.add('not_', 1, lambda a: 1.0 if a <= 0.5 else 0.0)

        # Conditional
        self.add('if_then_else', 3,
                 lambda cond, t, f: t if cond > 0.5 else f)

    def add(self, name: str, arity: int, impl: Callable, return_type: str = 'float'):
        """Add a new function to the set."""
        self.functions[name] = FunctionDefinition(name, arity, impl, return_type)

    def get(self, name: str) -> Optional[FunctionDefinition]:
        """Get a function definition by name."""
        return self.functions.get(name)

    def get_by_arity(self, arity: int) -> List[FunctionDefinition]:
        """Return all functions with a given arity."""
        return [f for f in self.functions.values() if f.arity == arity]

    def get_random(self, arity: Optional[int] = None) -> FunctionDefinition:
        """Return a random function, optionally filtered by arity."""
        if arity is not None:
            candidates = self.get_by_arity(arity)
        else:
            candidates = list(self.functions.values())
        if not candidates:
            raise ValueError(f"No functions available with arity={arity}")
        return random.choice(candidates)

    def get_names(self) -> List[str]:
        """Return list of all function names."""
        return list(self.functions.keys())

    def get_implementation_map(self) -> Dict[str, Callable]:
        """Return a name-to-implementation mapping for evaluation."""
        return {name: func.implementation for name, func in self.functions.items()}

    def __len__(self) -> int:
        return len(self.functions)

    def __contains__(self, name: str) -> bool:
        return name in self.functions


# =============================================================================
# 3. TerminalSet - Available Terminals
# =============================================================================

class TerminalSet:
    """Collection of available terminals for GP expression trees."""

    def __init__(self, num_variables: int = 1,
                 constant_min: float = -10.0, constant_max: float = 10.0,
                 include_constants: bool = True):
        self.num_variables = num_variables
        self.constant_min = constant_min
        self.constant_max = constant_max
        self.include_constants = include_constants
        self.variables: List[str] = [f'x{i}' for i in range(1, num_variables + 1)]
        self.named_constants: Dict[str, float] = {}
        if include_constants:
            self.named_constants['pi'] = math.pi
            self.named_constants['e'] = math.e

    def get_variable(self, index: int) -> str:
        """Get variable name by index."""
        return self.variables[index]

    def generate_erc(self) -> str:
        """Generate an Ephemeral Random Constant as a string."""
        val = random.uniform(self.constant_min, self.constant_max)
        return f"{val:.6g}"

    def get_random_terminal(self) -> str:
        """Return a random terminal (variable, ERC, or named constant)."""
        choices = list(self.variables)
        if self.include_constants:
            choices.extend(self.named_constants.keys())
        choices.append('__erc__')
        choice = random.choice(choices)
        if choice == '__erc__':
            return self.generate_erc()
        return choice

    def get_variables_only(self) -> List[str]:
        """Return only variable names (no constants)."""
        return list(self.variables)

    def get_all_terminals(self) -> List[str]:
        """Return all possible terminal values including named constants."""
        terminals = list(self.variables)
        if self.include_constants:
            terminals.extend(self.named_constants.keys())
        return terminals

    def __len__(self) -> int:
        return len(self.variables) + len(self.named_constants) + 1  # +1 for ERC


def _get_random_subtree(tree) -> Tuple[GPNode, object, int]:
    """Select a random node and its parent from the tree."""
    all_nodes = tree.root.get_all_nodes()
    node = random.choice(all_nodes)
    parent = node.parent
    child_index = -1
    if parent is not None:
        for i, c in enumerate(parent.children):
            if c is node:
                child_index = i
                break
    return node, parent, child_index


def _replace_node(root: GPNode, old_node: GPNode, new_node: GPNode) -> GPNode:
    """Replace old_node with new_node in the tree. Returns (possibly new) root."""
    if root is old_node:
        return new_node
    for i, child in enumerate(root.children):
        if child is old_node:
            root.children[i] = new_node
            new_node.parent = root
            return root
        result = _replace_node(child, old_node, new_node)
        if result is not child:
            return root
    return root


# =============================================================================
# 4. GPTree - Full Expression Tree with Operations
# =============================================================================

class GPTree:
    """A genetic programming expression tree with generation and variation operators."""

    def __init__(self, root: Optional[GPNode] = None):
        self.root = root

    @property
    def depth(self) -> int:
        return self.root.depth() if self.root else 0

    @property
    def size(self) -> int:
        return self.root.size() if self.root else 0

    def evaluate(self, inputs: Optional[Dict[str, float]] = None,
                 func_impl: Optional[Dict[str, Callable]] = None) -> float:
        if self.root is None:
            return 0.0
        return self.root.evaluate(inputs, func_impl)

    def to_string(self) -> str:
        return self.root.to_string() if self.root else "EMPTY"

    def copy(self) -> 'GPTree':
        return GPTree(self.root.copy() if self.root else None)

    @staticmethod
    def _create_random_node(func_set: FunctionSet, term_set: TerminalSet,
                            depth: int, max_depth: int, method: str) -> GPNode:
        """Recursively create a random node using specified method."""
        if depth >= max_depth:
            return GPNode('terminal', term_set.get_random_terminal())

        if method == 'full':
            func = func_set.get_random()
            children = [
                GPTree._create_random_node(func_set, term_set, depth + 1, max_depth, method)
                for _ in range(func.arity)
            ]
            return GPNode('function', func.name, children)
        elif method == 'grow':
            if random.random() < 0.5 and depth < max_depth - 1:
                func = func_set.get_random()
                children = [
                    GPTree._create_random_node(func_set, term_set, depth + 1, max_depth, method)
                    for _ in range(func.arity)
                ]
                return GPNode('function', func.name, children)
            else:
                return GPNode('terminal', term_set.get_random_terminal())
        else:
            raise ValueError(f"Unknown generation method: '{method}'")

    @classmethod
    def generate_full(cls, func_set: FunctionSet, term_set: TerminalSet,
                      max_depth: int) -> 'GPTree':
        """Generate a full tree where every branch reaches max_depth."""
        root = cls._create_random_node(func_set, term_set, 0, max_depth, 'full')
        return cls(root)

    @classmethod
    def generate_grow(cls, func_set: FunctionSet, term_set: TerminalSet,
                      max_depth: int) -> 'GPTree':
        """Generate a tree with variable shape using grow method."""
        root = cls._create_random_node(func_set, term_set, 0, max_depth, 'grow')
        return cls(root)

    @classmethod
    def generate_ramped_half_and_half(cls, func_set: FunctionSet, term_set: TerminalSet,
                                      pop_size: int, min_depth: int = 2,
                                      max_depth: int = 6) -> List['GPTree']:
        """Generate initial population using ramped half-and-half."""
        population = []
        depth_range = list(range(min_depth, max_depth + 1))
        for i in range(pop_size):
            depth = depth_range[i % len(depth_range)]
            method = 'full' if (i // len(depth_range)) % 2 == 0 else 'grow'
            if method == 'full':
                tree = cls.generate_full(func_set, term_set, depth)
            else:
                tree = cls.generate_grow(func_set, term_set, depth)
            population.append(tree)
        random.shuffle(population)
        return population

    @classmethod
    def subtree_crossover(cls, parent1: 'GPTree', parent2: 'GPTree',
                          max_depth: int = 17) -> Tuple['GPTree', 'GPTree']:
        """Perform standard subtree crossover between two parent trees."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        node1, parent1_ref, idx1 = _get_random_subtree(child1)
        node2, parent2_ref, idx2 = _get_random_subtree(child2)

        subtree1_copy = node1.copy()
        subtree2_copy = node2.copy()

        if parent1_ref is not None:
            parent1_ref.children[idx1] = subtree2_copy
            subtree2_copy.parent = parent1_ref
        else:
            child1.root = subtree2_copy

        if parent2_ref is not None:
            parent2_ref.children[idx2] = subtree1_copy
            subtree1_copy.parent = parent2_ref
        else:
            child2.root = subtree1_copy

        if child1.depth > max_depth:
            child1 = parent1.copy()
        if child2.depth > max_depth:
            child2 = parent2.copy()

        return child1, child2

    @classmethod
    def subtree_mutation(cls, tree: 'GPTree', func_set: FunctionSet,
                         term_set: TerminalSet, max_depth: int = 17) -> 'GPTree':
        """Replace a random subtree with a newly generated one."""
        mutant = tree.copy()
        new_subtree = cls.generate_grow(func_set, term_set,
                                        random.randint(1, max(2, max_depth // 2)))
        node, parent, idx = _get_random_subtree(mutant)
        if parent is not None:
            parent.children[idx] = new_subtree.root
            new_subtree.root.parent = parent
        else:
            mutant.root = new_subtree.root
        if mutant.depth > max_depth:
            return tree.copy()
        return mutant

    @classmethod
    def point_mutation(cls, tree: 'GPTree', func_set: FunctionSet,
                       term_set: TerminalSet) -> 'GPTree':
        """Mutate a single node in the tree."""
        mutant = tree.copy()
        all_nodes = mutant.root.get_all_nodes()
        node = random.choice(all_nodes)
        if node.node_type == 'function':
            new_func = func_set.get_random(arity=len(node.children))
            node.value = new_func.name
        else:
            node.value = term_set.get_random_terminal()
        return mutant

    @classmethod
    def hoist_mutation(cls, tree: 'GPTree') -> 'GPTree':
        """Replace the tree with a randomly selected subtree (reduces size)."""
        mutant = tree.copy()
        all_nodes = mutant.root.get_all_nodes()
        function_nodes = [n for n in all_nodes if n.node_type == 'function']
        if function_nodes:
            chosen = random.choice(function_nodes)
            return GPTree(chosen.copy())
        return mutant

    @classmethod
    def permutation_mutation(cls, tree: 'GPTree') -> 'GPTree':
        """Swap arguments of a function node (same-arity permutation)."""
        mutant = tree.copy()
        all_nodes = mutant.root.get_all_nodes()
        function_nodes = [n for n in all_nodes if n.node_type == 'function' and len(n.children) >= 2]
        if function_nodes:
            node = random.choice(function_nodes)
            i, j = random.sample(range(len(node.children)), 2)
            node.children[i], node.children[j] = node.children[j], node.children[i]
        return mutant

    @classmethod
    def constant_mutation(cls, tree: 'GPTree', sigma: float = 0.1) -> 'GPTree':
        """Slightly perturb terminal constant values."""
        mutant = tree.copy()
        all_nodes = mutant.root.get_all_nodes()
        terminals = [n for n in all_nodes if n.node_type == 'terminal']
        for node in terminals:
            try:
                val = float(node.value)
                perturbation = random.gauss(0, sigma * abs(val) + 1e-8)
                node.value = f"{val + perturbation:.6g}"
            except ValueError:
                continue
        return mutant


# =============================================================================
# 5. GPConfig - Configuration Dataclass
# =============================================================================

@dataclass
class GPConfig:
    """Configuration parameters for the GP algorithm."""
    population_size: int = 500
    max_generations: int = 50
    max_depth: int = 17
    max_init_depth: int = 6
    max_tree_size: int = 512
    min_init_depth: int = 2
    crossover_rate: float = 0.85
    mutation_rate: float = 0.10
    reproduction_rate: float = 0.05
    subtree_mutation_rate: float = 0.5
    point_mutation_rate: float = 0.25
    hoist_mutation_rate: float = 0.1
    permutation_mutation_rate: float = 0.1
    constant_mutation_rate: float = 0.05
    tournament_size: int = 7
    elitism_count: int = 1
    parsimony_coefficient: float = 0.001
    num_variables: int = 1
    constant_min: float = -10.0
    constant_max: float = 10.0
    random_seed: Optional[int] = None
    num_islands: int = 1
    migration_interval: int = 10
    migration_fraction: float = 0.1
    fitness_threshold: float = 1e-6
    max_evaluations: Optional[int] = None
    target: str = 'regression'  # 'regression' or 'classification'


# =============================================================================
# 6. GPFitness - Fitness Evaluation
# =============================================================================

class GPFitness:
    """Fitness evaluation metrics for GP individuals."""

    @staticmethod
    def _safe_divide(a: float, b: float) -> float:
        return a / b if abs(b) > 1e-10 else 0.0

    @staticmethod
    def rmse(predicted: List[float], actual: List[float]) -> float:
        """Root Mean Square Error."""
        n = len(predicted)
        if n == 0:
            return float('inf')
        total = sum((p - a) ** 2 for p, a in zip(predicted, actual))
        return math.sqrt(total / n)

    @staticmethod
    def mae(predicted: List[float], actual: List[float]) -> float:
        """Mean Absolute Error."""
        n = len(predicted)
        if n == 0:
            return float('inf')
        return sum(abs(p - a) for p, a in zip(predicted, actual)) / n

    @staticmethod
    def r_squared(predicted: List[float], actual: List[float]) -> float:
        """Coefficient of determination R^2."""
        n = len(predicted)
        if n < 2:
            return 0.0
        mean_actual = sum(actual) / n
        ss_tot = sum((a - mean_actual) ** 2 for a in actual)
        if ss_tot < 1e-10:
            return 1.0
        ss_res = sum((p - a) ** 2 for p, a in zip(predicted, actual))
        return 1.0 - ss_res / ss_tot

    @staticmethod
    def accuracy(predicted: List[float], actual: List[float],
                 threshold: float = 0.5) -> float:
        """Classification accuracy."""
        if not predicted:
            return 0.0
        correct = sum(
            1 for p, a in zip(predicted, actual)
            if (p >= threshold) == (a >= threshold)
        )
        return correct / len(predicted)

    @staticmethod
    def f1_score(predicted: List[float], actual: List[float],
                 threshold: float = 0.5) -> float:
        """F1 score for binary classification."""
        tp = sum(1 for p, a in zip(predicted, actual) if p >= threshold and a >= threshold)
        fp = sum(1 for p, a in zip(predicted, actual) if p >= threshold and a < threshold)
        fn = sum(1 for p, a in zip(predicted, actual) if p < threshold and a >= threshold)
        precision = GPFitness._safe_divide(tp, tp + fp)
        recall = GPFitness._safe_divide(tp, tp + fn)
        if precision + recall < 1e-10:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def multi_objective(predicted: List[float], actual: List[float],
                        complexity: int, parsimony_coeff: float = 0.001,
                        metric: str = 'rmse') -> float:
        """Multi-objective fitness combining error and complexity penalty."""
        if metric == 'rmse':
            error = GPFitness.rmse(predicted, actual)
        elif metric == 'mae':
            error = GPFitness.mae(predicted, actual)
        else:
            error = GPFitness.rmse(predicted, actual)
        return error + parsimony_coeff * complexity

    @staticmethod
    def evaluate_individual(tree: GPTree, X: List[Dict[str, float]],
                            y: List[float], config: GPConfig,
                            func_impl: Optional[Dict[str, Callable]] = None) -> Tuple[float, float]:
        """Evaluate a single individual. Returns (fitness, raw_error)."""
        predicted = []
        for inputs in X:
            try:
                val = tree.evaluate(inputs, func_impl)
                if math.isnan(val) or math.isinf(val):
                    val = 1e6
                val = max(-1e6, min(1e6, val))
                predicted.append(val)
            except Exception:
                predicted.append(1e6)

        if config.target == 'classification':
            raw_error = 1.0 - GPFitness.accuracy(predicted, y)
        else:
            raw_error = GPFitness.rmse(predicted, y)

        complexity = tree.size
        fitness = GPFitness.multi_objective(
            predicted, y, complexity, config.parsimony_coefficient,
            metric='rmse' if config.target == 'regression' else 'accuracy'
        )
        return fitness, raw_error


# =============================================================================
# 7. GPIndividual - Individual Solution
# =============================================================================

class GPIndividual:
    """A single individual in the GP population."""

    def __init__(self, tree: GPTree, fitness: float = float('inf'),
                 raw_fitness: float = float('inf'), rank: int = -1):
        self.tree = tree
        self.fitness = fitness
        self.raw_fitness = raw_fitness
        self.rank = rank
        self.age = 0
        self.is_elite = False

    @property
    def complexity(self) -> int:
        return self.tree.size

    @property
    def depth(self) -> int:
        return self.tree.depth

    def copy(self) -> 'GPIndividual':
        ind = GPIndividual(self.tree.copy(), self.fitness, self.raw_fitness, self.rank)
        ind.age = self.age
        ind.is_elite = self.is_elite
        return ind

    def __lt__(self, other: 'GPIndividual') -> bool:
        return self.fitness < other.fitness

    def __repr__(self) -> str:
        return (f"GPIndividual(fitness={self.fitness:.6f}, "
                f"size={self.complexity}, depth={self.depth})")


# =============================================================================
# 8. Selection Operators
# =============================================================================

class TournamentSelection:
    """Tournament selection operator."""

    def __init__(self, tournament_size: int = 7):
        self.tournament_size = tournament_size

    def select(self, population: List[GPIndividual]) -> GPIndividual:
        """Select one individual via tournament."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament, key=lambda ind: ind.fitness)

    def select_multiple(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        """Select multiple individuals."""
        return [self.select(population) for _ in range(count)]


class RouletteWheelSelection:
    """Fitness-proportionate (roulette wheel) selection."""

    @staticmethod
    def _compute_weights(population: List[GPIndividual]) -> List[float]:
        min_fit = min(ind.fitness for ind in population)
        shifted = [ind.fitness - min_fit + 1e-10 for ind in population]
        max_shifted = max(shifted)
        if max_shifted > 1e-10:
            weights = [max_shifted - s + 1e-10 for s in shifted]
        else:
            weights = [1.0] * len(population)
        total = sum(weights)
        return [w / total for w in weights]

    def select(self, population: List[GPIndividual]) -> GPIndividual:
        """Select one individual via roulette wheel."""
        weights = self._compute_weights(population)
        r = random.random()
        cumulative = 0.0
        for ind, w in zip(population, weights):
            cumulative += w
            if r <= cumulative:
                return ind
        return population[-1]

    def select_multiple(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        return [self.select(population) for _ in range(count)]


class LexicaseSelection:
    """Lexicase selection: case-by-case filtering with random ordering."""

    def __init__(self, epsilon: float = 0.0):
        self.epsilon = epsilon
        self._case_errors: Optional[List[List[float]]] = None

    def set_case_errors(self, case_errors: List[List[float]]):
        """Set per-case errors for each individual. case_errors[i][j] = error of individual i on case j."""
        self._case_errors = case_errors

    def select(self, population: List[GPIndividual]) -> GPIndividual:
        """Select one individual via lexicase selection."""
        if self._case_errors is None or len(population) == 1:
            return random.choice(population)

        n_cases = len(self._case_errors[0])
        case_order = list(range(n_cases))
        random.shuffle(case_order)

        candidates = list(range(len(population)))

        for case_idx in case_order:
            if len(candidates) <= 1:
                break
            errors = [self._case_errors[i][case_idx] for i in candidates]
            min_error = min(errors)
            threshold = min_error + self.epsilon
            candidates = [c for c, e in zip(candidates, errors) if e <= threshold]

        return population[candidates[0]]

    def select_multiple(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        return [self.select(population) for _ in range(count)]


class DoubleTournament:
    """Double tournament selection: first on fitness, then on parsimony."""

    def __init__(self, fitness_tournament_size: int = 7,
                 parsimony_tournament_size: int = 2,
                 fitness_first: bool = True):
        self.fitness_tournament_size = fitness_tournament_size
        self.parsimony_tournament_size = parsimony_tournament_size
        self.fitness_first = fitness_first

    def select(self, population: List[GPIndividual]) -> GPIndividual:
        """Select using double tournament."""
        pool = random.sample(population, min(self.fitness_tournament_size, len(population)))

        if self.fitness_first:
            pool.sort(key=lambda ind: ind.fitness)
            pool = pool[:self.parsimony_tournament_size]
            return min(pool, key=lambda ind: ind.complexity)
        else:
            pool.sort(key=lambda ind: ind.complexity)
            pool = pool[:self.parsimony_tournament_size]
            return min(pool, key=lambda ind: ind.fitness)

    def select_multiple(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        return [self.select(population) for _ in range(count)]


# =============================================================================
# 9. BloatControl - Tree Size Control
# =============================================================================

class BloatControl:
    """Strategies for controlling bloat in GP trees."""

    def __init__(self, max_depth: int = 17, max_size: int = 512,
                 parsimony_coefficient: float = 0.001,
                 tarpeian_rate: float = 0.0, tarpeian_threshold: float = 0.8):
        self.max_depth = max_depth
        self.max_size = max_size
        self.parsimony_coefficient = parsimony_coefficient
        self.tarpeian_rate = tarpeian_rate
        self.tarpeian_threshold = tarpeian_threshold

    def check_depth_limit(self, tree: GPTree) -> bool:
        """Check if tree exceeds depth limit."""
        return tree.depth <= self.max_depth

    def check_size_limit(self, tree: GPTree) -> bool:
        """Check if tree exceeds size limit."""
        return tree.size <= self.max_size

    def apply_parsimony_pressure(self, individual: GPIndividual) -> float:
        """Apply parsimony coefficient to fitness."""
        return individual.fitness + self.parsimony_coefficient * individual.complexity

    def apply_tarpeian(self, population: List[GPIndividual],
                       worst_fitness: float) -> List[GPIndividual]:
        """Apply Tarpeian bloat control: randomly penalize large individuals."""
        if self.tarpeian_rate <= 0:
            return population
        median_size = sorted([ind.complexity for ind in population])[len(population) // 2]
        for ind in population:
            if ind.complexity > median_size * self.tarpeian_threshold:
                if random.random() < self.tarpeian_rate:
                    ind.fitness = worst_fitness
        return population

    def operator_equalization(self, population: List[GPIndividual],
                              max_depth: int) -> List[GPIndividual]:
        """Penalize individuals that are too deep relative to the limit."""
        for ind in population:
            depth_ratio = ind.depth / max_depth if max_depth > 0 else 1.0
            if depth_ratio > 0.8:
                penalty = (depth_ratio - 0.8) * 10.0
                ind.fitness += penalty
        return population

    def apply(self, population: List[GPIndividual],
              worst_fitness: float) -> List[GPIndividual]:
        """Apply all bloat control strategies."""
        population = self.apply_tarpeian(population, worst_fitness)
        population = self.operator_equalization(population, self.max_depth)
        return population


# =============================================================================
# 10. GeneticProgramming - Main GP Algorithm
# =============================================================================

class GeneticProgramming:
    """Main Genetic Programming algorithm with full evolutionary loop."""

    def __init__(self, config: Optional[GPConfig] = None):
        self.config = config if config is not None else GPConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

        self.func_set = FunctionSet()
        self.term_set = TerminalSet(
            num_variables=self.config.num_variables,
            constant_min=self.config.constant_min,
            constant_max=self.config.constant_max
        )
        self.bloat_control = BloatControl(
            max_depth=self.config.max_depth,
            max_size=self.config.max_tree_size,
            parsimony_coefficient=self.config.parsimony_coefficient
        )
        self.selector = TournamentSelection(self.config.tournament_size)
        self.population: List[GPIndividual] = []
        self.best_individual: Optional[GPIndividual] = None
        self.generation = 0
        self.evaluations = 0
        self.history: List[Dict[str, float]] = []
        self.X: List[Dict[str, float]] = []
        self.y: List[float] = []
        self.func_impl = self.func_set.get_implementation_map()
        self._islands: List[List[GPIndividual]] = []

    def initialize_population(self) -> List[GPIndividual]:
        """Create initial population using ramped half-and-half."""
        trees = GPTree.generate_ramped_half_and_half(
            self.func_set, self.term_set,
            self.config.population_size,
            self.config.min_init_depth,
            self.config.max_init_depth
        )
        self.population = [GPIndividual(tree) for tree in trees]
        return self.population

    def evaluate_population(self) -> List[GPIndividual]:
        """Evaluate all individuals in the population."""
        for ind in self.population:
            if self.config.max_evaluations is not None and self.evaluations >= self.config.max_evaluations:
                break
            fitness, raw_fitness = GPFitness.evaluate_individual(
                ind.tree, self.X, self.y, self.config, self.func_impl
            )
            ind.fitness = fitness
            ind.raw_fitness = raw_fitness
            self.evaluations += 1
        self.population.sort(key=lambda ind: ind.fitness)
        if self.best_individual is None or self.population[0].fitness < self.best_individual.fitness:
            self.best_individual = self.population[0].copy()
        return self.population

    def selection(self) -> List[GPIndividual]:
        """Select parents for the next generation."""
        return self.selector.select_multiple(self.population, self.config.population_size)

    def crossover(self, parent1: GPIndividual, parent2: GPIndividual) -> Tuple[GPIndividual, GPIndividual]:
        """Perform crossover between two parents."""
        child1_tree, child2_tree = GPTree.subtree_crossover(
            parent1.tree, parent2.tree, self.config.max_depth
        )
        child1 = GPIndividual(child1_tree)
        child2 = GPIndividual(child2_tree)
        return child1, child2

    def mutation(self, parent: GPIndividual) -> GPIndividual:
        """Apply a randomly chosen mutation operator."""
        r = random.random()
        if r < self.config.subtree_mutation_rate:
            tree = GPTree.subtree_mutation(
                parent.tree, self.func_set, self.term_set, self.config.max_depth
            )
        elif r < self.config.subtree_mutation_rate + self.config.point_mutation_rate:
            tree = GPTree.point_mutation(parent.tree, self.func_set, self.term_set)
        elif r < self.config.subtree_mutation_rate + self.config.point_mutation_rate + self.config.hoist_mutation_rate:
            tree = GPTree.hoist_mutation(parent.tree)
        elif r < (self.config.subtree_mutation_rate + self.config.point_mutation_rate +
                  self.config.hoist_mutation_rate + self.config.permutation_mutation_rate):
            tree = GPTree.permutation_mutation(parent.tree)
        else:
            tree = GPTree.constant_mutation(parent.tree)
        return GPIndividual(tree)

    def reproduction(self) -> GPIndividual:
        """Return a copy of a selected individual (no variation)."""
        return self.selector.select(self.population).copy()

    def _create_next_generation(self) -> List[GPIndividual]:
        """Build the next generation from the current population."""
        next_gen = []
        # Elitism: carry over best individuals
        for i in range(min(self.config.elitism_count, len(self.population))):
            elite = self.population[i].copy()
            elite.is_elite = True
            next_gen.append(elite)

        # Fill the rest with crossover, mutation, reproduction
        while len(next_gen) < self.config.population_size:
            r = random.random()
            if r < self.config.crossover_rate and len(next_gen) + 1 < self.config.population_size:
                parents = self.selector.select_multiple(self.population, 2)
                child1, child2 = self.crossover(parents[0], parents[1])
                next_gen.append(child1)
                if len(next_gen) < self.config.population_size:
                    next_gen.append(child2)
            elif r < self.config.crossover_rate + self.config.mutation_rate:
                parent = self.selector.select(self.population)
                child = self.mutation(parent)
                next_gen.append(child)
            else:
                child = self.reproduction()
                next_gen.append(child)

        # Age individuals
        for ind in next_gen:
            ind.age += 1

        return next_gen[:self.config.population_size]

    def _record_stats(self):
        """Record generation statistics."""
        fitnesses = [ind.fitness for ind in self.population]
        raw_errors = [ind.raw_fitness for ind in self.population]
        sizes = [ind.complexity for ind in self.population]
        self.history.append({
            'generation': self.generation,
            'best_fitness': min(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': max(fitnesses),
            'best_raw_error': min(raw_errors),
            'avg_size': sum(sizes) / len(sizes),
            'max_size': max(sizes),
            'avg_depth': sum(ind.depth for ind in self.population) / len(self.population),
        })

    def _migrate_between_islands(self):
        """Perform migration between sub-populations (islands)."""
        if self.config.num_islands <= 1:
            return
        island_size = len(self._islands[0])
        num_migrants = max(1, int(island_size * self.config.migration_fraction))
        for i in range(len(self._islands)):
            source_idx = (i + 1) % len(self._islands)
            source = self._islands[source_idx]
            source.sort(key=lambda ind: ind.fitness)
            migrants = [ind.copy() for ind in source[:num_migrants]]
            target = self._islands[i]
            target.sort(key=lambda ind: ind.fitness, reverse=True)
            for j, migrant in enumerate(migrants):
                target[-(j + 1)] = migrant

    def generational_loop(self) -> GPIndividual:
        """Run the main generational evolution loop."""
        for gen in range(self.config.max_generations):
            self.generation = gen
            self._record_stats()

            if self.best_individual.fitness <= self.config.fitness_threshold:
                break
            if self.config.max_evaluations is not None and self.evaluations >= self.config.max_evaluations:
                break

            next_gen = self._create_next_generation()
            self.population = next_gen
            self.evaluate_population()

            worst = max(ind.fitness for ind in self.population)
            self.population = self.bloat_control.apply(self.population, worst)

            if self.config.num_islands > 1 and gen > 0 and gen % self.config.migration_interval == 0:
                self._migrate_between_islands()

        self._record_stats()
        return self.best_individual

    def run(self, X: List[Dict[str, float]], y: List[float]) -> GPIndividual:
        """Run the full GP evolution. X is a list of input dicts, y is target values."""
        self.X = X
        self.y = y
        self.generation = 0
        self.evaluations = 0
        self.best_individual = None
        self.history = []

        if self.config.num_islands > 1:
            self._run_multi_population()
        else:
            self.initialize_population()
            self.evaluate_population()
            self.generational_loop()

        return self.best_individual

    def _run_multi_population(self):
        """Run multi-population (island model) GP."""
        island_size = self.config.population_size // self.config.num_islands
        self._islands = []
        for _ in range(self.config.num_islands):
            trees = GPTree.generate_ramped_half_and_half(
                self.func_set, self.term_set, island_size,
                self.config.min_init_depth, self.config.max_init_depth
            )
            island = [GPIndividual(tree) for tree in trees]
            self._islands.append(island)

        # Evaluate all islands
        self.population = []
        for island in self._islands:
            self.population = island
            self.evaluate_population()
            island[:] = self.population

        # Merge for global best tracking
        self.population = [ind for island in self._islands for ind in island]
        self.population.sort(key=lambda ind: ind.fitness)
        self.best_individual = self.population[0].copy()

        for gen in range(self.config.max_generations):
            self.generation = gen
            self._record_stats()

            if self.best_individual.fitness <= self.config.fitness_threshold:
                break

            for island in self._islands:
                self.population = island
                next_gen = self._create_next_generation()
                self.population = next_gen
                self.evaluate_population()
                worst = max(ind.fitness for ind in self.population)
                self.population = self.bloat_control.apply(self.population, worst)
                island[:] = self.population

            if gen > 0 and gen % self.config.migration_interval == 0:
                self._migrate_between_islands()

            # Update global best
            all_individuals = [ind for island in self._islands for ind in island]
            all_individuals.sort(key=lambda ind: ind.fitness)
            if all_individuals[0].fitness < self.best_individual.fitness:
                self.best_individual = all_individuals[0].copy()

        self.population = [ind for island in self._islands for ind in island]
        self._record_stats()

    def get_best_solution(self) -> Optional[GPIndividual]:
        """Return the best solution found."""
        return self.best_individual

    def predict(self, X: List[Dict[str, float]]) -> List[float]:
        """Predict using the best found tree."""
        if self.best_individual is None:
            raise RuntimeError("No solution available. Run the GP first.")
        predictions = []
        for inputs in X:
            try:
                val = self.best_individual.tree.evaluate(inputs, self.func_impl)
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
                predictions.append(max(-1e6, min(1e6, val)))
            except Exception:
                predictions.append(0.0)
        return predictions

    def get_stats(self) -> List[Dict[str, float]]:
        """Return the evolution history statistics."""
        return self.history


# =============================================================================
# 11. StronglyTypedGP - Strongly Typed Genetic Programming
# =============================================================================

class STGPType:
    """Represents a type in the STGP type system."""

    def __init__(self, name: str, supertype: Optional['STGPType'] = None):
        self.name = name
        self.supertype = supertype
        self._compatible_cache: Dict[str, bool] = {}

    def is_compatible(self, other: 'STGPType') -> bool:
        """Check if this type is compatible with another (subtype check)."""
        key = other.name
        if key in self._compatible_cache:
            return self._compatible_cache[key]
        current = self
        while current is not None:
            if current.name == other.name:
                self._compatible_cache[key] = True
                return True
            current = current.supertype
        self._compatible_cache[key] = False
        return False

    def __repr__(self) -> str:
        return f"STGPType({self.name})"

    def __eq__(self, other) -> bool:
        return isinstance(other, STGPType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class STGPNode(GPNode):
    """A typed GP node for Strongly Typed Genetic Programming."""

    __slots__ = ['node_type', 'value', 'children', 'parent', 'return_type', 'arg_types']

    def __init__(self, node_type: str, value: str,
                 return_type: STGPType,
                 arg_types: Optional[List[STGPType]] = None,
                 children: Optional[List['STGPNode']] = None):
        super().__init__(node_type, value, children)
        self.return_type = return_type
        self.arg_types = arg_types if arg_types is not None else []

    def copy(self) -> 'STGPNode':
        new_children = [child.copy() for child in self.children]
        new_node = STGPNode(self.node_type, self.value, self.return_type,
                            self.arg_types, new_children)
        return new_node


class STGPFunctionDef:
    """A typed function definition for STGP."""

    def __init__(self, name: str, arg_types: List[STGPType], return_type: STGPType,
                 implementation: Callable):
        self.name = name
        self.arg_types = arg_types
        self.return_type = return_type
        self.arity = len(arg_types)
        self.implementation = implementation


class StronglyTypedGP(GeneticProgramming):
    """Strongly Typed Genetic Programming with type constraints."""

    def __init__(self, config: Optional[GPConfig] = None):
        super().__init__(config)
        self.types: Dict[str, STGPType] = {}
        self.typed_functions: Dict[str, STGPFunctionDef] = {}
        self._init_type_system()

    def _init_type_system(self):
        """Initialize the default type hierarchy."""
        self.types['float'] = STGPType('float')
        self.types['bool'] = STGPType('bool')
        self.types['int'] = STGPType('int', self.types['float'])

        # Arithmetic functions: float -> float
        self._add_typed_func('add', ['float', 'float'], 'float',
                             lambda a, b: a + b)
        self._add_typed_func('sub', ['float', 'float'], 'float',
                             lambda a, b: a - b)
        self._add_typed_func('mul', ['float', 'float'], 'float',
                             lambda a, b: a * b)
        self._add_typed_func('div', ['float', 'float'], 'float',
                             lambda a, b: a / b if abs(b) > 1e-10 else 1.0)
        self._add_typed_func('neg', ['float'], 'float', lambda a: -a)
        self._add_typed_func('abs', ['float'], 'float', lambda a: abs(a))
        self._add_typed_func('sin', ['float'], 'float', lambda a: math.sin(a))
        self._add_typed_func('cos', ['float'], 'float', lambda a: math.cos(a))
        self._add_typed_func('sqrt', ['float'], 'float',
                             lambda a: math.sqrt(abs(a)))

        # Comparison: float, float -> bool
        self._add_typed_func('lt', ['float', 'float'], 'bool',
                             lambda a, b: 1.0 if a < b else 0.0)
        self._add_typed_func('gt', ['float', 'float'], 'bool',
                             lambda a, b: 1.0 if a > b else 0.0)

        # Boolean functions: bool, bool -> bool
        self._add_typed_func('and_', ['bool', 'bool'], 'bool',
                             lambda a, b: 1.0 if (a > 0.5 and b > 0.5) else 0.0)
        self._add_typed_func('or_', ['bool', 'bool'], 'bool',
                             lambda a, b: 1.0 if (a > 0.5 or b > 0.5) else 0.0)
        self._add_typed_func('not_', ['bool'], 'bool',
                             lambda a: 1.0 if a <= 0.5 else 0.0)

        # Conditional: bool, float, float -> float
        self._add_typed_func('if_then_else', ['bool', 'float', 'float'], 'float',
                             lambda c, t, f: t if c > 0.5 else f)

    def _add_typed_func(self, name: str, arg_type_names: List[str],
                        return_type_name: str, impl: Callable):
        """Add a typed function to the system."""
        arg_types = [self.types[n] for n in arg_type_names]
        return_type = self.types[return_type_name]
        self.typed_functions[name] = STGPFunctionDef(name, arg_types, return_type, impl)

    def _get_compatible_functions(self, return_type: STGPType) -> List[STGPFunctionDef]:
        """Get all functions that return a compatible type."""
        return [f for f in self.typed_functions.values()
                if return_type.is_compatible(f.return_type)]

    def _get_compatible_terminals(self, return_type: STGPType) -> List[str]:
        """Get terminals compatible with the given type."""
        if return_type.name in ('float', 'int'):
            return self.term_set.get_all_terminals()
        elif return_type.name == 'bool':
            return ['0.0', '1.0'] + self.term_set.get_variables_only()
        return self.term_set.get_all_terminals()

    def _generate_typed_node(self, return_type: STGPType, depth: int,
                             max_depth: int, method: str) -> STGPNode:
        """Generate a typed node respecting type constraints."""
        if depth >= max_depth:
            terminals = self._get_compatible_terminals(return_type)
            return STGPNode('terminal', random.choice(terminals), return_type)

        compatible_funcs = self._get_compatible_functions(return_type)
        if not compatible_funcs or (method == 'grow' and random.random() < 0.4):
            terminals = self._get_compatible_terminals(return_type)
            return STGPNode('terminal', random.choice(terminals), return_type)

        func = random.choice(compatible_funcs)
        children = []
        for arg_type in func.arg_types:
            child = self._generate_typed_node(arg_type, depth + 1, max_depth, method)
            children.append(child)
        return STGPNode('function', func.name, func.return_type,
                        func.arg_types, children)

    def initialize_population(self) -> List[GPIndividual]:
        """Create type-correct initial population."""
        float_type = self.types['float']
        population = []
        depth_range = list(range(self.config.min_init_depth, self.config.max_init_depth + 1))
        for i in range(self.config.population_size):
            depth = depth_range[i % len(depth_range)]
            method = 'full' if (i // len(depth_range)) % 2 == 0 else 'grow'
            root = self._generate_typed_node(float_type, 0, depth, method)
            tree = GPTree.__new__(GPTree)
            tree.root = root
            population.append(GPIndividual(tree))
        random.shuffle(population)
        self.population = population
        return population

    def _typed_crossover(self, parent1: GPIndividual, parent2: GPIndividual) -> Tuple[GPIndividual, GPIndividual]:
        """Perform type-compatible crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if not isinstance(child1.tree.root, STGPNode) or not isinstance(child2.tree.root, STGPNode):
            return GPTree.subtree_crossover(parent1.tree, parent2.tree, self.config.max_depth)

        nodes1 = child1.tree.root.get_all_nodes()
        nodes2 = child2.tree.root.get_all_nodes()

        # Find nodes with compatible return types
        random.shuffle(nodes1)
        random.shuffle(nodes2)
        for n1 in nodes1:
            for n2 in nodes2:
                if isinstance(n1, STGPNode) and isinstance(n2, STGPNode):
                    if n1.return_type.is_compatible(n2.return_type):
                        copy1 = n1.copy()
                        copy2 = n2.copy()
                        _replace_node(child1.tree.root, n1, copy2)
                        _replace_node(child2.tree.root, n2, copy1)
                        if child1.tree.depth <= self.config.max_depth and child2.tree.depth <= self.config.max_depth:
                            return child1, child2
                        child1 = parent1.copy()
                        child2 = parent2.copy()
                        break

        return child1, child2

    def crossover(self, parent1: GPIndividual, parent2: GPIndividual) -> Tuple[GPIndividual, GPIndividual]:
        """Override crossover with type-safe version."""
        return self._typed_crossover(parent1, parent2)

    def type_infer(self, node: GPNode) -> Optional[str]:
        """Infer the return type of a node."""
        if isinstance(node, STGPNode):
            return node.return_type.name
        if node.node_type == 'terminal':
            try:
                float(node.value)
                return 'float'
            except ValueError:
                return 'float'
        if node.value in ('lt', 'gt', 'eq', 'and_', 'or_', 'not_'):
            return 'bool'
        return 'float'


# =============================================================================
# 12. AutoConstructiveGP - Self-Modifying GP with ADFs
# =============================================================================

class ADF:
    """Automatically Defined Function."""

    def __init__(self, name: str, num_args: int, body: Optional[GPNode] = None):
        self.name = name
        self.num_args = num_args
        self.body = body
        self.arg_names = [f'{name}_arg{i}' for i in range(num_args)]
        self.fitness_contribution = 0.0
        self.usage_count = 0

    def evaluate(self, args: List[float],
                 func_impl: Optional[Dict[str, Callable]] = None) -> float:
        """Evaluate the ADF body with given arguments."""
        if self.body is None:
            return 0.0
        inputs = {name: val for name, val in zip(self.arg_names, args)}
        return self.body.evaluate(inputs, func_impl)

    def copy(self) -> 'ADF':
        return ADF(self.name, self.num_args,
                   self.body.copy() if self.body else None)


class AutoConstructiveGP(GeneticProgramming):
    """Self-modifying GP with Automatically Defined Functions."""

    def __init__(self, config: Optional[GPConfig] = None,
                 max_adfs: int = 3, max_adf_args: int = 3,
                 adf_mutation_rate: float = 0.1):
        super().__init__(config)
        self.max_adfs = max_adfs
        self.max_adf_args = max_adf_args
        self.adf_mutation_rate = adf_mutation_rate
        self.adfs: Dict[int, List[ADF]] = {}  # individual index -> ADFs

    def _create_random_adf(self, adf_index: int) -> ADF:
        """Create a random ADF."""
        num_args = random.randint(0, self.max_adf_args)
        name = f'ADF{adf_index}'
        adf = ADF(name, num_args)

        # Create a terminal set that includes ADF arguments
        adf_term_set = TerminalSet(
            num_variables=self.config.num_variables,
            constant_min=self.config.constant_min,
            constant_max=self.config.constant_max
        )
        adf_term_set.variables.extend(adf.arg_names)

        max_body_depth = random.randint(2, min(4, self.config.max_init_depth))
        body_tree = GPTree.generate_grow(self.func_set, adf_term_set, max_body_depth)
        adf.body = body_tree.root
        return adf

    def _evaluate_with_adfs(self, individual: GPIndividual,
                            inputs: Dict[str, float]) -> float:
        """Evaluate an individual's tree with ADF support."""
        ind_idx = self.population.index(individual) if individual in self.population else -1
        adf_list = self.adfs.get(ind_idx, [])

        # Build extended function implementations including ADFs
        extended_impl = dict(self.func_impl)
        for adf in adf_list:
            adf_ref = adf
            def make_adf_func(a, ar=adf_ref):
                def adf_func(*args):
                    return a.evaluate(list(args), extended_impl)
                return adf_func
            extended_impl[adf_ref.name] = make_adf_func(adf_ref)

        return individual.tree.evaluate(inputs, extended_impl)

    def initialize_population(self) -> List[GPIndividual]:
        """Initialize population with optional ADFs."""
        super().initialize_population()
        self.adfs = {}
        for i, ind in enumerate(self.population):
            self.adfs[i] = []
            num_adfs = random.randint(0, self.max_adfs)
            for j in range(num_adfs):
                self.adfs[i].append(self._create_random_adf(j))
        return self.population

    def evaluate_population(self) -> List[GPIndividual]:
        """Evaluate population with ADF support."""
        for idx, ind in enumerate(self.population):
            if self.config.max_evaluations is not None and self.evaluations >= self.config.max_evaluations:
                break
            predicted = []
            for inputs in self.X:
                try:
                    val = self._evaluate_with_adfs(ind, inputs)
                    if math.isnan(val) or math.isinf(val):
                        val = 1e6
                    predicted.append(max(-1e6, min(1e6, val)))
                except Exception:
                    predicted.append(1e6)

            if config_target_is_regression(self.config):
                raw_error = GPFitness.rmse(predicted, self.y)
            else:
                raw_error = 1.0 - GPFitness.accuracy(predicted, self.y)

            fitness = raw_error + self.config.parsimony_coefficient * ind.complexity
            ind.fitness = fitness
            ind.raw_fitness = raw_error
            self.evaluations += 1

        self.population.sort(key=lambda ind: ind.fitness)
        if self.best_individual is None or self.population[0].fitness < self.best_individual.fitness:
            self.best_individual = self.population[0].copy()
        return self.population

    def _mutate_adfs(self, individual_idx: int):
        """Mutate the ADFs of an individual."""
        if individual_idx not in self.adfs:
            self.adfs[individual_idx] = []

        adf_list = self.adfs[individual_idx]

        if random.random() < 0.3 and len(adf_list) < self.max_adfs:
            # Add a new ADF
            new_adf = self._create_random_adf(len(adf_list))
            adf_list.append(new_adf)
        elif random.random() < 0.2 and len(adf_list) > 0:
            # Remove an ADF
            idx = random.randint(0, len(adf_list) - 1)
            adf_list.pop(idx)
        elif len(adf_list) > 0:
            # Mutate an existing ADF's body
            idx = random.randint(0, len(adf_list) - 1)
            adf = adf_list[idx]
            adf_term_set = TerminalSet(
                num_variables=self.config.num_variables,
                constant_min=self.config.constant_min,
                constant_max=self.config.constant_max
            )
            adf_term_set.variables.extend(adf.arg_names)
            adf.body = GPTree.subtree_mutation(
                GPTree(adf.body), self.func_set, adf_term_set,
                self.config.max_depth // 2
            ).root

    def _create_next_generation(self) -> List[GPIndividual]:
        """Build next generation with ADF mutation."""
        next_gen = super()._create_next_generation()
        new_adfs = {}
        for i, ind in enumerate(next_gen):
            old_idx = None
            for j, old_ind in enumerate(self.population):
                if old_ind.tree.to_string() == ind.tree.to_string():
                    old_idx = j
                    break
            if old_idx is not None and old_idx in self.adfs:
                new_adfs[i] = [adf.copy() for adf in self.adfs[old_idx]]
                if random.random() < self.adf_mutation_rate:
                    self._mutate_adfs(i)
            else:
                new_adfs[i] = []
        self.adfs = new_adfs
        return next_gen

    def crossover(self, parent1: GPIndividual, parent2: GPIndividual) -> Tuple[GPIndividual, GPIndividual]:
        """Crossover that also exchanges ADFs."""
        child1, child2 = super().crossover(parent1, parent2)
        # Exchange ADFs with some probability
        if random.random() < 0.5:
            idx1 = self.population.index(parent1) if parent1 in self.population else -1
            idx2 = self.population.index(parent2) if parent2 in self.population else -1
            adfs1 = self.adfs.get(idx1, [])
            adfs2 = self.adfs.get(idx2, [])
            if adfs1 and adfs2:
                i = random.randint(0, min(len(adfs1), len(adfs2)) - 1)
                adfs1[i], adfs2[i] = adfs2[i].copy(), adfs1[i].copy()
        return child1, child2

    def get_adfs(self) -> Dict[int, List[ADF]]:
        """Return the ADFs for all individuals."""
        return self.adfs


# =============================================================================
# Helper Functions
# =============================================================================

def config_target_is_regression(config: GPConfig) -> bool:
    """Check if the config targets regression."""
    return config.target == 'regression'


def create_symbolic_regression_gp(num_variables: int = 1,
                                  population_size: int = 500,
                                  max_generations: int = 50,
                                  random_seed: Optional[int] = None) -> GeneticProgramming:
    """Factory function for symbolic regression GP."""
    config = GPConfig(
        population_size=population_size,
        max_generations=max_generations,
        num_variables=num_variables,
        target='regression',
        random_seed=random_seed,
        parsimony_coefficient=0.0005,
        tournament_size=7,
        crossover_rate=0.85,
        mutation_rate=0.10,
    )
    return GeneticProgramming(config)


def create_classification_gp(num_variables: int = 1,
                             population_size: int = 500,
                             max_generations: int = 50,
                             random_seed: Optional[int] = None) -> GeneticProgramming:
    """Factory function for classification GP."""
    config = GPConfig(
        population_size=population_size,
        max_generations=max_generations,
        num_variables=num_variables,
        target='classification',
        random_seed=random_seed,
        parsimony_coefficient=0.001,
        tournament_size=7,
    )
    return GeneticProgramming(config)


# =============================================================================
# Self-Test / Demo
# =============================================================================

def _self_test():
    """Run a quick self-test to verify the implementation works."""
    random.seed(42)
    print("=== GP Core Self-Test ===")

    # Test GPNode
    node = GPNode('function', 'add', [
        GPNode('terminal', 'x1'),
        GPNode('function', 'mul', [
            GPNode('terminal', '2.0'),
            GPNode('terminal', 'x2')
        ])
    ])
    assert node.depth() == 2
    assert node.size() == 5
    result = node.evaluate({'x1': 3.0, 'x2': 4.0})
    assert abs(result - 11.0) < 1e-10, f"Expected 11.0, got {result}"
    print(f"  GPNode evaluate: x1 + 2*x2 = {result} (expected 11.0) OK")

    # Test FunctionSet
    fs = FunctionSet()
    assert len(fs) > 0
    assert 'add' in fs
    print(f"  FunctionSet: {len(fs)} functions registered OK")

    # Test TerminalSet
    ts = TerminalSet(num_variables=3)
    assert len(ts.variables) == 3
    print(f"  TerminalSet: {len(ts)} terminals available OK")

    # Test GPTree generation
    trees = GPTree.generate_ramped_half_and_half(fs, ts, 20, 2, 4)
    assert len(trees) == 20
    print(f"  Ramped half-and-half: {len(trees)} trees generated OK")

    # Test crossover
    c1, c2 = GPTree.subtree_crossover(trees[0], trees[1], max_depth=10)
    assert c1.depth <= 10
    assert c2.depth <= 10
    print(f"  Subtree crossover: depths {c1.depth}, {c2.depth} OK")

    # Test mutations
    m1 = GPTree.subtree_mutation(trees[0], fs, ts, max_depth=10)
    m2 = GPTree.point_mutation(trees[0], fs, ts)
    m3 = GPTree.hoist_mutation(trees[0])
    m4 = GPTree.permutation_mutation(trees[0])
    m5 = GPTree.constant_mutation(trees[0])
    print(f"  Mutations: subtree, point, hoist, permutation, constant OK")

    # Test GPConfig
    config = GPConfig(population_size=100, max_generations=10, num_variables=1)
    assert config.population_size == 100
    print(f"  GPConfig: population={config.population_size} OK")

    # Test GPFitness
    preds = [1.0, 2.0, 3.0, 4.0]
    actual = [1.1, 2.2, 2.9, 4.1]
    rmse_val = GPFitness.rmse(preds, actual)
    mae_val = GPFitness.mae(preds, actual)
    r2_val = GPFitness.r_squared(preds, actual)
    assert rmse_val >= 0
    assert r2_val > 0.9
    print(f"  GPFitness: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, R2={r2_val:.4f} OK")

    # Test Selection
    pop = [GPIndividual(GPTree.generate_grow(fs, ts, 3), fitness=float(i))
           for i in range(20, 0, -1)]
    ts_sel = TournamentSelection(5)
    selected = ts_sel.select(pop)
    assert selected.fitness <= 5.0
    print(f"  TournamentSelection: selected fitness={selected.fitness:.1f} OK")

    rw_sel = RouletteWheelSelection()
    selected_rw = rw_sel.select(pop)
    print(f"  RouletteWheelSelection: selected fitness={selected_rw.fitness:.1f} OK")

    # Test BloatControl
    bc = BloatControl(max_depth=10, max_size=100)
    assert bc.check_depth_limit(GPTree.generate_grow(fs, ts, 5))
    print(f"  BloatControl: depth/size checks OK")

    # Test full GP run (symbolic regression: y = x^2 + 2*x + 1)
    print("\n  Running symbolic regression (y = x^2 + 2x + 1)...")
    X = [{'x1': float(i)} for i in range(-5, 6)]
    y = [x['x1'] ** 2 + 2 * x['x1'] + 1 for x in X]

    gp = GeneticProgramming(GPConfig(
        population_size=200,
        max_generations=30,
        max_depth=8,
        max_init_depth=4,
        num_variables=1,
        random_seed=42,
        parsimony_coefficient=0.001,
        tournament_size=5,
    ))
    best = gp.run(X, y)
    print(f"  Best fitness: {best.fitness:.6f}")
    print(f"  Best expression: {best.tree.to_string()}")
    print(f"  Best size: {best.complexity}, depth: {best.depth}")

    # Test prediction
    test_X = [{'x1': 3.0}]
    preds = gp.predict(test_X)
    expected = 3.0 ** 2 + 2 * 3.0 + 1  # = 16
    print(f"  Prediction for x=3: {preds[0]:.4f} (expected {expected:.1f})")

    # Test StronglyTypedGP
    print("\n  Testing StronglyTypedGP...")
    stgp = StronglyTypedGP(GPConfig(
        population_size=50, max_generations=5, num_variables=2, random_seed=42
    ))
    stgp.initialize_population()
    stgp.evaluate_population()
    print(f"  STGP population initialized and evaluated OK")
    print(f"  STGP best fitness: {stgp.population[0].fitness:.6f}")

    # Test AutoConstructiveGP
    print("\n  Testing AutoConstructiveGP...")
    acgp = AutoConstructiveGP(GPConfig(
        population_size=30, max_generations=3, num_variables=1, random_seed=42
    ), max_adfs=2)
    acgp.initialize_population()
    acgp.evaluate_population()
    total_adfs = sum(len(v) for v in acgp.adfs.values())
    print(f"  ACGP initialized with {total_adfs} ADFs across population OK")

    print("\n=== All self-tests passed! ===")


if __name__ == '__main__':
    _self_test()
