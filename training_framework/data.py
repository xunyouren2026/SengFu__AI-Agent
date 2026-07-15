"""
Training Framework - Data Module

Provides data loading, sampling, and collation utilities implemented
from scratch using only the Python standard library.
"""

import random
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Dataset Base
# ---------------------------------------------------------------------------
class Dataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Return a single sample by index."""
        ...


class ListDataset(Dataset):
    """A simple dataset backed by a list."""

    def __init__(self, data: List[Any]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        return self._data[index]


class DictDataset(Dataset):
    """A dataset backed by a dictionary of lists (columnar format)."""

    def __init__(self, data: Dict[str, List[Any]]) -> None:
        self._data = data
        lengths = [len(v) for v in data.values()]
        if lengths:
            self._length = lengths[0]
            for l in lengths:
                if l != self._length:
                    raise ValueError("All columns must have the same length")
        else:
            self._length = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {k: v[index] for k, v in self._data.items()}

    def columns(self) -> List[str]:
        """Return column names."""
        return list(self._data.keys())


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
class RandomSampler:
    """Samples elements randomly, without replacement."""

    def __init__(self, data_source: Any, replacement: bool = False, seed: Optional[int] = None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._rng = random.Random(seed)
        self._length = len(data_source)

    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            for _ in range(self._length):
                yield self._rng.randint(0, self._length - 1)
        else:
            indices = list(range(self._length))
            self._rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return self._length


class SequentialSampler:
    """Samples elements sequentially, in order."""

    def __init__(self, data_source: Any) -> None:
        self.data_source = data_source
        self._length = len(data_source)

    def __iter__(self) -> Iterator[int]:
        yield from range(self._length)

    def __len__(self) -> int:
        return self._length


class WeightedRandomSampler:
    """Samples elements with probabilities proportional to given weights."""

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            weights: Weight for each sample.
            num_samples: Number of samples to draw. Defaults to len(weights).
            replacement: Whether to sample with replacement.
            seed: Random seed.
        """
        if not weights:
            raise ValueError("Weights must not be empty")
        self.weights = list(weights)
        total = sum(self.weights)
        self._probabilities = [w / total for w in self.weights]
        self._num_samples = num_samples or len(weights)
        self.replacement = replacement
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[int]:
        if self.replacement:
            # Use inverse CDF sampling
            for _ in range(self._num_samples):
                r = self._rng.random()
                cumulative = 0.0
                for i, p in enumerate(self._probabilities):
                    cumulative += p
                    if r <= cumulative:
                        yield i
                        break
                else:
                    yield len(self._probabilities) - 1
        else:
            # Weighted sampling without replacement using reservoir-style
            indices = list(range(len(self.weights)))
            self._rng.shuffle(indices)
            # Sort by weight for approximate weighted sampling
            indices.sort(key=lambda i: self._probabilities[i], reverse=True)
            yield from indices[: self._num_samples]

    def __len__(self) -> int:
        return self._num_samples


class BatchSampler:
    """Wraps a sampler to yield batches of indices."""

    def __init__(
        self,
        sampler: Any,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        """
        Args:
            sampler: Base sampler (RandomSampler, SequentialSampler, etc.).
            batch_size: Number of samples per batch.
            drop_last: Whether to drop the last incomplete batch.
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        sampler_len = len(self.sampler)
        if self.drop_last:
            return sampler_len // self.batch_size
        return math.ceil(sampler_len / self.batch_size)


class DistributedSampler:
    """Sampler that partitions data across multiple workers/processes."""

    def __init__(
        self,
        data_source: Any,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        """
        Args:
            data_source: Dataset or data source.
            num_replicas: Total number of workers/processes.
            rank: Rank of the current worker.
            shuffle: Whether to shuffle indices.
            seed: Random seed for shuffling.
            drop_last: Whether to drop tail data to make it evenly divisible.
        """
        self.data_source = data_source
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._num_samples = self._compute_num_samples()

    def _compute_num_samples(self) -> int:
        total = len(self.data_source)
        if self.drop_last:
            per_replica = total // self.num_replicas
            return per_replica
        else:
            per_replica = math.ceil(total / self.num_replicas)
            return per_replica

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)

        if self.drop_last:
            total_size = self._num_samples * self.num_replicas
            indices = indices[:total_size]
        else:
            # Pad to make evenly divisible
            total_size = self._num_samples * self.num_replicas
            padding = total_size - len(indices)
            if padding > 0:
                indices += indices[:padding]

        # Subset for this rank
        start = self.rank * self._num_samples
        end = start + self._num_samples
        yield from indices[start:end]

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
class Collator:
    """Data collation utilities for batching heterogeneous data."""

    @staticmethod
    def default_collate(batch: List[Any]) -> Any:
        """
        Collate a list of samples into a batch.

        Handles:
        - Scalars: returns list
        - Lists/tuples of same length: returns list of lists
        - Dicts: recursively collate each value
        - Mixed types: returns list

        Args:
            batch: List of samples from dataset.

        Returns:
            Collated batch.
        """
        if not batch:
            return batch

        sample = batch[0]

        if isinstance(sample, dict):
            collated = {}
            for key in sample:
                values = [item[key] for item in batch]
                collated[key] = Collator.default_collate(values)
            return collated

        if isinstance(sample, (list, tuple)):
            # Check if all elements are scalars (numbers/strings)
            if all(isinstance(x, (int, float, str, bool)) for x in sample):
                # List of lists -> transpose to batch format
                return [list(items) for items in zip(*batch)]
            else:
                # Nested structure: collate recursively
                return [Collator.default_collate(items) for items in zip(*batch)]

        # Scalars: just return as list
        return list(batch)

    @staticmethod
    def padded_collate(
        batch: List[Any],
        padding_value: float = 0.0,
        pad_to_length: Optional[int] = None,
    ) -> Any:
        """
        Collate with padding for variable-length sequences.

        Args:
            batch: List of samples (each a list/tuple of variable length).
            padding_value: Value to use for padding.
            pad_to_length: If set, pad all sequences to this length.

        Returns:
            Padded batch as list of lists.
        """
        if not batch:
            return batch

        sample = batch[0]

        if isinstance(sample, dict):
            collated = {}
            for key in sample:
                values = [item[key] for item in batch]
                collated[key] = Collator.padded_collate(values, padding_value, pad_to_length)
            return collated

        if isinstance(sample, (list, tuple)):
            # Determine max length
            if pad_to_length is not None:
                max_len = pad_to_length
            else:
                max_len = max(len(item) for item in batch)

            # Pad each sequence
            padded = []
            for item in batch:
                if len(item) < max_len:
                    padding = [padding_value] * (max_len - len(item))
                    padded.append(list(item) + padding)
                else:
                    padded.append(list(item[:max_len]))
            return padded

        return list(batch)

    @staticmethod
    def stack_collate(batch: List[Any]) -> List[List[Any]]:
        """
        Simple stack collation: convert list of items to list of columns.

        Args:
            batch: List of samples.

        Returns:
            Transposed list (columns first).
        """
        if not batch:
            return []
        return [list(col) for col in zip(*batch)]


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------
class DataLoader:
    """Data loader that combines a dataset and sampler, and provides batched iteration."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Any] = None,
        batch_sampler: Optional[Any] = None,
        drop_last: bool = False,
        collate_fn: Optional[Any] = None,
        num_workers: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data.
            sampler: Custom sampler (overrides shuffle).
            batch_sampler: Custom batch sampler (overrides batch_size/shuffle/drop_last).
            drop_last: Whether to drop the last incomplete batch.
            collate_fn: Custom collation function.
            num_workers: Number of worker processes (simulated, kept for API compatibility).
            seed: Random seed for shuffling.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or Collator.default_collate
        self.num_workers = num_workers
        self._seed = seed

        # Set up sampler
        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
        elif sampler is not None:
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        elif shuffle:
            self.batch_sampler = BatchSampler(
                RandomSampler(dataset, seed=seed), batch_size, drop_last
            )
        else:
            self.batch_sampler = BatchSampler(
                SequentialSampler(dataset), batch_size, drop_last
            )

        self._batch_iter: Optional[Iterator] = None

    def __iter__(self) -> Iterator[Any]:
        """Iterate over batches."""
        for indices in self.batch_sampler:
            batch = [self.dataset[idx] for idx in indices]
            yield self.collate_fn(batch)

    def __next__(self) -> Any:
        """Get next batch."""
        if self._batch_iter is None:
            self._batch_iter = iter(self)
        return next(self._batch_iter)

    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self.batch_sampler)

    def shuffle_data(self, seed: Optional[int] = None) -> None:
        """Re-shuffle the data with an optional new seed."""
        rng_seed = seed if seed is not None else self._seed
        self.batch_sampler = BatchSampler(
            RandomSampler(self.dataset, seed=rng_seed),
            self.batch_size,
            self.drop_last,
        )

    def get_batch_sizes(self) -> List[int]:
        """Return the size of each batch."""
        return [len(list(batch)) for batch in self.batch_sampler]

    def get_total_samples(self) -> int:
        """Return total number of samples across all batches."""
        return len(self.dataset)

    def get_num_batches(self) -> int:
        """Return number of batches."""
        return len(self)
