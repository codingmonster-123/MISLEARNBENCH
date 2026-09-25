"""Base dataset classes for Evo-Memory.

Evo-Memory restructures conventional static my_datasets into streaming task
sequences, enabling evaluation of how LLMs reuse and evolve memory over time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Callable
from enum import Enum
import random


class DatasetSplit(Enum):
    """Dataset splits."""
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
    NO_SPLIT = "no_split"


@dataclass
class TaskInstanceEVOMEM:
    """
    A single task instance in the streaming evaluation.

    Attributes:
        task_id: Unique identifier for this task
        scenario: the scenario of the task
        initial_user_query: the initial user query
        normative_trajectory: the normative trajectory of the task
        cross_session_queries: queries for cross-session generalization(Same to the initial user query but in a different session)
        cross_task_generalization_queries: queries for cross-task generalization(not same as the initial user query)
    """
    task_id: str
    scenario: str
    initial_user_query: str
    normative_trajectory: List[str] = field(default_factory=list)
    cross_session_queries: List[str] = field(default_factory=list)
    cross_task_generalization_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scenario": self.scenario,
            "initial_user_query": self.initial_user_query,
            "normative_trajectory": self.normative_trajectory,
            "cross_session_queries": self.cross_session_queries,
            "cross_task_generalization_queries": self.cross_task_generalization_queries,
        }



class BaseDataset(ABC):
    """
    Abstract base class for Evo-Memory my_datasets.

    Datasets are converted into streaming task sequences:
    τ = {(x_1, y_1), ..., (x_T, y_T)}
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to dataset files
            seed: Random seed for shuffling
            shuffle: Whether to shuffle the data
            max_samples: Maximum number of samples to use
        """
        self.data_path = data_path
        self.seed = seed
        self.shuffle = shuffle
        self.max_samples = max_samples

        self._instances: List[TaskInstanceEVOMEM] = []
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass



    @abstractmethod
    def _load_data(self) -> List[TaskInstanceEVOMEM]:
        """Load data from source. To be implemented by subclasses."""
        pass



    def load(self) -> None:
        """Load the dataset."""
        if self._loaded:
            return

        self._instances = self._load_data()

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self._instances)

        if self.max_samples:
            self._instances = self._instances[:self.max_samples]

        self._loaded = True

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._instances)

    def __iter__(self) -> Iterator[TaskInstanceEVOMEM]:
        """Iterate over task instances in streaming fashion."""
        if not self._loaded:
            self.load()
        return iter(self._instances)

    def __getitem__(self, idx: int) -> TaskInstanceEVOMEM:
        if not self._loaded:
            self.load()
        return self._instances[idx]


    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._loaded:
            self.load()

        # Count by scenario
        scenarios = {}
        difficulties = {}

        for inst in self._instances:
            scenario = inst.scenario or "unknown"
            scenarios[scenario] = scenarios.get(scenario, 0) + 1

        return {
            "name": self.name,
            "total_instances": len(self._instances),
            "scenarios": scenarios
        }





