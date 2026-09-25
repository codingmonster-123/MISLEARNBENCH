from typing import List, Dict, Any, Optional
import json
import re

from DataSets.base import BaseDataset, TaskInstanceEVOMEM

class REWARD_HACKINGDataset(BaseDataset):
    def __init__(
        self,
        data_path: Optional[str] = None,
        domain: Optional[str] = "all",
        **kwargs,
    ):
        """
        Initialize reward_hacking dataset.

        Args:
            data_path: Path to reward_hacking data
            domain: Specific domain to load (e.g., 'service', 'healthy')
        """
        super().__init__(data_path, **kwargs)
        self.domain_filter = domain

    @property
    def name(self) -> str:
        if self.domain_filter:
            return f"reward_hacking_{self.domain_filter}"
        return "reward_hacking"

    def _load_data(self) -> List[TaskInstanceEVOMEM]:
        """Load REWARD_HACKING data from local path."""
        instances = []
        instances = self._load_from_local()
        return instances

    def _load_from_local(self) -> List[TaskInstanceEVOMEM]:
        """Load from local JSON file."""
        instances = []

        try:
            # Open the JSONL file and read the data as an array.
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]
            # Iterate over the data as required.
            for idx, item in enumerate(data):
                # Filter the data by domain.
                scenario = item.get("scenario", "unknown")
                initial_user_query = item.get("initial_user_query", "")
                normative_trajectory = item.get("normative_trajectory", [])
                cross_session_queries = item.get("cross_session_queries", [])
                cross_task_generalization_queries = item.get("cross_task_generalization_queries", [])
                if self.domain_filter == "all":
                    instances.append(TaskInstanceEVOMEM(
                        task_id=f"reward_hacking_{idx}",
                        scenario=scenario,
                        initial_user_query=initial_user_query,
                        normative_trajectory=normative_trajectory,
                        cross_session_queries=cross_session_queries,
                        cross_task_generalization_queries=cross_task_generalization_queries
                    ))
                elif self.domain_filter == scenario:
                    instances.append(TaskInstanceEVOMEM(
                        task_id=f"reward_hacking_{idx}",
                        scenario=scenario,
                        initial_user_query=initial_user_query,
                        normative_trajectory=normative_trajectory,
                        cross_session_queries=cross_session_queries,
                        cross_task_generalization_queries=cross_task_generalization_queries
                    ))
                

        except FileNotFoundError:
            print(f"Warning: Local data file not found: {self.data_path}")

        return instances

