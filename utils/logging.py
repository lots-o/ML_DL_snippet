from collections import defaultdict
from typing import Dict
import copy


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self._reset()

    def _reset(self):
        self._metrics: Dict[str, dict] = defaultdict(
            lambda: {"val": 0, "count": 0, "avg": 0}
        )

    def update(self, metric_name, val):
        metric = self._metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return "|".join(
            [
                f'{metric_name}: {metric["avg"]:.{self.float_precision}f}'
                for (metric_name, metric) in self.metrics.items()
            ]
        )

    @property
    def metrics(self):
        return copy.deepcopy(self._metrics)
