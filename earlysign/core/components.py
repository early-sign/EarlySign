"""
earlysign.core.components
=========================

Clean base classes for components that write to and read from the ledger.

Instead of using mixins/traits, these components work directly with
the Ledger class and use ibis expressions for querying. This provides
better separation of concerns and leverages ibis-framework's capabilities.

Component Types:
- `Statistic`: Compute and register statistical values from observations
- `Criteria`: Compute and register boundaries, thresholds, or critical values
- `Signaler`: Emit stop/continue decisions based on statistics and criteria
- `Recommender`: Emit recommendations or guidance (optional)
- `Observer`: Validate, transform, and register raw observations

Examples
--------
>>> from earlysign.core.ledger import Ledger, create_test_connection
>>> from earlysign.core.names import Namespace
>>>
>>> conn = create_test_connection("duckdb")
>>> ledger = Ledger(conn, "test")
>>>
>>> class DummyStat(Statistic):
...     def step(self, ledger, experiment_id, step_key, time_index):
...         # Direct ibis querying for observations
...         obs_count = (ledger.table
...                     .filter(ledger.table.namespace == str(self.ns_stats))
...                     .filter(ledger.table.entity.contains(str(experiment_id)))
...                     .count()
...                     .execute())
...
...         # Write computed statistic
...         ledger.write_event(
...             time_index=time_index,
...             namespace=self.ns_stats,
...             kind="updated",
...             experiment_id=experiment_id,
...             step_key=step_key,
...             payload_type="StatValue",
...             payload={"obs_count": obs_count}
...         )
...
>>> stat = DummyStat()
>>> stat.step(ledger, "exp1", "step1", "t1")
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union, Optional, TYPE_CHECKING

from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex

# Type aliases
NamespaceLike = Union[Namespace, str]

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


class ComponentBase(ABC):
    """
    Base class for all ledger components.

    Provides namespace conventions and requires subclasses to implement step().
    Components work directly with Ledger instances and use ibis for querying.
    """

    @abstractmethod
    def step(
        self,
        ledger: "Ledger",
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """
        Execute this component's logic for a given experiment step.

        Components should use ibis expressions on ledger.table for queries
        and ledger.write_event() for recording results.
        """
        pass


@dataclass(kw_only=True)
class Statistic(ComponentBase):
    """
    Base class for statistics updaters.

    Statistics read observations and compute statistical values,
    which they then write back to the ledger for use by other components.
    """

    ns_stats: NamespaceLike = Namespace.STATS
    tag_stats: str = "stat:generic"

    def step(
        self,
        ledger: "Ledger",
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Override this method to implement statistic computation."""
        raise NotImplementedError("Subclasses must implement step()")


@dataclass(kw_only=True)
class Criteria(ComponentBase):
    """
    Base class for criteria/boundary updaters.

    Criteria components compute decision boundaries, thresholds, or
    critical values based on current statistics and experimental design.
    """

    ns_crit: NamespaceLike = Namespace.CRITERIA
    tag_crit: str = "crit:generic"

    def step(
        self,
        ledger: "Ledger",
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Override this method to implement criteria computation."""
        raise NotImplementedError("Subclasses must implement step()")


@dataclass(kw_only=True)
class Signaler(ComponentBase):
    """
    Base class for signal emitters (e.g., stop/continue decisions).

    Signalers compare statistics to criteria and emit actionable signals
    about whether to continue, stop, or take other actions.
    """

    ns_sig: NamespaceLike = Namespace.SIGNALS
    tag_sig: str = "signal:generic"

    def step(
        self,
        ledger: "Ledger",
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Override this method to implement signaling logic."""
        raise NotImplementedError("Subclasses must implement step()")


@dataclass(kw_only=True)
class Recommender(ComponentBase):
    """
    Base class for recommendation emitters.

    Recommenders provide higher-level guidance based on experimental
    results and can suggest next actions or parameter adjustments.
    """

    ns_rec: NamespaceLike = Namespace.SIGNALS  # Often use same namespace as signals
    tag_rec: str = "recommendation:generic"

    def step(
        self,
        ledger: "Ledger",
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Override this method to implement recommendation logic."""
        raise NotImplementedError("Subclasses must implement step()")


@dataclass(kw_only=True)
class Observer(ComponentBase):
    """
    Base class for observation validators/transformers.

    Observers validate, clean, and transform raw observations before
    they are processed by statistical components.
    """

    ns_obs: NamespaceLike = Namespace.OBS
    tag_obs: str = "obs:generic"

    def step(
        self,
        ledger: "Ledger",
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        time_index: Union[TimeIndex, str],
    ) -> None:
        """Override this method to implement observation processing."""
        raise NotImplementedError("Subclasses must implement step()")


@dataclass(kw_only=True)
class Observation:
    """Base class for data observation components."""

    ns_obs: NamespaceLike = Namespace.OBS
    tag_obs: Optional[str] = "obs"

    def step(
        self, ledger: Ledger, experiment_id: str, step_key: str, time_index: str
    ) -> None:
        raise NotImplementedError
