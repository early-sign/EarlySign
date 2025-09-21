"""
Base template infrastructure for EarlySign (scheme-agnostic).

This module defines an abstract base class for *templates* that orchestrate
scheme-specific records and operators into a ready-to-use analysis pipeline.
Templates live under ``earlysign.templates.*`` and are intended to be portable
facades over lower-level framework modules.

Design goals
------------
- Keep the *ledger* and *operators* as the source of truth
- Let each template declare its components and pipeline in a single place
- Support multiple testing styles (Interim Analysis / Safe Testing) uniformly
- Be compatible with the framework's injection model
  (``__init__(scoped, **inputs)`` + ``derived_records()`` + ``self.outputs[...]``)

Quick example
-------------
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.framework.templates import TemplateBase, AnalysisResult
>>>
>>> class NoopTemplate(TemplateBase):
...     \"\"\"Minimal example that does not touch the ledger.\"\"\"
...     def build_components(self) -> None:
...         self.registry['ids'] = {}  # normally record ids go here
...     def add_observations(self, **kwargs):
...         pass  # subclasses decide how to persist inputs
...     def run_analysis(self) -> AnalysisResult:
...         return AnalysisResult(signal='continue', reason='none', stopped=False,
...                               summary={'template': 'noop'}, raw={})
>>> import ibis
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, 'table')
>>> ledger.ensure()
>>> t = NoopTemplate(experiment_id='exp1')
>>> t.setup(ledger.bind(experiment_id='exp1'))
>>> res = t.analyze()
>>> res.signal
'continue'
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

from earlysign.core.ledger import Ledger


@dataclass
class AnalysisResult:
    """Structured outcome of a template ``analyze()`` run.

    Parameters
    ----------
    signal : str
        High-level outcome (e.g., 'stop_efficacy', 'stop_futility', 'reject', 'continue').
    reason : str
        Short reason tag ('efficacy', 'futility', 'ville', 'none', ...).
    stopped : bool
        Whether the process should stop (True) or continue (False).
    summary : dict
        Lightweight JSON-serializable summary for UI / logs.
    raw : dict
        Raw details (e.g., last rows of records) for debugging or downstream use.
    """

    signal: str
    reason: str
    stopped: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class TemplateBase(ABC):
    """Abstract base class for experiment templates (scheme-agnostic).

    Lifecycle
    ---------
    1) ``setup(scoped_ledger)``
       - Save scoped ledger
       - Build components (record IDs, operator wiring)
    2) ``add_observations(**kwargs)`` (repeatable)
       - Insert new observations into input records (template-defined)
    3) ``analyze()`` (repeatable)
       - Execute the pipeline (operators) and return :class:`AnalysisResult`

    Subclasses must implement :meth:`build_components`, :meth:`add_observations`,
    and :meth:`run_analysis`. Any notion of "look", "step", or "time" is
    **intentionally left to subclasses** to define and manage.
    """

    def __init__(self, experiment_id: str, *, namespace: Optional[str] = None):
        self.experiment_id = str(experiment_id)
        self.namespace = str(namespace) if namespace is not None else "default"
        self._scoped: Optional[Ledger] = None
        self._is_setup: bool = False

        # Components/IDs/metadata kept here by convention
        # e.g., self.registry['ids'] = {'counts': 'exp:ns:counts', 'wald': '...'}
        self.registry: Dict[str, Any] = {}

        # Free-form per-template runtime state (subclasses may use it or ignore it)
        self.state: Dict[str, Any] = {}

    # -------------------------- properties & helpers --------------------------

    @property
    def scoped(self) -> Ledger:
        """Return the scoped ledger set via :meth:`setup`.

        Examples
        --------
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.framework.templates import TemplateBase
        >>> class T(TemplateBase):
        ...     def build_components(self): pass
        ...     def add_observations(self, **k): pass
        ...     def run_analysis(self): pass
        ...
        >>> import ibis
        >>> con = ibis.duckdb.connect(":memory:")
        >>> ledger = Ledger(con, 'table')
        >>> ledger.ensure()
        >>> scoped = ledger.bind(experiment_id='e1')
        >>> t = T('e1'); t.setup(scoped)
        >>> isinstance(t.scoped, Ledger)
        True
        """
        if self._scoped is None:
            raise RuntimeError(
                "Template is not setup. Call setup(scoped_ledger) first."
            )
        return self._scoped

    def mkid(self, name: str) -> str:
        """Create a stable record/operator identifier for this template.

        Default pattern: ``f"{experiment_id}:{namespace}:{name}"``.

        Examples
        --------
        >>> from earlysign.framework.templates import TemplateBase
        >>> class T(TemplateBase):
        ...     def build_components(self): pass
        ...     def add_observations(self, **k): pass
        ...     def run_analysis(self): pass
        ...
        >>> T('exp1', namespace='tp').mkid('counts')
        'exp1:tp:counts'
        """
        return f"{self.experiment_id}:{self.namespace}:{name}"

    # ------------------------------ lifecycle --------------------------------

    def setup(self, scoped: Ledger) -> None:
        """Bind a scoped ledger and build components.

        Subclasses should override :meth:`build_components` to register:
        - input records (to be attached via ``.attach(scoped)`` when used)
        - output record ids (strings) that operators will use via ``derived_records()``
        - design/parameters for operators
        """
        self._scoped = scoped
        self._is_setup = True
        self.registry.clear()
        self.state.clear()
        self.build_components()

    @abstractmethod
    def build_components(self) -> None:
        """Declare component IDs/objects under ``self.registry``.

        Recommended keys
        ----------------
        - ``ids`` : dict of record ids (e.g., {'counts': 'exp:ns:counts', 'wald': '...'})
        - ``design`` : dict of design payloads
        - ``params`` : dict of parameter defaults
        - ``notes`` : misc metadata
        """
        raise NotImplementedError

    # @abstractmethod
    # def add_observations(self, **kwargs: Any) -> None:
    #     """Insert new observations into input records.

    #     Implementations decide how to persist inputs (counts, metrics, etc.)
    #     and whether to maintain any notion of look/step/time in ``self.state``
    #     or directly in record payloads (e.g., ``payload['look']``).
    #     """
    #     raise NotImplementedError

    def analyze(self) -> AnalysisResult:
        """Run the analysis pipeline and return a structured result.

        Subclasses may override this to customize pre/post hooks. The default
        implementation delegates to :meth:`run_analysis`.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup(scoped_ledger) before analyze().")
        return self.run_analysis()

    @abstractmethod
    def run_analysis(self) -> AnalysisResult:
        """Execute operators and synthesize an :class:`AnalysisResult`."""
        raise NotImplementedError

    # ------------------------------- utilities --------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a generic, JSON-serializable summary for UIs/logs.

        Subclasses can extend by merging additional fields.
        """
        base: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "namespace": self.namespace,
            "status": "ready" if self._is_setup else "not_setup",
        }
        ids = self.registry.get("ids")
        if isinstance(ids, dict):
            base["components"] = sorted(ids.keys())
        if self.state:
            base["state"] = dict(self.state)
        return base

    def reset(self) -> None:
        """Reset per-run state (keeps configuration/registry intact)."""
        self.state.clear()
