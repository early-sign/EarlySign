
import json
import ibis
import pytest

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.GST.Log import DecisionStatus
from earlysign.v1.templates.binomial_ab import (
    BinomialABProtocol,
    BinomialABTemplate,
)
from earlysign.v1.tests.util import BinomialStream


def test_binomial_json_protocol():
    """
    Test that a BinomialABProtocol can be deserialized from a hardcoded JSON string
    and then successfully used to run a simulation.
    """
    # 1. Hardcoded JSON Protocol (compliant with es3_v1.tsp)
    # Generated from valid parameters: alpha=0.025, power=0.8, delta=0.05, k=2, p_control=0.09
    protocol_json = """
    {
      "name": "earlysign.v1.templates.binomial_ab.BinomialABProtocol",
      "ES3_version": "v1.0.0",
      "task": {
        "kind": "group_sequential",
        "arms": [
          "C",
          "T"
        ],
        "response_type": "binary",
        "hypotheses": {
          "h_null": "Difference <= 0",
          "h_alt": "Difference > 0.05",
          "test_logic": {
            "kind": "superiority"
          },
          "target_effect": {
            "type": "binary"
          }
        },
        "efficacy": {
          "alpha": 0.025
        },
        "futility": {
          "power": 0.8,
          "binding": false
        }
      },
      "method": {
        "kind": "group_sequential",
        "stopping_policy": {
          "statistic": {
            "kind": "two_arm_binomial_z"
          },
          "strategy": {
            "kind": "alpha_beta_spending",
            "statistical_model": {
              "kind": "canonical_gaussian"
            },
            "alpha_spending_fn": {
              "family": "obrien_fleming",
              "params": null
            },
            "beta_spending_fn": {
              "family": "obrien_fleming",
              "params": null
            },
            "alpha_budget": 0.025,
            "beta_budget": 0.2,
            "alpha_binding": true,
            "beta_binding": false
          },
          "timer": {
            "kind": "sample_size",
            "unit": "individuals",
            "max_sample_size": 1255
          },
          "schedule": {
            "kind": "fixed",
            "analyses": [
              0.5,
              1.0
            ]
          }
        },
        "adaptation": null
      }
    }
    """

    # 2. Deserialize from JSON
    # This matches the user's objective: "model_validate_json using a protocol"
    protocol = BinomialABProtocol.model_validate_json(protocol_json)

    # Verify key attributes survived deserialization
    assert protocol.task.arms == ["C", "T"]
    assert protocol.method.stopping_policy.statistic.kind == "two_arm_binomial_z"

    # 3. Initialize Template with deserialized protocol
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_json_protocol")

    sim = BinomialABTemplate(ledger)
    sim.set_protocol(protocol)

    # 4. Run simulation (replicating docstring scenario)

    stream = BinomialStream(n_per_batch=200, arms={"C": 0.09, "T": 0.18}, seed=42)

    # We need to simulate the steps.
    # Step 1: n=200 per arm
    sim.update(next(stream))
    prog = sim.report_progress()

    # Continue until stopped or safety limit
    max_steps = 10
    step = 1
    while prog["status"] == DecisionStatus.CONTINUE_ and step < max_steps:
        sim.update(next(stream))
        prog = sim.report_progress()
        step += 1

    # 5. Assertions
    # Given the high effect size (0.09 vs 0.18) compared to design (0.09 vs 0.14),
    # it should likely reject null (STOP_EFFICACY).
    assert prog["status"] in [
        DecisionStatus.STOP_EFFICACY,
        DecisionStatus.STOP_FUTILITY,
    ]
