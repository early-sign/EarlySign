![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Release](https://img.shields.io/badge/v1.0%20release-end%20of%202025-blue)

[![PyPI version](https://img.shields.io/pypi/v/earlysign.svg)](https://pypi.org/project/earlysign/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/earlysign)
[![Documentation](https://img.shields.io/badge/docs-early--sign-blue?label=documentation)](https://early-sign.github.io/EarlySign/)

[![TestPyPI version](https://img.shields.io/pypi/v/earlysign?label=test-pypi&pypiBaseUrl=https://test.pypi.org&color=lightgray)](https://test.pypi.org/project/earlysign/)

# EarlySign

<center>
<img src="https://raw.githubusercontent.com/early-sign/EarlySign/refs/heads/main/docs/logo.png" width="80%"><br/>
Early signs, faster decisions.
</center>

---

## What is this?

EarlySign is a Python library for sequential/safe testing (alpha-spending, e-processes, etc.).

1. Group sequential tests for interim analysis
    - By using alpha-spending functions to control the overall Type I error rate, you can stop early for efficacy or futility, making your experiments more efficient without compromising statistical integrity. This approach allows for a pre-specified number of interim analyses during an experiment.
1. e-processes for anytime-valid inference
    - It allows you to continuously monitor your experiments and make decisions as soon as the evidence is strong enough, without waiting for a predetermined sample size. This can lead to faster conclusions, saving time and resources, while maintaining statistical rigor.


## Quick Start

```
pip install earlysign
```

Please check our [up-to-date documentation](https://early-sign.github.io/EarlySign/) site for explanations, references, how-to's, and tutorials.

## Usage

This library supports the following steps in your experimentation.

1. Planning / Designing
1. Executing / Analyzing
1. Reporting / Visualizing
1. (optionally) Educating
