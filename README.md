![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Release](https://img.shields.io/badge/v1.0%20release-end%20of%202025-blue)

[![PyPI version](https://img.shields.io/pypi/v/earlysign.svg)](https://pypi.org/project/earlysign/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/earlysign)
[![Documentation](https://img.shields.io/badge/docs-early--sign-blue?label=documentation)](https://early-sign.github.io/EarlySign/)

<!-- [![TestPyPI version](https://img.shields.io/pypi/v/earlysign?label=test-pypi&pypiBaseUrl=https://test.pypi.org&color=lightgray)](https://test.pypi.org/project/earlysign/) -->

# EarlySign

<center><picture>
<img src="https://raw.githubusercontent.com/early-sign/EarlySign/refs/heads/main/docs/logo.png" width="80%"><br/>
Early signs, faster decisions.
</picture></center>

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

This library supports the following steps that emerge in your implementation of statistical practice.

1. Planning / Designing
1. Executing / Analyzing
1. Reporting / Visualizing
1. (optionally) Educating

## ✨ Key Features

<table border="1" style="border-collapse: collapse; border: 2px solid #ddd;">
<tr>
<td width="50%" align="center" style="border: 1px solid #ddd; padding: 20px;">

<div style="font-size: 48px; margin-bottom: 10px;">🗄</div>

### Auditable by Design
**via Event Sourcing Pattern**

</td>
<td width="50%" align="center" style="border: 1px solid #ddd; padding: 20px;">

<div style="font-size: 48px; margin-bottom: 10px;">🎁</div>

### Usage Templates
**for Off-the-shelf Usability**

</td>
</tr>
<tr>
<td width="50%" align="center" style="border: 1px solid #ddd; padding: 20px;">

<div style="font-size: 48px; margin-bottom: 10px;">🔧</div>

### Development Framework
**for Customizability**

</td>
<td width="50%" align="center" style="border: 1px solid #ddd; padding: 20px;">

<div style="font-size: 48px; margin-bottom: 10px;">📙</div>

### Standardized Schema
**the ES3 Spec Format**

</td>
</tr>
</table>

**1. Auditable by Design (via Event Sourcing Pattern)**
Every statistical decision is traceable and reproducible. The event sourcing pattern ensures complete audit trails of your sequential testing procedures, making your analyses transparent and verifiable.

**2. Usage Templates for Off-the-shelf Usability**
Get started quickly with pre-built protocol templates. Common experimental designs (A/B tests, monitoring, etc.) are ready to use out of the box, allowing you to focus on your experiments.

**3. Development Framework for Customizability**
Develop your own sequential testing procedures to fit your needs. EarlySign is a platform for building custom methods while taking advantage of the utilities provided by the framework.

**4. Standardized Schema for Protocol Specification (the ES3 Spec Format)**
We have a clearly defined and declared protocol specification schema, ES3 (EarlySign Schema Specification), that ensures consistency, portability, and clear communication of experimental designs across teams and tools.

## Citing
If you use EarlySign in publications, presentations, or reports, please consider citing it.
Example citations are provided below.
```bibtex
@misc{earlysign2025,
    title = {EarlySign Python package},
    author = {{Takeshi Teshima}},
    year = {2025},
    howpublished = {\url{https://github.com/early-sign/EarlySign}},
}
```

```
Takeshi Teshima (2025). EarlySign Python package. https://github.com/early-sign/EarlySign.
```
