# Developing Your Own Methods and Config Schema

EarlySign is designed to be highly extensible. While it provides powerful built-in methods like Group Sequential Testing (GST) and Anytime Valid Inference (AVI), you can define your own statistical methods and configuration schemas while still leveraging the core **EarlySign Standard Schema (ES3)**.

---

## 1. Defining Your Custom Method

Every EarlySign protocol follows the fundamental pattern:
**Protocol = Task + Method**

To create a custom method, you define two Pydantic models inheriting from the core ES3 specifications.

### A. Define the Task and Method Models

```python
from typing import Literal
from pydantic import Field
from earlysign.schema.ES3.base import TaskSpec, MethodSpec, Protocol

class MyCustomTask(TaskSpec):
    """Define the parameters of your specific problem."""
    kind: Literal["my_custom_task"] = "my_custom_task"
    target_metric: str = Field(..., description="The name of the metric to monitor.")

class MyCustomMethod(MethodSpec):
    """Define how the problem is solved operationally."""
    kind: Literal["my_custom_method"] = "my_custom_method"
    learning_rate: float = 0.01

class MyCustomProtocol(Protocol):
    """Combine them into a full Protocol."""
    task: MyCustomTask
    method: MyCustomMethod
```

---

## 2. Exporting Your Schema

To enable IDE autocompletion (VSCode) and offline validation for your custom configuration files, we recommend **Bundled (Inlined)** schema generation.

### A. Recommended Strategy: Bundled Export

Bundling inlines all EarlySign Core (ES3) definitions directly into your custom `schema.json`. This makes the file completely self-contained—zero configuration is required for VSCode to provide full IntelliSense and validation.

```python
from pathlib import Path
from earlysign.parts.export_schema import export_json_schema
from my_package.schema import MyCustomProtocol

# This generates a single, portable JSON file
export_json_schema(
    MyCustomProtocol, 
    output_path=Path("my_package/schema.json"),
    bundle=True  # Recommended for zero-touch IDE support
)
```

### B. Specialized Strategies (Advanced)
If you prefer a modular setup or stable global identifiers, you can toggle the `bundle` and `use_relative` flags:

- **Modular Relative**: Use `bundle=False, use_relative=True`. Best for developing extensions inside a fork or mono-repo of EarlySign.
- **Canonical URI**: Use `bundle=False, use_relative=False`. Best for stable, versioned public distribution where you want a globally unique `$id`.

---

## 3. Best Practices

1. **Use Literals for `kind`**: This acts as a discriminator for both Pydantic and JSON Schema.
2. **Offline-First**: Bundled schemas work 100% offline. No `SchemaRegistry` or network connection is required for tools to validate your protocol configuration.
3. **Keep it Consistent**: Whenever you modify your Pydantic models, re-run the `export_json_schema` script to keep your `.json` configurations in sync.
