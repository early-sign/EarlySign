---
applyTo: "earlysign/**"
---

# EarlySign Package Development Standards

## 📚 Documentation & Testing

### Documentation Requirements
- **Always update documentation**: Ensure docstrings and docs/ are current and deployable
- **Comprehensive docstrings**: Include mathematical formulations, parameter descriptions, and usage examples
- **API documentation**: Keep docs/ synchronized with code changes for public APIs

### Testing Strategy
- **Doctests first**: Use doctests as primary examples and basic testing
- **Standalone tests**: Reserve for complex scenarios, edge cases, and integration testing
- **Test coverage**: Maintain high coverage especially for core framework components
- **Example-driven**: Write tests that serve as usage examples for other developers

## 🔧 Coding Standards

### Import Conventions
- **Absolute imports only**: Always use `from earlysign.package.module import Component`
- **Poetry-based imports**: Leverage poetry's package resolution for consistent imports
- **No relative imports**: Avoid `from .module import Component` patterns

### Module Organization
- **`__init__.py` purpose**: Designed for documentation generation and namespace structure
  - ❌ **DO NOT** use for access restriction or selective imports
  - ✅ **DO** provide namespace-level docstrings with doctests
  - ✅ **DO** organize public API exports clearly
- **Direct access**: Components should be accessible via absolute paths regardless of `__init__.py`

### Code Quality
- **No development artifacts**: Remove working comments before final delivery
  - ❌ Examples: "this method moved from...", "TODO: refactor later"
  - ✅ Keep only comments that add value to future maintainers
- **Clean commits**: Ensure production code doesn't contain debug prints or temporary code
- **Consistent formatting**: Follow project formatting standards (enforced via `make format`)

### Type Safety
- **Type hints**: Use comprehensive type annotations for all public APIs
- **TypedDict**: Use for structured payloads and configuration objects
- **Protocol compliance**: Ensure components implement required protocols correctly
