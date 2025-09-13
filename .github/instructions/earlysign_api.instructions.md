---
applyTo: "earlysign/api/**"
---

# 🗣️ EarlySign API Namespace: Ubiquitous Domain Language

## 🎯 Domain-First Design Principle

### Business Domain Language Required
- **Use experiment terminology**: `ab_test`, `experiment`, `guardrails` not technical terms
- **User-facing concepts**: Terms that business stakeholders and data scientists use daily
- **Avoid technical implementation details**: No `multi_metric`, `statistic_collection`, etc.

### Preferred API Modules
- ✅ `earlysign.api.ab_test` - A/B testing experiments
- ❌ `earlysign.api.multi_metric` - Too technical, use domain concepts instead
- ❌ `earlysign.api.statistics` - Implementation detail, not user-facing

### Class Naming Conventions
- **Experiment Types**: `ABTest`, `GuardrailExperiment`, `AdaptiveExperiment`
- **Configuration**: `GuardrailConfig`, `ExperimentConfig`, `TestingStrategy`
- **Results**: `ABTestResult`, `ExperimentOutcome`, `GuardrailSignal`

### Documentation Standards
- **Business context first**: Explain why this experiment pattern matters
- **Use cases**: Real-world scenarios where this applies
- **Decision support**: Help users choose the right approach
- **Avoid jargon**: Technical terms only when necessary for precision

## 🔄 Migration Guidelines

### Moving Technical Components
When technical implementations need to be exposed:
1. **Wrap in domain concepts**: Create business-meaningful interfaces
2. **Hide complexity**: Technical details go in `methods/` or `core/`
3. **Provide examples**: Show real business use cases
4. **Clear naming**: Names that non-statisticians understand

### API Evolution
- **Backward compatibility**: Always provide migration paths
- **Deprecation warnings**: Give clear guidance for transitions
- **Domain alignment**: Regularly review if names match business language
