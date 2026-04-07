ES3 Schema Reference
====================

**ES3 (EarlySign Static Schema)** is the formal specification for all data structures
used throughout the EarlySign framework. It ensures consistency across different
components and provides a language-neutral definition of our data models.

LinkML Source of Truth
----------------------

The authoritative source for EarlySign schemas is defined using **LinkML** (Linked Data Modeling Language). 
These YAML-based definitions serve as the primary source of truth, from which Pydantic models 
are automatically generated using our custom ``RobustEarlySignGenerator``.

For the historical context and the rationale behind switching from TypeSpec to LinkML, 
please refer to :doc:`ADR/ADR_009`.

The schema definitions are located within the ``earlysign/schema/ES3`` and ``earlysign/builtin/*/`` directories.

Core Schemas (ES3)
------------------

.. literalinclude:: ../../../earlysign/schema/ES3/base.yaml
   :language: yaml
   :linenos:
   :caption: base.yaml

.. literalinclude:: ../../../earlysign/schema/ES3/binomial.yaml
   :language: yaml
   :linenos:
   :caption: binomial.yaml

.. literalinclude:: ../../../earlysign/schema/ES3/continuous.yaml
   :language: yaml
   :linenos:
   :caption: continuous.yaml

Statistical Method Schemas
--------------------------

AVI
^^^

.. literalinclude:: ../../../earlysign/builtin/AVI/schema.yaml
   :language: yaml
   :linenos:
   :caption: schema.yaml

YEAST
^^^^^

.. literalinclude:: ../../../earlysign/builtin/YEAST/schema.yaml
   :language: yaml
   :linenos:
   :caption: schema.yaml

Group Sequential (GST)
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../earlysign/builtin/group_sequential/schema.yaml
   :language: yaml
   :linenos:
   :caption: schema.yaml
