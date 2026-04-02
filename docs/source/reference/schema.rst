ES3 Schema Reference
====================

**ES3 (EarlySign Static Schema)** is the formal specification for all data structures
used throughout the EarlySign framework. It ensures consistency across different
components and provides a language-neutral definition of our data models.

TypeSpec Origin
---------------

The authoritative source for these schemas is defined in **TypeSpec** (formerly ADL).
These definitions serve as the primary source of truth, from which Pydantic models
are automatically generated for the Python implementation.

The source files are located in the ``ES3/schema`` directory of the repository.

Core Manifest
-------------

.. literalinclude:: ../../../ES3/schema/es3_v1.tsp
   :language: typescript
   :linenos:
   :caption: es3_v1.tsp

AVI
---

.. literalinclude:: ../../../earlysign/builtin/AVI/AVI.tsp
   :language: typescript
   :linenos:
   :caption: AVI.tsp

SequentialQuantile
------------------

.. literalinclude:: ../../../earlysign/builtin/AVI/SequentialQuantile.tsp
   :language: typescript
   :linenos:
   :caption: SequentialQuantile.tsp

YEAST
-----

.. literalinclude:: ../../../earlysign/builtin/YEAST/YEAST.tsp
   :language: typescript
   :linenos:
   :caption: YEAST.tsp

GST
---

.. literalinclude:: ../../../earlysign/builtin/group_sequential/GST.tsp
   :language: typescript
   :linenos:
   :caption: GST.tsp
