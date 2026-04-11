ES3 Schema Reference
====================

**ES3 (EarlySign Static Schema)** is the formal specification for all data structures
used throughout the EarlySign framework. It ensures consistency across different
components and provides a language-neutral definition of our data models.

Pydantic-First Architecture
---------------------------

The authoritative source for these schemas is defined in **Pydantic Models**.
These definitions serve as the primary source of truth, ensuring type-safety
and runtime validation within the Python environment.

The source files are located in the ``earlysign/schema/ES3`` directory of the repository.

Core Protocol
-------------

.. literalinclude:: ../../../earlysign/schema/ES3/base.py
   :language: python
   :linenos:
   :caption: base.py

Anytime Valid Inference (AVI)
-----------------------------

.. literalinclude:: ../../../earlysign/builtin/AVI/schema.py
   :language: python
   :linenos:
   :caption: schema.py

YEAST
-----

.. literalinclude:: ../../../earlysign/builtin/YEAST/schema.py
   :language: python
   :linenos:
   :caption: schema.py

Group Sequential (GST)
----------------------

.. literalinclude:: ../../../earlysign/builtin/group_sequential/schema/protocol.py
   :language: python
   :linenos:
   :caption: protocol.py
