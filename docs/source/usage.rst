Usage Guide
===========

Workflow
--------

.. mermaid::

   flowchart TD
       A[Raw FAST5] -->|segment_normalize_signal.py| B(Normalised signal)
       B --> C{DeepRM model}
       C -->|PM6A, DOM| D[pileup.py]

CLI reference
-------------

.. code-block:: bash

   python inference/pileup.py -i <predictions_dir> -o results.npz