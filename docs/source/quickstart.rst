Quick‑start
===========

A 10‑line snippet to run DeepRM on a demo file:

.. code-block:: python

   from inference.pileup import run_pileup
   import numpy as np

   demo_npz = "tests/demo_prediction.npz"
   pm6a, dom = run_pileup(demo_npz)
   print(pm6a[:4], dom[:4])