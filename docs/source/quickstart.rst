Quick‑start
===========

To run DeepRM Inference on a demo file:

.. code-block:: bash

   deeprm preprocess -i <input_file> -o <output_dir>
   deeprm inference -i <output_dir> -o <predictions_dir>
   deeprm pileup -i <predictions_dir> -o results.npz

The demo POD5 file can be found at:

.. code-block:: text
    <deeprm_dir>/example_data/example.pod5

The full demo POD5 files from the human cell lines can be downloaded from ENA (European Nucleotide Archive) using the following accession:
.. code-block:: text
    ERRXXXXXX
