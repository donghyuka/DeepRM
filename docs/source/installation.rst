Installation
============

Prerequisites
--------------------------
* Linux (tested on Ubuntu 24.04, RHEL 7 and 8)
* Python >= 3.9 (tested on 3.11)
* CUDA 11.x+ (for GPU inference, tested on 11.8 and 12.6)


Installation Options
--------------------------
1. Install via PIP (recommended)

.. code-block:: bash

   python -m pip install -e deeprm


2. Install via Conda

.. code-block:: bash

   conda -c conda-forge install -e deeprm


3. Install from GitHub

.. code-block:: bash

   git clone https://github.com/vadanamu/deeprm
   cd deeprm
   python -m pip install -e .


Verify Installation
--------------------------

.. code-block:: bash

   deeprm --version