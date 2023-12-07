## Compile dataset into petastorm format.
## IDK if numpy memmap is faster than petastorm. Should test.

import argparse
import petastorm
import torch
import os
import sys


