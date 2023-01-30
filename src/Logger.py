from contextlib import closing
import glob
import gc
import math
from multiprocessing import Pool
import os
import pprint
from re import M, T
import subprocess
import sys

from Bio import SeqIO
import click
import numpy as np
import numpy.testing as npt
import pyhmmer
from pyhmmer.plan7 import HMM,HMMFile
from scipy.sparse import lil_matrix

class Logger:
    log_path = None
    file_handle = None
    pp = pprint.PrettyPrinter(indent=4)
    np.set_printoptions(threshold=np.inf)

    def __init__(self, output_prefix, debug_flag, verbose_flag):
        self.log_path = f"{output_prefix}/hmmerge.log"
        self.file_handle = open(self.log_path, "w")
        self.debug_flag = debug_flag
        self.verbose_flag = verbose_flag

    def __del__(self):
        if(self.file_handle):
            self.file_handle.close()

    def info(self, statement):
        if(self.file_handle):
            self.file_handle.write(f"[INFO]: {statement}\n")

    def verbose(self, statement):
        if(self.file_handle and self.verbose_flag):
            self.file_handle.write(f"[DEBUG]: {statement}\n")

    def verbose_pprint(self, to_be_printed):
        if(self.file_handle and self.verbose_flag):
            with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
                pprint.pprint(to_be_printed, self.file_handle)

    def error(self, statement):
        if(self.file_handle):
            self.file_handle.write(f"[ERROR]: {statement}\n")
