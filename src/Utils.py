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


class Utils:
    @staticmethod
    def is_match(state_index):
        return (state_index - 2) % 3 == 0

    @staticmethod
    def is_insertion(state_index):
        return state_index == 1 or (state_index - 2) % 3 == 1

    @staticmethod
    def is_deletion(state_index):
        return (state_index - 2) % 3 == 2

    @staticmethod
    def get_index(state_type, column_index):
        if(column_index == 0):
            if(state_type == "M"):
                return 0
            elif(state_type == "I"):
                return 1
            elif(state_type == "D"):
                raise Exception("State D0 dooesn't exist")

        if(state_type == "M"):
            return 2 + (3 * (int(column_index) - 1))
        elif(state_type == "I"):
            return 2 + (3 * (int(column_index) - 1)) + 1
        elif(state_type == "D"):
            return 2 + (3 * (int(column_index) - 1)) + 2
        else:
            raise Exception(str(state_type) + " is not a valid type")

    @staticmethod
    def get_column_type_and_index(state_index):
        if(state_index < 2):
            if(state_index == 0):
                return "M",0
            elif(state_index == 1):
                return "I",0
        else:
            adjusted_state_index = state_index - 2
            column_index = int((state_index + 1)/ 3)
            if(adjusted_state_index % 3 == 0):
                return "M",column_index
            elif(adjusted_state_index % 3 == 1):
                return "I",column_index
            elif(adjusted_state_index % 3 == 2):
                return "D",column_index
        raise Exception("Unknown State Index")

