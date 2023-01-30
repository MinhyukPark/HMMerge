from contextlib import closing
import glob
import gc
import math
from pathos.multiprocessing import ProcessingPool as Pool
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

from src.Logger import Logger
from src.HMMerge import HMMerge

@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="The input temp root dir of sepp that contains all the HMMs")
@click.option("--backbone-alignment", required=True, type=click.Path(exists=True), help="The input backbone alignment")
@click.option("--query-sequence-file", required=True, type=click.Path(exists=True), help="The input query sequence file")
@click.option("--output-prefix", required=True, type=click.Path(), help="Output prefix")
@click.option("--input-type", required=False, default="custom", type=click.Choice(["custom", "sepp", "upp"]), help="The type of input")
@click.option("--num-processes", required=False, type=int, default=1, help="Number of Processes")
@click.option("--support-value", required=False, default=1.0, type=click.FloatRange(min=0.0, max=1.0), help="the weigt support of Top HMMs to choose for merge, 1.0 for all HMMs")
@click.option("--equal-probabilities", required=False, type=bool, default=True, help="Whether to have equal enty/exit probabilities")
@click.option("--model", required=True, type=click.Choice(["DNA", "RNA", "amino"]), help="DNA, RNA, or amino acid analysis")
@click.option("--output-format", required=True, default="FASTA", type=click.Choice(["FASTA", "A3M"]), help="FASTA or A3M format for the output alignment")
@click.option("--debug", required=False, is_flag=True, help="Whether to run in debug mode or not")
@click.option("--verbose", required=False, is_flag=True, help="Whether to run in verbose mode or not")
def merge_hmms(input_dir, backbone_alignment, query_sequence_file, output_prefix, input_type, num_processes, support_value, equal_probabilities, model, output_format, debug, verbose):
    logger = Logger(output_prefix, debug, verbose)
    hmmerge = HMMerge(input_dir, backbone_alignment, query_sequence_file, output_prefix, input_type, num_processes, support_value, model, output_format, equal_probabilities, logger, debug, verbose)
    hmmerge.merge_hmms_helper()


if __name__ == "__main__":
    merge_hmms()
