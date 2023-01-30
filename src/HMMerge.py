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


from src.HMMIO import HMMIO
from src.HMMManipulation import HMMManipulation
from src.Viterbi import Viterbi
from src.Utils import Utils

class HMMerge:
    aligned_sequences_dict = None
    backtraced_states_dict = None
    def __init__(self, input_dir, backbone_alignment, query_sequence_file, output_prefix, input_type, num_processes, support_value, model, output_format, equal_probabilities, logger, debug, verbose):
        self.input_dir = input_dir
        self.backbone_alignment = backbone_alignment
        self.query_sequence_file = query_sequence_file
        self.output_prefix = output_prefix
        self.input_type = input_type
        self.num_processes = num_processes
        self.support_value = support_value
        self.model = model
        self.output_format = output_format
        self.equal_probabilities = equal_probabilities
        self.logger = logger
        self.debug = debug
        self.verbose = verbose

    def run_align_wrapper(self, args):
        return self.run_align(*args)

    def run_align(self, hmms, input_sequence_files, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores):
        hmm_manipulation = HMMManipulation(hmms, self.support_value, input_sequence_files, self.backbone_alignment, fragmentary_sequence_id, mappings, bitscores, self.equal_probabilities, self.model, self.output_prefix, self.logger, self.debug, self.verbose)
        hmm_manipulation.get_probabilities_helper()
        hmm_manipulation.calculate_matrices()
        sparse_adjacency_matrix = hmm_manipulation.sparse_adjacency_matrix
        sparse_emission_probabilities = hmm_manipulation.sparse_emission_probabilities
        sparse_transition_probabilities = hmm_manipulation.sparse_transition_probabilities
        alphabet = hmm_manipulation.alphabet
        self.logger.info("starting viterbi for sequence " + str(fragmentary_sequence_id))
        with np.errstate(divide='ignore'):
            viterbi_run = Viterbi(np.asarray(sparse_adjacency_matrix.todense()), np.log2(np.asarray(sparse_emission_probabilities.todense())), np.log2(np.asarray(sparse_transition_probabilities.todense())), alphabet, fragmentary_sequence, fragmentary_sequence_id, self.logger, self.debug, self.verbose)
        viterbi_run.run()
        aligned_sequences = viterbi_run.aligned_result
        backtraced_states = viterbi_run.backtraced_states
        del sparse_emission_probabilities # could be optional now
        del sparse_transition_probabilities
        gc.collect()
        return fragmentary_sequence_id,aligned_sequences,backtraced_states

    def merge_hmms_helper(self):
        aligned_sequences_dict = {}
        backtraced_states_dict = {}

        hmmio = HMMIO(self.input_type, self.backbone_alignment, self.input_dir, self.model, self.query_sequence_file, self.output_prefix, self.logger, self.debug, self.verbose)
        bitscores = hmmio.bitscores
        hmms = hmmio.hmms
        input_sequence_files = hmmio.input_sequence_files
        mappings = hmmio.mappings


        run_align_args = []
        if(self.num_processes > 1):
            for fragmentary_sequence_record in SeqIO.parse(self.query_sequence_file, "fasta"):
                fragmentary_sequence = fragmentary_sequence_record.seq
                fragmentary_sequence_id = fragmentary_sequence_record.id
                run_align_args.append((hmms, input_sequence_files, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores))
            aligned_results = None

            with closing(Pool(processes=self.num_processes, maxtasksperchild=1)) as pool:
                aligned_results = pool.imap(self.run_align_wrapper, run_align_args)

            for aligned_result in aligned_results:
                aligned_sequences_dict[aligned_result[0]] = aligned_result[1]
                backtraced_states_dict[aligned_result[0]] = aligned_result[2]
        else:
            for fragmentary_sequence_record in SeqIO.parse(self.query_sequence_file, "fasta"):
                fragmentary_sequence = fragmentary_sequence_record.seq
                fragmentary_sequence_id = fragmentary_sequence_record.id
                _,aligned_sequences,backtraced_states = self.run_align(hmms, input_sequence_files, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores)
                aligned_sequences_dict[fragmentary_sequence_id] = aligned_sequences
                backtraced_states_dict[fragmentary_sequence_id] = backtraced_states

        merged_alignment = self.get_merged_alignments(aligned_sequences_dict, backtraced_states_dict, self.backbone_alignment, self.output_format)
        # NOTE: top 1
        # merged_alignment = get_merged_alignments_top_1(aligned_sequences_dict, backtraced_states_dict, backbone_alignment, mappings)
        if(self.output_format == "FASTA"):
            with open(self.output_prefix + "HMMerge.aligned.fasta", "w") as f:
                for merged_aligned_sequence in merged_alignment:
                    if(merged_aligned_sequence != "backbone_indices"):
                        f.write(">" + merged_aligned_sequence + "\n")
                        f.write(merged_alignment[merged_aligned_sequence] + "\n")

            self.logger.info("merged alignment is written to " + str(self.output_prefix) + "HMMerge.aligned.fasta")
        elif(self.output_format == "A3M"):
            with open(self.output_prefix + "HMMerge.aligned.a3m", "w") as f:
                for merged_aligned_sequence in merged_alignment:
                    f.write(">" + merged_aligned_sequence + "\n")
                    f.write(merged_alignment[merged_aligned_sequence] + "\n")

            self.logger.info("merged alignment is written to " + str(self.output_prefix) + "HMMerge.aligned.a3m")
        return merged_alignment


    def get_merged_alignments(self, aligned_sequences_dict, backtraced_states_dict, backbone_alignment, output_format):
        if(output_format == "FASTA"):
            return self.get_merged_alignments_fasta(aligned_sequences_dict, backtraced_states_dict, backbone_alignment)
        elif(output_format == "A3M"):
            return self.get_merged_alignments_a3m(aligned_sequences_dict, backbone_alignment)

    def get_merged_alignments_a3m(self, aligned_sequences_dict, backbone_alignment):
        merged_alignment = {}
        num_columns = None
        for backbone_sequence_record in SeqIO.parse(backbone_alignment, "fasta"):
            if(num_columns == None):
                num_columns = len(backbone_sequence_record)
            merged_alignment[backbone_sequence_record.id] = str(backbone_sequence_record.seq)

        for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
            merged_alignment[aligned_sequence_id] = aligned_sequence
        return merged_alignment

    def get_merged_alignments_fasta(self, aligned_sequences_dict, backtraced_states_dict, backbone_alignment):
        merged_alignment = {}
        num_columns = None
        for backbone_sequence_record in SeqIO.parse(backbone_alignment, "fasta"):
            if(num_columns == None):
                num_columns = len(backbone_sequence_record)
            merged_alignment[backbone_sequence_record.id] = str(backbone_sequence_record.seq)
        merged_alignment["backbone_indices"] = list(range(1, num_columns + 1))
        for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
            current_backtraced_states = backtraced_states_dict[aligned_sequence_id]
            current_backtraced_states = list(map(Utils.get_column_type_and_index, current_backtraced_states))
            if(self.debug):
                assert current_backtraced_states[0] == ("M",0)
            current_sequence_list = ["-" for _ in range(num_columns)]
            for aligned_sequence_index,(column_type,backbone_column_index) in enumerate(current_backtraced_states[1:len(current_backtraced_states)-1]):
                if(column_type != "I"):
                    current_sequence_list[backbone_column_index - 1] = aligned_sequence[aligned_sequence_index]
            merged_alignment[aligned_sequence_id] = "".join(current_sequence_list)

        if(self.debug):
            alignment_length = None
            for aligned_sequence in merged_alignment:
                if(alignment_length == None):
                    alignment_length = len(merged_alignment[aligned_sequence])
                    assert alignment_length == len(merged_alignment[aligned_sequence])

        # time to add insertion columns
        for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
            current_backtraced_states = backtraced_states_dict[aligned_sequence_id]
            current_backtraced_states = list(map(Utils.get_column_type_and_index, current_backtraced_states))
            for aligned_sequence_index,(column_type,backbone_column_index) in enumerate(current_backtraced_states[1:len(current_backtraced_states)-1]):
                if(column_type == "I"):
                    if(backbone_column_index == num_columns):
                        merged_alignment["backbone_indices"] = merged_alignment["backbone_indices"] + ["I"]
                        for merged_sequence_id,merged_sequence in merged_alignment.items():
                            if(merged_sequence_id not in ["backbone_indices", aligned_sequence_id]):
                                merged_alignment[merged_sequence_id] = merged_alignment[merged_sequence_id] + "-"
                        merged_alignment[aligned_sequence_id] += aligned_sequence[aligned_sequence_index]
                    else:
                        insertion_index_in_backbone = merged_alignment["backbone_indices"].index(backbone_column_index + 1)
                        merged_alignment["backbone_indices"] = merged_alignment["backbone_indices"][:insertion_index_in_backbone] + ["I"] + merged_alignment["backbone_indices"][insertion_index_in_backbone:]
                        for merged_sequence_id,merged_sequence in merged_alignment.items():
                            if(merged_sequence_id != "backbone_indices"):
                                merged_alignment[merged_sequence_id] = merged_alignment[merged_sequence_id][:insertion_index_in_backbone] + "-" + merged_alignment[merged_sequence_id][insertion_index_in_backbone:]

                        current_alignment_list = list(merged_alignment[aligned_sequence_id])
                        if(self.debug):
                            assert current_alignment_list[insertion_index_in_backbone] == "-"
                        current_alignment_list[insertion_index_in_backbone] = aligned_sequence[aligned_sequence_index]
                        merged_alignment[aligned_sequence_id] = "".join(current_alignment_list)

        if(self.debug):
            alignment_length = None
            for aligned_sequence in merged_alignment:
                if(alignment_length == None):
                    alignment_length = len(merged_alignment[aligned_sequence])
                assert alignment_length == len(merged_alignment[aligned_sequence])
        return merged_alignment
