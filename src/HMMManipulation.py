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

from src.Utils import Utils

class HMMManipulation:
    output_hmm = None
    sparse_adjacency_matrix = None
    sparse_emission_probabilities = None
    sparse_transition_probabilities = None
    alphabet = None

    def __init__(self, hmms, support_value, input_sequence_files, backbone_alignment, fragmentary_sequence_id, mappings, bitscores, equal_probabilities, model, output_prefix, logger, debug, verbose):
        self.hmms = hmms
        self.support_value = support_value
        self.input_sequence_files = input_sequence_files
        self.backbone_alignment = backbone_alignment
        self.fragmentary_sequence_id = fragmentary_sequence_id
        self.mappings = mappings
        self.bitscores = bitscores
        self.equal_probabilities = equal_probabilities
        self.model = model
        self.output_prefx = output_prefix
        self.logger = logger
        self.debug = debug
        self.verbose = verbose

    def get_probabilities_helper(self):
        self.logger.verbose("the input hmms are")
        self.logger.verbose_pprint(self.hmms)
        hmm_freq_dict = {}
        input_alignment_sizes = {}
        for current_hmm_index,current_input_alignment in self.input_sequence_files.items():
            input_alignment_sizes[current_hmm_index] = 0
            for _ in SeqIO.parse(current_input_alignment, "fasta"):
                input_alignment_sizes[current_hmm_index] += 1

        backbone_records = SeqIO.to_dict(SeqIO.parse(self.backbone_alignment, "fasta"))
        total_columns = None
        for record in backbone_records:
            total_columns = len(backbone_records[record].seq)
            break
        current_hmm_bitscores = self.bitscores[self.fragmentary_sequence_id]

        output_hmm = {}
        for backbone_state_index in range(total_columns + 1):
            output_hmm[backbone_state_index] = {
                "match": [],
                "insertion": [],
                "insert_loop": 0,
                "self_match_to_insert": 0,
                "transition": {
                },
            }
            current_states_probabilities = {}
            hmm_weights = {}

            for current_hmm_index,current_hmm in self.hmms.items():
                current_hmm_mapping = self.mappings[current_hmm_index]
                if(backbone_state_index in current_hmm_mapping):
                    current_states_probabilities[current_hmm_index] = current_hmm

            # current states probabilities is a map HMM index to the actual hmm
            # current hmm_bitscores contains a map of HMM index to bitscores
            for current_hmm_file in current_states_probabilities:
                current_sum = 0.0
                current_num_sequences = 0
                if(current_hmm_file in current_hmm_bitscores):
                    for compare_to_hmm_file in current_states_probabilities:
                        if(compare_to_hmm_file in current_hmm_bitscores):
                            with np.errstate(divide='ignore'):
                                current_sum += 2**(float(current_hmm_bitscores[compare_to_hmm_file]) - float(current_hmm_bitscores[current_hmm_file]) + np.log2(input_alignment_sizes[compare_to_hmm_file] / input_alignment_sizes[current_hmm_file]))
                    hmm_weights[current_hmm_file] = 1 / current_sum
                else:
                    hmm_weights[current_hmm_file] = 0
            self.logger.verbose("uncorrected hmm weights for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            self.logger.verbose_pprint(hmm_weights)

            is_hmm_weights_all_zero = True
            for hmm_weight_index in hmm_weights:
                if(hmm_weights[hmm_weight_index] != 0):
                    is_hmm_weights_all_zero = False
            if(is_hmm_weights_all_zero):
                for hmm_weight_index in hmm_weights:
                    hmm_weights[hmm_weight_index] = 1 / (len(hmm_weights))
            self.logger.verbose("zero corrected hmm weights for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            self.logger.verbose_pprint(hmm_weights)
            if(self.debug):
                npt.assert_almost_equal(sum(hmm_weights.values()), 1)

            hmm_weights_tuple_arr = []
            for hmm_weight_index in hmm_weights:
                hmm_weights_tuple_arr.append((hmm_weight_index, hmm_weights[hmm_weight_index]))
            hmm_weights_tuple_arr.sort(key=lambda x: x[1], reverse=True)
            adjusted_support_value = self.support_value

            self.logger.verbose("hmm tuple arr for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            self.logger.verbose_pprint(hmm_weights_tuple_arr)

            if(self.debug):
                assert adjusted_support_value >= 0.0
                assert adjusted_support_value <= 1.0

            hmm_weights = {}
            hmm_weight_value_sum = 0.0
            for hmm_weight_index,hmm_weight_value in hmm_weights_tuple_arr:
                hmm_weights[hmm_weight_index] = hmm_weight_value
                hmm_weight_value_sum += hmm_weight_value
                if(hmm_weight_value_sum > adjusted_support_value):
                    break
            for hmm_weight_index in hmm_weights:
                if(hmm_weight_value_sum == 0.0):
                    hmm_weights[hmm_weight_index] = 1 / len(hmm_weights)
                else:
                    hmm_weights[hmm_weight_index] /= hmm_weight_value_sum
            self.logger.verbose("corrected hmm weights for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            self.logger.verbose_pprint(hmm_weights)

            for hmm_weight_index in hmm_weights:
                if(hmm_weight_index not in hmm_freq_dict):
                    hmm_freq_dict[hmm_weight_index] = 0
                hmm_freq_dict[hmm_weight_index] += 1

            if(self.debug):
                npt.assert_almost_equal(sum(hmm_weights.values()), 1)

            if(backbone_state_index == 0):
                # this is the begin state
                for current_hmm_file in hmm_weights:
                    if(backbone_state_index not in self.mappings[current_hmm_file]):
                        self.logger.error("Every mapping should have the state 0, the begin state")
                        raise Exception("Every mapping should have the state 0, the begin state")
                    current_state_in_hmm = self.mappings[current_hmm_file][backbone_state_index]
                    next_state_in_hmm = current_state_in_hmm + 1
                    corresponding_next_backbone_state = None
                    for state_index in self.mappings[current_hmm_file]:
                        if(self.mappings[current_hmm_file][state_index] == next_state_in_hmm):
                            corresponding_next_backbone_state = state_index
                    if(len(output_hmm[backbone_state_index]["insertion"]) == 0):
                        output_hmm[backbone_state_index]["insertion"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["insertion"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                    if(corresponding_next_backbone_state not in output_hmm[backbone_state_index]["transition"]):
                        output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                    output_hmm[backbone_state_index]["insert_loop"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][4]
                    output_hmm[backbone_state_index]["self_match_to_insert"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][1]
            elif(backbone_state_index == total_columns):
                for current_hmm_file in hmm_weights:
                    if(backbone_state_index not in self.mappings[current_hmm_file]):
                        continue
                    current_state_in_hmm = self.mappings[current_hmm_file][backbone_state_index]
                    if(len(output_hmm[backbone_state_index]["match"]) == 0):
                        output_hmm[backbone_state_index]["match"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["match"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]
                    if(len(output_hmm[backbone_state_index]["insertion"]) == 0):
                        output_hmm[backbone_state_index]["insertion"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["insertion"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]

                    if(total_columns + 1 not in output_hmm[backbone_state_index]["transition"]):
                        output_hmm[backbone_state_index]["transition"][total_columns + 1] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["transition"][total_columns + 1] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                    output_hmm[backbone_state_index]["insert_loop"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][4]
                    output_hmm[backbone_state_index]["self_match_to_insert"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][1]
            else:
                for current_hmm_file in hmm_weights:
                    if(backbone_state_index not in self.mappings[current_hmm_file]):
                        continue
                    current_state_in_hmm = self.mappings[current_hmm_file][backbone_state_index]
                    next_state_in_hmm = current_state_in_hmm + 1
                    corresponding_next_backbone_state = None
                    for state_index in self.mappings[current_hmm_file]:
                        if(self.mappings[current_hmm_file][state_index] == next_state_in_hmm):
                            corresponding_next_backbone_state = state_index
                    if(len(output_hmm[backbone_state_index]["match"]) == 0):
                        output_hmm[backbone_state_index]["match"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["match"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]

                    if(len(output_hmm[backbone_state_index]["insertion"]) == 0):
                        output_hmm[backbone_state_index]["insertion"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["insertion"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]

                    if(corresponding_next_backbone_state not in output_hmm[backbone_state_index]["transition"]):
                        output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]

                    output_hmm[backbone_state_index]["insert_loop"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][4]
                    output_hmm[backbone_state_index]["self_match_to_insert"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][1]
        self.logger.verbose("hmm_freq_dict")
        self.logger.verbose_pprint(hmm_freq_dict)
        self.output_hmm = output_hmm
        del self.hmms
        del self.bitscores
        gc.collect()

    def calculate_matrices(self):
        alphabet = None
        if(self.model == "DNA"):
            alphabet = ["A", "C", "G", "T"]
        elif(self.model == "RNA"):
            alphabet = ["A", "C", "G", "U"]
        elif(self.model == "amino"):
            alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

        backbone_records = SeqIO.to_dict(SeqIO.parse(self.backbone_alignment, "fasta"))
        total_columns = None
        for record in backbone_records:
            total_columns = len(backbone_records[record].seq)
            break
        num_states = (3 * total_columns) + 2 + 1
        num_characters = len(alphabet)

        M = lil_matrix((num_states, num_states), dtype=np.int8)
        T = lil_matrix(M.shape, dtype=np.float32) # transition probability table
        P = lil_matrix((num_states, num_characters), dtype=np.float32) # emission probability table

        for current_column in self.output_hmm:
            for letter_index,letter in enumerate(alphabet):
                if(current_column == total_columns + 1):
                    # this is the end state which has no emission probabilities or transitions going out
                    break
                if(current_column != 0):
                    P[Utils.get_index("M", current_column), letter_index] = self.output_hmm[current_column]["match"][letter_index]
                P[Utils.get_index("I", current_column), letter_index] = self.output_hmm[current_column]["insertion"][letter_index]
            T[Utils.get_index("I", current_column),Utils.get_index("I", current_column)] = self.output_hmm[current_column]["insert_loop"] # Ii to Ii
            M[Utils.get_index("I", current_column),Utils.get_index("I", current_column)] = 2
            T[Utils.get_index("M", current_column),Utils.get_index("I", current_column)] = self.output_hmm[current_column]["self_match_to_insert"]# Mi to Ii
            M[Utils.get_index("M", current_column),Utils.get_index("I", current_column)] = 1
            for current_transition_destination_column in self.output_hmm[current_column]["transition"]:
                current_transition_probabilities = self.output_hmm[current_column]["transition"][current_transition_destination_column]
                # these transitions are always valid
                T[Utils.get_index("M", current_column),Utils.get_index("M", current_transition_destination_column)] = current_transition_probabilities[0] # Mi to Mi+1
                M[Utils.get_index("M", current_column),Utils.get_index("M", current_transition_destination_column)] = 1
                T[Utils.get_index("I", current_column),Utils.get_index("M", current_transition_destination_column)] = current_transition_probabilities[3] # Ii to Mi+1
                M[Utils.get_index("I", current_column),Utils.get_index("M", current_transition_destination_column)] = 1

                # this transition isn't valid on the 0th column(the column before the first column)  since D0 doesn't exist
                if(current_column != 0):
                    T[Utils.get_index("D", current_column),Utils.get_index("M", current_transition_destination_column)] = current_transition_probabilities[5] # Di to Mi+1
                    M[Utils.get_index("D", current_column),Utils.get_index("M", current_transition_destination_column)] = 1
                # this transition is only valid if it's not going to the end state. End state is techincially a match state in this scheme
                if(current_transition_destination_column != total_columns + 1):
                    T[Utils.get_index("M", current_column),Utils.get_index("D", current_transition_destination_column)] = current_transition_probabilities[2] # Mi to Di+1
                    M[Utils.get_index("M", current_column),Utils.get_index("D", current_transition_destination_column)] = 1
                    if(current_column != 0):
                        T[Utils.get_index("D", current_column),Utils.get_index("D", current_transition_destination_column)] = current_transition_probabilities[6] # Di to Di+1
                        M[Utils.get_index("D", current_column),Utils.get_index("D", current_transition_destination_column)] = 1

        if(self.debug):
            for row_index,row in enumerate(P):
                if(row_index == 0):
                    # start state does not emit anything
                    npt.assert_almost_equal(np.sum(row), 0, decimal=2)
                elif(row_index == 1):
                    # I0 state emits things
                    npt.assert_almost_equal(np.sum(row), 1, decimal=2)
                elif(row_index == num_states - 1):
                    # end state does not emit anything
                    npt.assert_almost_equal(np.sum(row), 0, decimal=2)
                elif(Utils.is_match(row_index)):
                    npt.assert_almost_equal(np.sum(row), 1, decimal=2)
                elif(Utils.is_insertion(row_index)):
                    npt.assert_almost_equal(np.sum(row), 1, decimal=2)
                elif(Utils.is_deletion(row_index)):
                    npt.assert_almost_equal(np.sum(row), 0, decimal=2)

            for row_index,row in enumerate(T[:num_states-1,:]):
                npt.assert_almost_equal(np.sum(row), 1, decimal=2)

        self.sparse_adjacency_matrix = M
        self.sparse_emission_probabilities = P
        self.sparse_transition_probabilities = T
        self.alphabet = alphabet
        if(self.equal_probabilities):
            self.add_equal_entry_exit_probabilities()

    def add_equal_entry_exit_probabilities(self):
        new_adjacency_matrix = self.sparse_adjacency_matrix.copy()
        new_transition_probabilities = self.sparse_transition_probabilities.copy()

        num_states = self.sparse_adjacency_matrix.shape[0]
        num_match_states = ((num_states - 3) / 3)
        p_total_entry = 0.1
        p_entry = p_total_entry / num_match_states
        p_exit = 0.1

        for current_state in range(1,num_states):
            if(Utils.is_match(current_state)):
                new_adjacency_matrix[0,current_state] = 1
                new_transition_probabilities[0,current_state] = p_entry
            elif(current_state != Utils.get_index("I", 0) and current_state != Utils.get_index("D", 1)):
                new_adjacency_matrix[0,current_state] = 0
                new_transition_probabilities[0,current_state] = 0

        cumulative_sum_begin = 1 - p_total_entry
        old_transition_sum = self.sparse_transition_probabilities[0,Utils.get_index("I", 0)] + self.sparse_transition_probabilities[0,Utils.get_index("D", 1)]
        ratio_i0 = self.sparse_transition_probabilities[0,Utils.get_index("I", 0)] / old_transition_sum
        ratio_d1 = self.sparse_transition_probabilities[0,Utils.get_index("D", 1)] / old_transition_sum

        new_transition_probabilities[0,Utils.get_index("I", 0)] = cumulative_sum_begin * ratio_i0 # begin to I0
        new_transition_probabilities[0,Utils.get_index("D", 1)] = cumulative_sum_begin * ratio_d1 # begin to D1

        if(self.debug):
            for row_index,row in enumerate(new_transition_probabilities[:num_states-1,:]):
                npt.assert_almost_equal(np.sum(row), 1, decimal=2)

        for current_state in range(1,num_states - 1):
            if(not Utils.is_insertion(current_state)):
                if(self.sparse_transition_probabilities[current_state, num_states - 1] < 1):
                    new_adjacency_matrix[current_state,num_states - 1] = 1
                    new_transition_probabilities[current_state,num_states - 1] = p_exit
                    cumulative_sum = 1 - p_exit
                    old_transition_sum = 0.0
                    original_destination_states_set = set()
                    for destination_state in range(current_state + 1, num_states - 1):
                        if(self.sparse_transition_probabilities[current_state,destination_state] > 0):
                            old_transition_sum += self.sparse_transition_probabilities[current_state, destination_state]
                            original_destination_states_set.add(destination_state)
                    for destination_state in original_destination_states_set:
                        new_transition_probabilities[current_state, destination_state] = cumulative_sum * (self.sparse_transition_probabilities[current_state, destination_state] / old_transition_sum)

        if(self.debug):
            for row_index,row in enumerate(new_transition_probabilities[:num_states-1,:]):
                npt.assert_almost_equal(np.sum(row), 1, decimal=2)

        self.sparse_adjacency_matrix = new_adjacency_matrix
        self.sparse_transition_probabilities = new_transition_probabilities
        gc.collect()


