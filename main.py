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


DEBUG = False
VERBOSE = False


class Logger:
    log_path = None
    file_handle = None
    pp = pprint.PrettyPrinter(indent=4)
    np.set_printoptions(threshold=np.inf)
    def __init__(self):
        pass

    def initialize(self, output_prefix):
        self.log_path = f"{output_prefix}/hmmerge.log"
        self.file_handle = open(self.log_path, "w")

    def __del__(self):
        self.file_handle.close()

    def info(self, statement):
        self.file_handle.write(f"[INFO]: {statement}\n")

    def verbose(self, statement):
        if(VERBOSE):
            self.file_handle.write(f"[DEBUG]: {statement}\n")

    def verbose_pprint(self, to_be_printed):
        if(VERBOSE):
            with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
                pprint.pprint(to_be_printed, self.file_handle)

    def error(self, statement):
        self.file_handle.write(f"[ERROR]: {statement}\n")

logger = Logger()

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
                logger.error("State D0 dooesn't exist")
                raise Exception("State D0 dooesn't exist")

        if(state_type == "M"):
            return 2 + (3 * (int(column_index) - 1))
        elif(state_type == "I"):
            return 2 + (3 * (int(column_index) - 1)) + 1
        elif(state_type == "D"):
            return 2 + (3 * (int(column_index) - 1)) + 2
        else:
            logger.error(str(state_type) + " is not a valid type")
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
        logger.error("Unknown State Index")
        raise Exception("Unknown State Index")

class Viterbi:
    aligned_result = ""
    backtraced_states = []
    def __init__(self, adjacency_matrix, emission_probabilities, transition_probabilities, alphabet, query_sequence, query_sequence_id):
        self.adjacency_matrix = adjacency_matrix
        self.emission_probabilities = emission_probabilities
        self.transition_probabilities = transition_probabilities
        self.alphabet = alphabet
        self.query_sequence = query_sequence
        self.query_sequence_id = query_sequence_id

    def run(self):
        with np.errstate(divide='ignore'):
            backtraced_states = []
            aligned_sequence = ""
            current_fragmentary_sequence = self.query_sequence
            lookup_table = np.zeros((len(current_fragmentary_sequence) + 1, self.emission_probabilities.shape[0]), dtype="float32")
            lookup_table.fill(np.NINF)
            backtrace_table = np.empty(lookup_table.shape, dtype=object)
            backtrace_table.fill((-3,-3))
            lookup_table[0,0] = 0
            backtrace_table[0,0] = (-1,-1)
            num_states = self.emission_probabilities.shape[0]
            # [0,j] should be -inf but i'm making everything -inf in the initilazation
            for state_index in range(self.emission_probabilities.shape[0]):
                if(state_index % 2500 == 0):
                    logger.info("Viterbi in progress - state index: " + str(state_index) + "/" + str(num_states))
                for sequence_index in range(len(current_fragmentary_sequence) + 1):
                    if(state_index == 0 and sequence_index == 0):
                        # this is already handled by the base case
                        continue
                    if(np.sum(self.emission_probabilities[state_index,:]) > np.NINF):
                        # it's an emission state
                        lookup_add_transition = np.add(lookup_table[sequence_index-1,:state_index+1], self.transition_probabilities[:state_index+1,state_index])
                        max_lookup_add_transition_index = np.argmax(lookup_add_transition)
                        max_lookup_add_transition_value = lookup_add_transition[max_lookup_add_transition_index]
                        if(current_fragmentary_sequence[sequence_index - 1] not in self.alphabet):
                            current_emission_probability = np.log2(1/len(self.alphabet))
                        else:
                            current_emission_probability = self.emission_probabilities[state_index,self.alphabet.index(current_fragmentary_sequence[sequence_index - 1])]
                        if(sequence_index == 0):
                            # this means emitting an empty sequence which has a zero percent chance
                            current_emission_probability = np.NINF
                        backtrace_table[sequence_index,state_index] = (sequence_index - 1, max_lookup_add_transition_index)
                        lookup_table[sequence_index,state_index] = current_emission_probability + max_lookup_add_transition_value
                        if(DEBUG):
                            for search_state_index in range(state_index + 1):
                                if(self.adjacency_matrix[search_state_index,state_index] == 0):
                                    if(self.transition_probabilities[search_state_index,state_index] > np.NINF): # was np.NINF
                                        sys.stderr.write(str(search_state_index))
                                        sys.stderr.write("\n")
                                        sys.stderr.write(str(state_index))
                                        sys.stderr.write("\n")
                                        sys.stderr.write(str(self.transition_probabilities[search_state_index,state_index]))
                                        sys.stderr.write("\n")
                                        sys.stderr.flush()
                                        logger.error("No edge but transition probability exists")
                                        raise Exception("No edge but transition probability exists")
                                    continue
                    else:
                        if(state_index == 0):
                            backtrace_table[sequence_index,state_index] = (-2,-2)
                            lookup_table[sequence_index,state_index] = np.NINF
                        else:
                            lookup_add_transition = np.add(lookup_table[sequence_index,:state_index], self.transition_probabilities[:state_index,state_index])
                            max_lookup_add_transition_index = np.argmax(lookup_add_transition)
                            max_lookup_add_transition_value = lookup_add_transition[max_lookup_add_transition_index]
                            backtrace_table[sequence_index,state_index] = (sequence_index, max_lookup_add_transition_index)
                            lookup_table[sequence_index,state_index] = max_lookup_add_transition_value
                        if(DEBUG):
                            for search_state_index in range(state_index):
                                if(self.adjacency_matrix[search_state_index,state_index] == 0):
                                    if(self.transition_probabilities[search_state_index,state_index] > np.NINF): # was np.NINF
                                        logger.error("No edge but transition probability exists")
                                        raise Exception("No edge but transition probability exists")
                                    continue
            logger.info("Viterbi in progress - state index: " + str(num_states) + "/" + str(num_states))
            logger.verbose("lookup table:")
            logger.verbose_pprint(lookup_table)
            logger.verbose("backtrace table:")
            logger.verbose_pprint(backtrace_table)
            logger.verbose("transition probabilities:")
            logger.verbose_pprint(self.transition_probabilities)
            logger.verbose("emission probabilities:")
            logger.verbose_pprint(self.emission_probabilities)

            current_position = (len(current_fragmentary_sequence),self.emission_probabilities.shape[0] - 1)
            while(current_position != (-1,-1)):
                if(current_position == (-2,-2)):
                    logger.error("-2-2 state")
                    raise Exception("-2-2 state")
                if(current_position == (-3,-3)):
                    logger.error("-3-3 state")
                    raise Exception("-3-3 state")
                current_sequence_index = current_position[0]
                current_state = current_position[1]
                backtraced_states.append(current_state)

                current_position = backtrace_table[current_position]
                if(np.sum(self.emission_probabilities[current_state]) > np.NINF):
                    # current position is already the previous position here
                    previous_state_in_sequence = current_position[1]
                    if(DEBUG):
                        assert previous_state_in_sequence <= current_state
                        assert self.transition_probabilities[previous_state_in_sequence][current_state] > np.NINF
                        assert self.adjacency_matrix[previous_state_in_sequence][current_state] > 0
                    # the if statement is redundant since ==2 would always be an insertion state but not all insertion state is == 2 for instance m to i is not == 2
                    if(self.adjacency_matrix[previous_state_in_sequence][current_state] == 2 or Utils.is_insertion(current_state)):
                        # print("insertion in fragment")
                        aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].lower()
                    elif(self.adjacency_matrix[previous_state_in_sequence][current_state] == 1):
                        aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].upper()
                    else:
                        logger.error("Illegal transition")
                        raise Exception("Illegal transition")
                elif(current_state != 0 and current_state != self.emission_probabilities.shape[0] - 1):
                    aligned_sequence += "-"

            backtraced_states = backtraced_states[::-1]
            aligned_sequence = aligned_sequence[::-1]
            self.aligned_result = aligned_sequence
            self.backtraced_states = backtraced_states

            logger.verbose("backtraced states:")
            logger.verbose_pprint(backtraced_states)

class HMMManipulation:
    output_hmm = None
    sparse_adjacency_matrix = None
    sparse_emission_probabilities = None
    sparse_transition_probabilities = None
    alphabet = None

    def __init__(self, hmms, support_value, input_sequence_files, backbone_alignment, fragmentary_sequence_id, mappings, bitscores, equal_probabilities, model, output_prefix):
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

    def get_probabilities_helper(self):
        logger.verbose("the input hmms are")
        logger.verbose_pprint(self.hmms)
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
                            current_sum += 2**(float(current_hmm_bitscores[compare_to_hmm_file]) - float(current_hmm_bitscores[current_hmm_file]) + np.log2(input_alignment_sizes[compare_to_hmm_file] / input_alignment_sizes[current_hmm_file]))
                    hmm_weights[current_hmm_file] = 1 / current_sum
                else:
                    hmm_weights[current_hmm_file] = 0
            logger.verbose("uncorrected hmm weights for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            logger.verbose_pprint(hmm_weights)

            is_hmm_weights_all_zero = True
            for hmm_weight_index in hmm_weights:
                if(hmm_weights[hmm_weight_index] != 0):
                    is_hmm_weights_all_zero = False
            if(is_hmm_weights_all_zero):
                for hmm_weight_index in hmm_weights:
                    hmm_weights[hmm_weight_index] = 1 / (len(hmm_weights))
            logger.verbose("zero corrected hmm weights for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            logger.verbose_pprint(hmm_weights)
            if(DEBUG):
                npt.assert_almost_equal(sum(hmm_weights.values()), 1)

            hmm_weights_tuple_arr = []
            for hmm_weight_index in hmm_weights:
                hmm_weights_tuple_arr.append((hmm_weight_index, hmm_weights[hmm_weight_index]))
            hmm_weights_tuple_arr.sort(key=lambda x: x[1], reverse=True)
            adjusted_support_value = self.support_value

            logger.verbose("hmm tuple arr for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            logger.verbose_pprint(hmm_weights_tuple_arr)

            if(DEBUG):
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
            logger.verbose("corrected hmm weights for " + str(self.fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            logger.verbose_pprint(hmm_weights)

            for hmm_weight_index in hmm_weights:
                if(hmm_weight_index not in hmm_freq_dict):
                    hmm_freq_dict[hmm_weight_index] = 0
                hmm_freq_dict[hmm_weight_index] += 1

            if(DEBUG):
                npt.assert_almost_equal(sum(hmm_weights.values()), 1)

            if(backbone_state_index == 0):
                # this is the begin state
                for current_hmm_file in hmm_weights:
                    if(backbone_state_index not in self.mappings[current_hmm_file]):
                        logger.error("Every mapping should have the state 0, the begin state")
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
        logger.verbose("hmm_freq_dict")
        logger.verbose_pprint(hmm_freq_dict)
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

        if(DEBUG):
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

        if(DEBUG):
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

        if(DEBUG):
            for row_index,row in enumerate(new_transition_probabilities[:num_states-1,:]):
                npt.assert_almost_equal(np.sum(row), 1, decimal=2)

        self.sparse_adjacency_matrix = new_adjacency_matrix
        self.sparse_transition_probabilities = new_transition_probabilities
        gc.collect()


class HMMIO:
    num_hmms = None
    input_profile_files = None
    input_sequence_files = None
    bitscores = None
    hmms = None
    mappings = None
    def __init__(self, input_type, backbone_alignment, input_dir, model, fragmentary_sequence_file, output_prefix):
        self.input_type = input_type
        self.backbone_alignment = backbone_alignment
        self.input_dir = input_dir
        self.model = model
        self.fragmentary_sequence_file = fragmentary_sequence_file
        self.output_prefix = output_prefix
        self.process()

    def process(self):
        if(self.input_type == "custom"):
            self.custom_helper()
        elif(self.input_type == "sepp"):
            self.sepp_helper()
        elif(self.input_type == "upp"):
            self.upp_helper()
        else:
            logger.errror(f"Unsupported mode: {self.input_type}")
            raise Exception(f"Unsupported mode: {self.input_type}")

        self.mappings = self.create_mappings_helper()
        logger.verbose("mappings")
        logger.verbose_pprint(self.mappings)

        if(self.input_type == "custom"):
            logger.info("type is custom")
            self.build_hmm_profiles()
        self.generic_helper()


    def custom_helper(self):
        num_hmms = len(list(glob.glob(self.input_dir + "/input_*.fasta")))
        input_profile_files = {}
        input_sequence_files = {}
        for current_hmm_index in range(num_hmms):
            input_profile_files[current_hmm_index] = self.output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.profile"
            input_sequence_files[current_hmm_index] = self.input_dir + "/input_" + str(current_hmm_index) + ".fasta"
        self.num_hmms = num_hmms
        self.input_profile_files = input_profile_files
        self.input_sequence_files = input_sequence_files

    def sepp_helper(self):
        num_hmms = len(list(glob.glob(self.input_dir + "/P_*")))
        input_profile_files = {}
        input_sequence_files = {}
        for current_hmm_index in range(num_hmms):
            input_profile_files[current_hmm_index] = list(glob.glob(self.input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.model.*"))[0]
            input_sequence_files[current_hmm_index] = list(glob.glob(self.input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.input.*.fasta"))[0]
        self.num_hmms = num_hmms
        self.input_profile_files = input_profile_files
        self.input_sequence_files = input_sequence_files

    def upp_helper(self):
        num_hmms = len(list(glob.glob(self.input_dir + "/P_0/A_*")))
        input_profile_files = {}
        input_sequence_files = {}
        for current_hmm_index in range(num_hmms):
            input_profile_files[current_hmm_index] = list(glob.glob(self.input_dir + "/P_0/A_0_" + str(current_hmm_index) + "/hmmbuild.model.*"))[0]
            input_sequence_files[current_hmm_index] = list(glob.glob(self.input_dir + "/P_0/A_0_" + str(current_hmm_index) + "/hmmbuild.input.*.fasta"))[0]
        self.num_hmms = num_hmms
        self.input_profile_files = input_profile_files
        self.input_sequence_files = input_sequence_files

    def generic_helper(self):
        self.bitscores = self.get_bitscores_helper()
        self.hmms = self.read_hmms()

    def get_bitscores_helper(self):
        hmm_bitscores = {}
        string_infinity = "9" * 100
        for current_hmm_index,current_input_file in self.input_profile_files.items():
            with open(self.output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.out", "w") as stdout_f:
                with open(self.output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.err", "w") as stderr_f:
                    current_search_file = self.output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.output"
                    subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.2/hmmsearch", "--noali", "--cpu", "1", "-o", current_search_file, "-E", string_infinity, "--domE", string_infinity, "--max", current_input_file,self.fragmentary_sequence_file], stdout=stdout_f, stderr=stderr_f)

        for fragmentary_sequence_record in SeqIO.parse(self.fragmentary_sequence_file, "fasta"):
            current_fragmentary_sequence = fragmentary_sequence_record.seq
            current_hmm_bitscores = {}
            for current_hmm_index in range(self.num_hmms):
                current_search_file = self.output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.output"
                with open(current_search_file, "r") as f:
                    score_section_start = False
                    for line in f:
                        if(score_section_start):
                            current_line_arr = line.split()
                            if(len(current_line_arr) == 9):
                                if(current_line_arr[8] == fragmentary_sequence_record.id):
                                    current_hmm_bitscores[current_hmm_index] = current_line_arr[1]
                                    break
                        if("Scores for complete sequences"):
                            score_section_start = True
            hmm_bitscores[fragmentary_sequence_record.id] = current_hmm_bitscores
        return hmm_bitscores

    def read_hmms(self):
        hmms = {}
        for current_hmm_index,current_input_file in self.input_profile_files.items():
            current_hmm =None
            with HMMFile(current_input_file) as hmm_f:
                current_hmm = next(hmm_f)

            match_probabilities = np.asarray(current_hmm.mat, dtype="float32")
            insertion_probabilities = np.asarray(current_hmm.ins, dtype="float32")
            transition_probabilities = np.asarray(current_hmm.trans, dtype="float32")
            hmms[current_hmm_index] = {
                "match": match_probabilities,
                "insertion": insertion_probabilities,
                "transition": transition_probabilities,
            }
        return hmms

    def create_mappings_helper(self):
        backbone_records = SeqIO.to_dict(SeqIO.parse(self.backbone_alignment, "fasta"))
        total_columns = None
        for record in backbone_records:
            total_columns = len(backbone_records[record].seq)
            break

        match_state_mappings = {} # from hmm_index to {map of backbone match state index to current input hmm build fasta's match state index}
        for current_hmm_index,current_input_file in self.input_sequence_files.items():
            cumulative_mapping = {}
            for current_record in SeqIO.parse(current_input_file, "fasta"):
                current_sequence = current_record.seq
                record_from_backbone = backbone_records[current_record.id]
                backbone_sequence_length = len(record_from_backbone.seq)
                current_sequence_length = len(current_sequence)
                current_mapping = {}
                backbone_index = 0
                for current_sequence_index,current_sequence_character in enumerate(current_sequence):
                    backbone_character = record_from_backbone.seq[backbone_index]
                    if(current_sequence_character == "-"):
                        continue
                    elif(current_sequence_character == backbone_character):
                        current_mapping[backbone_index + 1] = current_sequence_index + 1
                        backbone_index += 1
                    else:
                        while(backbone_character == "-" or current_sequence_character != backbone_character):
                            backbone_index += 1
                            if(backbone_index == backbone_sequence_length):
                                break
                            backbone_character = record_from_backbone.seq[backbone_index]
                        if(current_sequence_character == backbone_character):
                            current_mapping[backbone_index + 1] = current_sequence_index + 1
                            backbone_index += 1
                if(DEBUG):
                    for key_so_far in cumulative_mapping:
                        if key_so_far in current_mapping:
                            assert cumulative_mapping[key_so_far] == current_mapping[key_so_far]
                cumulative_mapping.update(current_mapping)
            if(DEBUG):
                assert current_hmm_index not in match_state_mappings
            match_state_mappings[current_hmm_index] = cumulative_mapping

        for mapping_index,mapping in match_state_mappings.items():
            max_match_state_index = -1
            match_state_sum = 0
            for backbone_index,match_state_index in mapping.items():
                if(match_state_index > max_match_state_index):
                    max_match_state_index = match_state_index
                match_state_sum += match_state_index
            if(DEBUG):
                assert match_state_sum == (max_match_state_index * (max_match_state_index + 1) / 2)
            match_state_mappings[mapping_index][total_columns + 1] = max_match_state_index + 1

        for mapping_index,mapping in match_state_mappings.items():
            mapping[0] = 0
        return match_state_mappings

    def build_hmm_profiles(self):
        num_hmms = len(self.mappings)
        for current_hmm_index in range(num_hmms):
            with open(self.output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.out", "w") as stdout_f:
                with open(self.output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.err", "w") as stderr_f:
                    current_input_file = self.input_dir + "/input_" + str(current_hmm_index) + ".fasta"
                    current_output_file = self.output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.profile"
                    logger.info("calling hmmbuild with output at " + current_output_file)
                    subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.2/hmmbuild", "--cpu", "1", f"--{self.model.lower()}", "--ere", "0.59", "--symfrac", "0.0", "--informat", "afa", current_output_file, current_input_file], stdout=stdout_f, stderr=stderr_f)


class HMMerge():
    aligned_sequences_dict = None
    backtraced_states_dict = None
    def __init__(self, input_dir, backbone_alignment, query_sequence_file, output_prefix, input_type, num_processes, support_value, model, output_format, equal_probabilities):
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

    def run_align_wrapper(self, args):
        return self.run_align(self, *args)

    def run_align(self, hmms, input_sequence_files, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores ):
        hmm_manipulation = HMMManipulation(hmms, self.support_value, input_sequence_files, self.backbone_alignment, fragmentary_sequence_id, mappings, bitscores, self.equal_probabilities, self.model, self.output_prefix)
        hmm_manipulation.get_probabilities_helper()
        hmm_manipulation.calculate_matrices()
        sparse_adjacency_matrix = hmm_manipulation.sparse_adjacency_matrix
        sparse_emission_probabilities = hmm_manipulation.sparse_emission_probabilities
        sparse_transition_probabilities = hmm_manipulation.sparse_transition_probabilities
        alphabet = hmm_manipulation.alphabet
        logger.info("starting viterbi for sequence " + str(fragmentary_sequence_id))
        viterbi_run = Viterbi(np.asarray(sparse_adjacency_matrix.todense()), np.log2(np.asarray(sparse_emission_probabilities.todense())), np.log2(np.asarray(sparse_transition_probabilities.todense())), alphabet, fragmentary_sequence, fragmentary_sequence_id)
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

        hmmio = HMMIO(self.input_type, self.backbone_alignment, self.input_dir, self.model, self.query_sequence_file, self.output_prefix)
        bitscores = hmmio.bitscores
        hmms = hmmio.hmms
        input_sequence_files = hmmio.input_sequence_files
        mappings = hmmio.mappings


        run_align_args = []
        if(self.num_processes > 1):
            for fragmentary_sequence_record in SeqIO.parse(self.fragmentary_sequence_file, "fasta"):
                fragmentary_sequence = fragmentary_sequence_record.seq
                fragmentary_sequence_id = fragmentary_sequence_record.id
                run_align_args.append((self.input_dir, hmms, self.support_value, input_sequence_files, self.backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, self.output_prefix, self.model, self.equal_probabilities))
            aligned_results = None

            with closing(Pool(processes=self.num_processes, maxtasksperchild=1)) as pool:
                aligned_results = pool.imap_unordered(self.run_align_wrapper, run_align_args)

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

            logger.info("merged alignment is written to " + str(self.output_prefix) + "HMMerge.aligned.fasta")
        elif(self.output_format == "A3M"):
            with open(self.output_prefix + "HMMerge.aligned.a3m", "w") as f:
                for merged_aligned_sequence in merged_alignment:
                    f.write(">" + merged_aligned_sequence + "\n")
                    f.write(merged_alignment[merged_aligned_sequence] + "\n")

            logger.info("merged alignment is written to " + str(self.output_prefix) + "HMMerge.aligned.a3m")
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
            if(DEBUG):
                assert current_backtraced_states[0] == ("M",0)
            current_sequence_list = ["-" for _ in range(num_columns)]
            for aligned_sequence_index,(column_type,backbone_column_index) in enumerate(current_backtraced_states[1:len(current_backtraced_states)-1]):
                if(column_type != "I"):
                    current_sequence_list[backbone_column_index - 1] = aligned_sequence[aligned_sequence_index]
            merged_alignment[aligned_sequence_id] = "".join(current_sequence_list)

        if(DEBUG):
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
                        if(DEBUG):
                            assert current_alignment_list[insertion_index_in_backbone] == "-"
                        current_alignment_list[insertion_index_in_backbone] = aligned_sequence[aligned_sequence_index]
                        merged_alignment[aligned_sequence_id] = "".join(current_alignment_list)

        if(DEBUG):
            alignment_length = None
            for aligned_sequence in merged_alignment:
                if(alignment_length == None):
                    alignment_length = len(merged_alignment[aligned_sequence])
                assert alignment_length == len(merged_alignment[aligned_sequence])
        return merged_alignment

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
    global DEBUG
    global VERBOSE
    if(debug):
        DEBUG = True
    if(verbose):
        VERBOSE = True
    logger.initialize(output_prefix)
    hmmerge = HMMerge(input_dir, backbone_alignment, query_sequence_file, output_prefix, input_type, num_processes, support_value, model, output_format, equal_probabilities)
    hmmerge.merge_hmms_helper()


if __name__ == "__main__":
    merge_hmms()
