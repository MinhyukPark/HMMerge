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

class Viterbi:
    aligned_result = ""
    backtraced_states = []
    def __init__(self, adjacency_matrix, emission_probabilities, transition_probabilities, alphabet, query_sequence, query_sequence_id, logger, debug, verbose):
        self.adjacency_matrix = adjacency_matrix
        self.emission_probabilities = emission_probabilities
        self.transition_probabilities = transition_probabilities
        self.alphabet = alphabet
        self.query_sequence = query_sequence
        self.query_sequence_id = query_sequence_id
        self.logger = logger
        self.debug = debug
        self.verbose = verbose

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
                    self.logger.info("Viterbi in progress - state index: " + str(state_index) + "/" + str(num_states))
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
                        if(self.debug):
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
                                        self.logger.error("No edge but transition probability exists")
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
                        if(self.debug):
                            for search_state_index in range(state_index):
                                if(self.adjacency_matrix[search_state_index,state_index] == 0):
                                    if(self.transition_probabilities[search_state_index,state_index] > np.NINF): # was np.NINF
                                        self.logger.error("No edge but transition probability exists")
                                        raise Exception("No edge but transition probability exists")
                                    continue
            self.logger.info("Viterbi in progress - state index: " + str(num_states) + "/" + str(num_states))
            self.logger.verbose("lookup table:")
            self.logger.verbose_pprint(lookup_table)
            self.logger.verbose("backtrace table:")
            self.logger.verbose_pprint(backtrace_table)
            self.logger.verbose("transition probabilities:")
            self.logger.verbose_pprint(self.transition_probabilities)
            self.logger.verbose("emission probabilities:")
            self.logger.verbose_pprint(self.emission_probabilities)

            current_position = (len(current_fragmentary_sequence),self.emission_probabilities.shape[0] - 1)
            while(current_position != (-1,-1)):
                if(current_position == (-2,-2)):
                    self.logger.error("-2-2 state")
                    raise Exception("-2-2 state")
                if(current_position == (-3,-3)):
                    self.logger.error("-3-3 state")
                    raise Exception("-3-3 state")
                current_sequence_index = current_position[0]
                current_state = current_position[1]
                backtraced_states.append(current_state)

                current_position = backtrace_table[current_position]
                if(np.sum(self.emission_probabilities[current_state]) > np.NINF):
                    # current position is already the previous position here
                    previous_state_in_sequence = current_position[1]
                    if(self.debug):
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
                        self.logger.error("Illegal transition")
                        raise Exception("Illegal transition")
                elif(current_state != 0 and current_state != self.emission_probabilities.shape[0] - 1):
                    aligned_sequence += "-"

            backtraced_states = backtraced_states[::-1]
            aligned_sequence = aligned_sequence[::-1]
            self.aligned_result = aligned_sequence
            self.backtraced_states = backtraced_states

            self.logger.verbose("backtraced states:")
            self.logger.verbose_pprint(backtraced_states)

