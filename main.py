import glob
import math
import numpy as np
import numpy.testing as npt
import os
import pprint
import subprocess
import sys

from Bio import SeqIO
import click
import pyhmmer
from pyhmmer.plan7 import HMM,HMMFile
pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=np.inf)

DEBUG = False

@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="The input temp root dir of sepp that contains all the HMMs")
@click.option("--backbone-alignment", required=True, type=click.Path(exists=True), help="The input backbone alignment")
@click.option("--fragmentary-sequence-file", required=True, type=click.Path(exists=True), help="The input fragmentary sequence file to SEPP")
@click.option("--output-prefix", required=True, type=click.Path(), help="Output prefix")
@click.option("--build", required=False, is_flag=True, help="Whether to run hmmbuild. If yes, the input_dir should contain files of the form input_i.fasta")
@click.option("--debug", required=False, is_flag=True, help="Whether to run in debug mode or not")
def merge_hmms(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix, build, debug):
    if(debug):
        DEBUG = True
    merge_hmms_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix, build)

def custom_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix):
    mappings = create_custom_mappings(input_dir, backbone_alignment)
    build_hmm_profiles(input_dir, mappings, output_prefix)
    bitscores = get_custom_bitscores(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix)
    hmms = get_custom_probabilities(input_dir, backbone_alignment, mappings, bitscores, output_prefix)
    return mappings,bitscores,hmms

def sepp_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix):
    mappings = create_sepp_mappings(input_dir, backbone_alignment)
    bitscores = get_sepp_bitscores(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix)
    hmms = get_sepp_probabilities(input_dir, backbone_alignment, mappings, bitscores, output_prefix)
    return mappings,bitscores,hmms


def merge_hmms_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix, build):
    mappings = None
    bitscores = None
    aligned_sequences_dict = {}
    hmms = None
    if(build):
        mappings,bitscores,hmms = custom_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix)
    else:
        mappings,bitscores,hmms = sepp_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix)

    # print("the mappings are")
    # pp.pprint(mappings)
    # with np.printoptions(suppress=True, linewidth=np.inf):
        # print(mappings)
    # print("bitscores are")
    # pp.pprint(bitscores)
    # with np.printoptions(suppress=True, linewidth=np.inf):
        # print(bitscores)

    for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        fragmentary_sequence = fragmentary_sequence_record.seq
        fragmentary_sequence_id = fragmentary_sequence_record.id
        output_hmm = get_probabilities_helper(input_dir, hmms, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix)
        # print("the output hmm for sequence " + str(fragmentary_sequence_id) + " is")
        # pp.pprint(mappings)
        adjacency_matrix,emission_probabilities,transition_probabilities,alphabet = get_matrices(output_hmm, input_dir, backbone_alignment, output_prefix)
        print("adjacency matrix for sequence " + str(fragmentary_sequence_id) + " is")
        # with np.printoptions(suppress=True, linewidth=np.inf):
            # print(adjacency_matrix)
        # print(adjacency_matrix[len(adjacency_matrix) - 2,:])
        print("emission probabilities for sequence " + str(fragmentary_sequence_id) + " is")
        # with np.printoptions(suppress=True, linewidth=np.inf):
            # print(emission_probabilities)
        print("tranistion probabilities for sequence " + str(fragmentary_sequence_id) + " is")
        # with np.printoptions(suppress=True, linewidth=np.inf):
            # print(transition_probabilities)
        print("starting viterbi")
        aligned_sequences_dict[fragmentary_sequence_record.id] = run_viterbi_log_vectorized(adjacency_matrix, np.log2(emission_probabilities), np.log2(transition_probabilities), alphabet, fragmentary_sequence)
    # pp.pprint(adjacency_matrices_dict)
    # pp.pprint(emission_probabilities_dict)
    # pp.pprint(transition_probabilities_dict)
    # aligned_sequences_dict = align_sequences(backtraced_states_dict, fragmentary_sequence_file)

    print("aligned sequences are")
    # pp.pprint(aligned_sequences_dict)
    with np.printoptions(suppress=True, linewidth=np.inf):
        print(aligned_sequences_dict)
    merged_alignment = get_merged_alignments(aligned_sequences_dict, backbone_alignment)
    print("merged alignment is")
    # pp.pprint(merged_alignment)
    with np.printoptions(suppress=True, linewidth=np.inf):
        print(merged_alignment)
    return merged_alignment

def get_merged_alignments(aligned_sequences_dict, backbone_alignment):
    merged_alignment = {}
    for backbone_sequence_record in SeqIO.parse(backbone_alignment, "fasta"):
        merged_alignment[backbone_sequence_record.id] = str(backbone_sequence_record.seq)

    for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
        merged_alignment[aligned_sequence_id] = ""
        for current_position,current_letter in enumerate(aligned_sequence):
            # print("processing " + current_letter + " at position: " + str(current_position))
            if(current_letter == "-"):
                # deletion site so just advance the backbone alignment
                # print("deletion in transitivity merge")
                merged_alignment[aligned_sequence_id] += current_letter
            elif(current_letter.isupper()):
                # homology
                # print("homology in transitivity merge")
                is_already_insertion_column = False
                for backbone_sequence_id,backbone_sequence in merged_alignment.items():
                    if(current_position == len(backbone_sequence) or backbone_sequence[current_position].islower()):
                        if(current_position < len(backbone_sequence)):
                            # print(backbone_sequence[current_position] + " is lower")
                            is_already_insertion_column = True
                        else:
                            # print("end of sequence")
                            pass
                        break
                if(is_already_insertion_column):
                    for i in range(current_position, len(backbone_sequence)):
                        if(backbone_sequence[i].islower()):
                            merged_alignment[aligned_sequence_id] += "-"
                    # merged_alignment[aligned_sequence_id] += current_letter
                merged_alignment[aligned_sequence_id] += current_letter
            elif(current_letter.islower()):
                # this means it's an insertion site
                # print("insertion in transitivity merge")
                is_already_insertion_column = False
                for backbone_sequence_id,backbone_sequence in merged_alignment.items():
                    if(current_position == len(backbone_sequence) or backbone_sequence[current_position].islower()):
                        if(current_position < len(backbone_sequence)):
                            # print(backbone_sequence[current_position] + " is lower")
                            is_already_insertion_column = True
                        else:
                            pass
                            # print("end of sequence")
                        break
                if(not is_already_insertion_column):
                    # print("creating a new column for insertion")
                    for backbone_sequence_id,backbone_sequence in merged_alignment.items():
                        if(backbone_sequence_id != aligned_sequence_id):
                            merged_alignment[backbone_sequence_id] = backbone_sequence[:current_position] + "-" + backbone_sequence[current_position:]
                merged_alignment[aligned_sequence_id] += current_letter

    alignment_length = None
    for aligned_sequence in merged_alignment:
        if(alignment_length == None):
            alignment_length == len(aligned_sequence)
        else:
            assert alignment_length == len(aligned_sequence)
    return merged_alignment


# def aligned_sequences(backtraced_states_dict, fragmentary_sequence_file):
#     aligned_sequences = {}
#     for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
#         current_aligned_sequence = ""
#         current_fragmentary_sequence = fragmentary_sequence_record.seq
#         current_fragmentary_sequence_id = fragmentary_sequence_record.id
#         backtraced_states = backtraced_states_dict[current_fragmentary_sequence_id]
#         for state_index in backtraced_states:
#             if(state_index == 0):
#                 # start state does not emit anything
#                 pass
#             elif(state_index == 1):
#                 # I0 state emits things
#                 current_aligned_sequence.append(frag)
#             elif(state_index == num_states - 1):
#                 # end state does not emit anything
#                 npt.assert_almost_equal(np.sum(row), 0, decimal=4)
#             elif(is_match(state_index)):
#                 npt.assert_almost_equal(np.sum(row), 1, decimal=4)
#             elif(is_insertion(state_index)):
#                 npt.assert_almost_equal(np.sum(row), 1, decimal=4)
#             elif(is_deletion(state_index)):
#                 npt.assert_almost_equal(np.sum(row), 0, decimal=4)

#         aligned_sequences[current_fragmentary_sequence_id] = current_aligned_sequence
#     return aligned_sequences


def run_viterbi_log_vectorized(adjacency_matrix, emission_probabilities, transition_probabilities, alphabet, fragmentary_sequence):
    backtraced_states_dict = {}
    aligned_sequences_dict = {}

    backtraced_states = []
    aligned_sequence = ""
    current_fragmentary_sequence = fragmentary_sequence
    lookup_table = np.zeros((len(current_fragmentary_sequence) + 1, len(emission_probabilities)))
    lookup_table.fill(np.NINF)
    backtrace_table = np.empty(lookup_table.shape, dtype=object)
    backtrace_table.fill((-3,-3))
    lookup_table[0,0] = 0
    backtrace_table[0,0] = (-1,-1)
    num_states = len(emission_probabilities)
    # [0,j] should be -inf but i'm making everything -inf in the initilazation
    for state_index in range(len(emission_probabilities)):
        if(state_index % 300 == 0):
            print("state index: " + str(state_index))
        for sequence_index in range(len(current_fragmentary_sequence) + 1):
            if(state_index == 0 and sequence_index == 0):
                # this is already handled by the base case
                continue
            if(np.sum(emission_probabilities[state_index]) > np.NINF):
                # it's an emission state
                lookup_add_transition = np.add(lookup_table[sequence_index-1,:state_index+1], transition_probabilities[:state_index+1,state_index])
                max_lookup_add_transition_index = np.argmax(lookup_add_transition)
                max_lookup_add_transition_value = lookup_add_transition[max_lookup_add_transition_index]
                current_emission_probability = emission_probabilities[state_index,alphabet.index(current_fragmentary_sequence[sequence_index - 1])]
                if(sequence_index == 0):
                    # this means emitting an empty sequence which has a zero percent chance
                    current_emission_probability = np.NINF
                backtrace_table[sequence_index,state_index] = (sequence_index - 1, max_lookup_add_transition_index)
                lookup_table[sequence_index,state_index] = current_emission_probability + max_lookup_add_transition_value
                if(DEBUG):
                    for search_state_index in range(state_index + 1):
                        if(adjacency_matrix[search_state_index,state_index] == 0):
                            if(transition_probabilities[search_state_index,state_index] != np.NINF):
                                raise Exception("No edge but transition probability exists")
                            continue
            else:
                if(state_index == 0):
                    backtrace_table[sequence_index,state_index] = (-2,-2)
                    lookup_table[sequence_index,state_index] = np.NINF
                else:
                    lookup_add_transition = np.add(lookup_table[sequence_index,:state_index], transition_probabilities[:state_index,state_index])
                    max_lookup_add_transition_index = np.argmax(lookup_add_transition)
                    max_lookup_add_transition_value = lookup_add_transition[max_lookup_add_transition_index]
                    backtrace_table[sequence_index,state_index] = (sequence_index, max_lookup_add_transition_index)
                    lookup_table[sequence_index,state_index] = max_lookup_add_transition_value
                if(DEBUG):
                    for search_state_index in range(state_index):
                        if(adjacency_matrix[search_state_index,state_index] == 0):
                            if(transition_probabilities[search_state_index,state_index] != np.NINF):
                                raise Exception("No edge but transition probability exists")
                            continue
    if(DEBUG):
        readable_table = np.around(lookup_table, decimals=3)
        for state_index in range(len(emission_probabilities)):
            pp.pprint(readable_table[:,state_index])
            pp.pprint(transition_probabilities[state_index,:])
        pp.ppirint(backtrace_table)
        with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
            print("lookup table")
            print(lookup_table)
        with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
            print("backtrace table")
            print(backtrace_table)
        with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
            print("transition probabilities")
            print(transition_probabilities)
        with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
            print("emission probabilities")
            print(emission_probabilities)

    current_position = (len(current_fragmentary_sequence),len(emission_probabilities) - 1)
    while(current_position != (-1,-1)):
        if(current_position == (-2,-2)):
            raise Exception("-2-2 state")
        if(current_position == (-3,-3)):
            raise Exception("-3-3 state")
        # print("tracing back current positions")
        # pp.pprint(current_position)
        current_sequence_index = current_position[0]
        current_state = current_position[1]

        backtraced_states.append(current_state)
        current_position = backtrace_table[current_position]
        if(np.sum(emission_probabilities[current_state]) > np.NINF):
            # current position is already the previous position here
            previous_state_in_sequence = current_position[1]
            if(DEBUG):
                assert previous_state_in_sequence <= current_state
                assert transition_probabilities[previous_state_in_sequence][current_state] > np.NINF
                assert adjacency_matrix[previous_state_in_sequence][current_state] > 0
            if(adjacency_matrix[previous_state_in_sequence][current_state] == 2 or is_insertion(current_state)):
                # print("insertion in fragment")
                aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].lower()
            elif(adjacency_matrix[previous_state_in_sequence][current_state] == 1):
                aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].upper()
            else:
                raise Exception("Illegal transition")
        elif(current_state != 0 and current_state != len(emission_probabilities) - 1):
            aligned_sequence += "-"

    backtraced_states = backtraced_states[::-1]
    aligned_sequence = aligned_sequence[::-1]
    # pp.pprint(backtraced_states)
    # pp.pprint(aligned_sequence)
    return aligned_sequence


def run_viterbi_log(adjacency_matrix, emission_probabilities, transition_probabilities, alphabet, fragmentary_sequence):
    backtraced_states_dict = {}
    aligned_sequences_dict = {}

    backtraced_states = []
    aligned_sequence = ""
    current_fragmentary_sequence = fragmentary_sequence
    lookup_table = np.zeros((len(current_fragmentary_sequence) + 1, len(emission_probabilities)))
    lookup_table.fill(np.NINF)
    backtrace_table = np.empty(lookup_table.shape, dtype=object)
    backtrace_table.fill((-3,-3))
    lookup_table[0,0] = 0
    backtrace_table[0,0] = (-1,-1)
    # [0,j] should be -inf but i'm making everything -inf in the initilazation
    for state_index in range(len(emission_probabilities)):
        # if(emitted_all_letters):
        #     break
        if(state_index % 300 == 0):
            print("state index: " + str(state_index))
        for sequence_index in range(len(current_fragmentary_sequence) + 1):
            # if(emitted_all_letters):
            #     break
            if(state_index == 0 and sequence_index == 0):
                # this is already handled by the base case
                continue
            if(np.sum(emission_probabilities[state_index]) > np.NINF):
                # it's an emission state
                current_emission_probability = emission_probabilities[state_index,alphabet.index(current_fragmentary_sequence[sequence_index - 1])]
                if(sequence_index == 0):
                    # this means emitting an empty sequence which has a zero percent chance
                    current_emission_probability = np.NINF
                max_value = np.NINF

                for search_state_index in range(state_index + 1):
                    if(adjacency_matrix[search_state_index,state_index] == 0):
                        if(transition_probabilities[search_state_index,state_index] != np.NINF):
                            raise Exception("No edge but transition probability exists")
                        continue
                    else:
                        current_value = lookup_table[sequence_index - 1,search_state_index]
                        if(transition_probabilities[search_state_index,state_index] == np.NINF):
                        # if(transition_probabilities[search_state_index,state_index] == 0):
                            current_value = np.NINF
                        else:
                            current_value += transition_probabilities[search_state_index,state_index]
                        if(current_value > max_value):
                            max_value = current_value
                            backtrace_table[sequence_index,state_index] = (sequence_index - 1,search_state_index)
                lookup_table[sequence_index,state_index] = current_emission_probability + max_value
                # if(sequence_index == len(current_fragmentary_sequence) - 1):
                #     ending_tuple = (sequence_index,state_index)
                #     emitted_all_letters = True
            else:
                # it's not an emission state
                # DEBUG NONE
                max_value = np.NINF
                for search_state_index in range(state_index):
                    if(adjacency_matrix[search_state_index,state_index] == 0):
                        if(transition_probabilities[search_state_index,state_index] != np.NINF):
                            raise Exception("No edge but transition probability exists")
                        continue
                    else:
                        current_value = lookup_table[sequence_index,search_state_index]
                        if(transition_probabilities[search_state_index,state_index] == np.NINF):
                        # if(transition_probabilities[search_state_index,state_index] == 0):
                            current_value = np.NINF
                        else:
                            current_value += transition_probabilities[search_state_index,state_index]
                        if(current_value > max_value):
                            max_value = current_value
                            backtrace_table[sequence_index,state_index] = (sequence_index,search_state_index)
                # DEBUG: there should never be acase where we are in the begin state but have advanced in the sequence
                # it should have taken deletion states
                if(state_index == 0):
                    backtrace_table[sequence_index,state_index] = (-2,-2)
                    lookup_table[sequence_index,state_index] = np.NINF
                else:
                    lookup_table[sequence_index,state_index] = max_value

    # readable_table = np.around(lookup_table, decimals=3)
    # for state_index in range(len(emission_probabilities)):
        # pp.pprint(readable_table[:,state_index])
        # pp.pprint(transition_probabilities[state_index,:])
    # pp.ppirint(backtrace_table)
    # with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
    #     print("lookup table")
    #     print(lookup_table)
    # with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
    #     print("backtrace table")
    #     print(backtrace_table)
    # with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
    #     print("transition probabilities")
    #     print(transition_probabilities)
    # with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
    #     print("emission probabilities")
    #     print(emission_probabilities)

    current_position = (len(current_fragmentary_sequence),len(emission_probabilities) - 1)
    while(current_position != (-1,-1)):
        if(current_position == (-2,-2)):
            raise Exception("-2-2 state")
        if(current_position == (-3,-3)):
            raise Exception("-3-3 state")
        # print("tracing back current positions")
        # pp.pprint(current_position)
        current_sequence_index = current_position[0]
        current_state = current_position[1]

        backtraced_states.append(current_state)
        current_position = backtrace_table[current_position]
        if(np.sum(emission_probabilities[current_state]) > np.NINF):
            # current position is already the previous position here
            previous_state_in_sequence = current_position[1]
            assert previous_state_in_sequence <= current_state
            assert transition_probabilities[previous_state_in_sequence][current_state] > np.NINF
            # print(current_state)
            # print(previous_state_in_sequence)
            assert adjacency_matrix[previous_state_in_sequence][current_state] > 0
            if(adjacency_matrix[previous_state_in_sequence][current_state] == 2 or is_insertion(current_state)):
                # print("insertion in fragment")
                aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].lower()
            elif(adjacency_matrix[previous_state_in_sequence][current_state] == 1):
                aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].upper()
            else:
                raise Exception("Illegal transition")
        elif(current_state != 0 and current_state != len(emission_probabilities) - 1):
            aligned_sequence += "-"

    backtraced_states = backtraced_states[::-1]
    aligned_sequence = aligned_sequence[::-1]
    # pp.pprint(backtraced_states)
    # pp.pprint(aligned_sequence)
    return aligned_sequence

def run_viterbi(adjacency_matrices_dict, emission_probabilities_dict, transition_probabilities_dict, alphabet, fragmentary_sequence_file):
    backtraced_states_dict = {}
    aligned_sequences_dict = {}
    for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        backtraced_states = []
        aligned_sequence = ""
        current_fragmentary_sequence = fragmentary_sequence_record.seq
        current_fragmentary_sequence_id = fragmentary_sequence_record.id
        adjacency_matrix = adjacency_matrices_dict[current_fragmentary_sequence_id]
        emission_probabilities = emission_probabilities_dict[current_fragmentary_sequence_id]
        transition_probabilities = transition_probabilities_dict[current_fragmentary_sequence_id]
        lookup_table = np.zeros((len(current_fragmentary_sequence) + 1, len(emission_probabilities)))
        backtrace_table = np.empty(lookup_table.shape, dtype=object)
        lookup_table[0,0] = 1
        backtrace_table[0,0] = (-1,-1)
        # [0,j] should be 0 but i'm making everything zero in the initilazation
        for state_index in range(len(emission_probabilities)):
            # if(emitted_all_letters):
            #     break
            for sequence_index in range(len(current_fragmentary_sequence) + 1):
                # if(emitted_all_letters):
                #     break
                if(state_index == 0 and sequence_index == 0):
                    # this is already handled by the base case
                    continue
                if(np.sum(emission_probabilities[state_index]) > 0):
                    # it's an emission state
                    current_emission_probability = emission_probabilities[state_index,alphabet.index(current_fragmentary_sequence[sequence_index - 1])]
                    if(sequence_index == 0):
                        # this means emitting an empty sequence which has a zero percent chance
                        current_emission_probability = 0
                    max_value = -1
                    for search_state_index in range(state_index + 1):
                        current_value = lookup_table[sequence_index - 1,search_state_index] * transition_probabilities[search_state_index,state_index]
                        if(current_value > max_value):
                            max_value = current_value
                            backtrace_table[sequence_index,state_index] = (sequence_index - 1,search_state_index)
                    lookup_table[sequence_index,state_index] = current_emission_probability * max_value
                    # if(sequence_index == len(current_fragmentary_sequence) - 1):
                    #     ending_tuple = (sequence_index,state_index)
                    #     emitted_all_letters = True
                else:
                    # it's not an emission state
                    # DEBUG NONE
                    max_value = -1
                    for search_state_index in range(state_index):
                        current_value = lookup_table[sequence_index,search_state_index] * transition_probabilities[search_state_index,state_index]
                        if(current_value > max_value):
                            max_value = current_value
                            backtrace_table[sequence_index,state_index] = (sequence_index,search_state_index)
                    # DEBUG: there should never be acase where we are in the begin state but have advanced in the sequence
                    # it should have taken deletion states
                    if(state_index == 0):
                        backtrace_table[sequence_index,state_index] = (-2,-2)
                        lookup_table[sequence_index,state_index] = 0
                    else:
                        lookup_table[sequence_index,state_index] = max_value

        # readable_table = np.around(lookup_table, decimals=3)
        # for state_index in range(len(emission_probabilities)):
            # pp.pprint(readable_table[:,state_index])
            # pp.pprint(transition_probabilities[state_index,:])
        # pp.pprint(backtrace_table)

        current_position = (len(current_fragmentary_sequence),len(emission_probabilities) - 1)
        while(current_position != (-1,-1)):
            # print("tracing back current positions")
            # pp.pprint(current_position)
            current_sequence_index = current_position[0]
            current_state = current_position[1]

            backtraced_states.append(current_state)
            current_position = backtrace_table[current_position]
            if(np.sum(emission_probabilities[current_state]) > 0):
                # current position is already the previous position here
                previous_state_in_sequence = current_position[1]
                assert previous_state_in_sequence <= current_state
                assert transition_probabilities[previous_state_in_sequence][current_state] > 0
                # print(current_state)
                # print(previous_state_in_sequence)
                assert adjacency_matrix[previous_state_in_sequence][current_state] > 0
                if(adjacency_matrix[previous_state_in_sequence][current_state] == 2 or is_insertion(current_state)):
                    # print("insertion in fragment")
                    aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].lower()
                elif(adjacency_matrix[previous_state_in_sequence][current_state] == 1):
                    aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].upper()
                else:
                    raise Exception("Illegal transition")
            elif(current_state != 0 and current_state != len(emission_probabilities) - 1):
                aligned_sequence += "-"

        backtraced_states = backtraced_states[::-1]
        aligned_sequence = aligned_sequence[::-1]
        # pp.pprint(backtraced_states)
        # pp.pprint(aligned_sequence)
        backtraced_states_dict[current_fragmentary_sequence_id] = backtraced_states
        aligned_sequences_dict[current_fragmentary_sequence_id] = aligned_sequence
    return aligned_sequences_dict


def get_matrices(output_hmm, input_dir, backbone_alignment, output_prefix):
    alphabet = ["A", "C", "G", "T"]
    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_columns = None
    for record in backbone_records:
        total_columns = len(backbone_records[record].seq)
        break
    num_states = (3 * total_columns) + 2 + 1
    num_characters = 4

    # print("output_hmm:")
    # pp.pprint(output_hmm)

    M = np.zeros((num_states, num_states)) # HMM adjacency matrix
    T = np.zeros(M.shape) # transition probability table
    P = np.zeros((num_states, num_characters)) # emission probability table
    current_emission_state_mask = np.zeros(num_states)

    # M[0,1] = 1 # edge from start state to I0
    # M[0,2] = 1 # edge from start state to M1
    # # start state doesn't connect to I1
    # M[0,4] = 1 # edge from start state to D1

    # M[1,1] = 2 # edge from I0 to I0
    # M[1,2] = 1 # edge from I0 to M1

    # for i in range(1, total_columns):
    #     M[get_index("M", i),get_index("M", i + 1)] = 1 # Mi to Mi+1
    #     M[get_index("M", i),get_index("I", i)] = 1 # Mi to Ii
    #     M[get_index("M", i),get_index("D", i + 1)] = 1 # Mi to Di+1

    #     M[get_index("I", i),get_index("M", i + 1)] = 1 # Ii to Mi+1
    #     M[get_index("I", i),get_index("I", i)] = 2 # Ii to Ii

    #     M[get_index("D", i),get_index("M", i + 1)] = 1 # Di to Mi+1
    #     M[get_index("D", i),get_index("D", i + 1)] = 1 # Di to Di+1

    # M[get_index("M", total_columns), num_states - 1] = 1 # Last match state to end state
    # M[get_index("M", total_columns),get_index("I", total_columns)] = 1 # Last match state to last insertion state

    # M[get_index("I", total_columns), get_index("I", total_columns)] = 2 # Last insertion state to last insertion state
    # M[get_index("I", total_columns), num_states - 1] = 1 # Last insertion state to end state

    # M[get_index("D", total_columns), num_states - 1] = 1 # Last deletion state to end state

    for current_column in output_hmm:
        # print("current column: " + str(current_column))
        # if(current_column == 0):
            # continue
        for letter_index,letter in enumerate(alphabet):
            # print(current_column)
            # print(total_columns)
            if(current_column == total_columns + 1):
                # this is the end state which has no emission probabilities or transitions going out
                # print("breaking")
                break
            if(current_column != 0):
                P[get_index("M", current_column), letter_index] = output_hmm[current_column]["match"][letter_index]
            P[get_index("I", current_column), letter_index] = output_hmm[current_column]["insertion"][letter_index]
            # DEBUG: this is the problem
        T[get_index("I", current_column),get_index("I", current_column)] = output_hmm[current_column]["insert_loop"] # Ii to Ii
        M[get_index("I", current_column),get_index("I", current_column)] = 2
        T[get_index("M", current_column),get_index("I", current_column)] = output_hmm[current_column]["self_match_to_insert"]# Mi to Ii
        M[get_index("M", current_column),get_index("I", current_column)] = 1
        for current_transition_destination_column in output_hmm[current_column]["transition"]:
            current_transition_probabilities = output_hmm[current_column]["transition"][current_transition_destination_column]
            # these transitions are always valid
            T[get_index("M", current_column),get_index("M", current_transition_destination_column)] = current_transition_probabilities[0] # Mi to Mi+1
            M[get_index("M", current_column),get_index("M", current_transition_destination_column)] = 1
            T[get_index("I", current_column),get_index("M", current_transition_destination_column)] = current_transition_probabilities[3] # Ii to Mi+1
            M[get_index("I", current_column),get_index("M", current_transition_destination_column)] = 1

            # this transition isn't valid on the 0th column(the column before the first column)  since D0 doesn't exist
            if(current_column != 0):
                T[get_index("D", current_column),get_index("M", current_transition_destination_column)] = current_transition_probabilities[5] # Di to Mi+1
                M[get_index("D", current_column),get_index("M", current_transition_destination_column)] = 1
            # this transition is only valid if it's not going to the end state. End state is techincially a match state in this scheme
            if(current_transition_destination_column != total_columns + 1):
                T[get_index("M", current_column),get_index("D", current_transition_destination_column)] = current_transition_probabilities[2] # Mi to Di+1
                M[get_index("M", current_column),get_index("D", current_transition_destination_column)] = 1
                if(current_column != 0):
                    T[get_index("D", current_column),get_index("D", current_transition_destination_column)] = current_transition_probabilities[6] # Di to Di+1
                    M[get_index("D", current_column),get_index("D", current_transition_destination_column)] = 1

    # print(T)
    # print(M)
    # print(P)
    for row_index,row in enumerate(P):
        # print(row_index)
        # print(row)
        if(row_index == 0):
            # start state does not emit anything
            npt.assert_almost_equal(np.sum(row), 0, decimal=2)
        elif(row_index == 1):
            # I0 state emits things
            npt.assert_almost_equal(np.sum(row), 1, decimal=2)
        elif(row_index == num_states - 1):
            # end state does not emit anything
            npt.assert_almost_equal(np.sum(row), 0, decimal=2)
        elif(is_match(row_index)):
            npt.assert_almost_equal(np.sum(row), 1, decimal=2)
        elif(is_insertion(row_index)):
            npt.assert_almost_equal(np.sum(row), 1, decimal=2)
        elif(is_deletion(row_index)):
            npt.assert_almost_equal(np.sum(row), 0, decimal=2)

    for row_index,row in enumerate(T[:num_states-1,:]):
        # print("rowindex: " + str(row_index))
        # print(row)
        npt.assert_almost_equal(np.sum(row), 1, decimal=2)


    return M,P,T,alphabet

def is_match(state_index):
    return (state_index - 2) % 3 == 0

def is_insertion(state_index):
    return state_index == 1 or (state_index - 2) % 3 == 1

def is_deletion(state_index):
    return (state_index - 2) % 3 == 2

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


def build_hmm_profiles(input_dir, mappings, output_prefix):
    num_hmms = len(mappings)
    for current_hmm_index in range(num_hmms):
        with open(output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.out", "w") as stdout_f:
            with open(output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.err", "w") as stderr_f:
                current_input_file = input_dir + "/input_" + str(current_hmm_index) + ".fasta"
                current_output_file = output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.profile"
                subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.1/hmmbuild", "--cpu", "1", "--dna", "--ere", "0.59", "--symfrac", "0.0", "--informat", "afa", current_output_file, current_input_file], stdout=stdout_f, stderr=stderr_f)

def get_custom_probabilities(input_dir, backbone_alignment, mappings, bitscores, output_prefix):
    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_columns = None
    for record in backbone_records:
        total_columns = len(backbone_records[record].seq)
        break
    num_hmms = len(mappings)
    hmms = {}


    for current_hmm_index in range(num_hmms):
        current_input_file = output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.profile"
        current_hmm = None
        with HMMFile(current_input_file) as hmm_f:
            current_hmm = next(hmm_f)

        match_probabilities = np.asarray(current_hmm.mat)
        insertion_probabilities = np.asarray(current_hmm.ins)
        transition_probabilities = np.asarray(current_hmm.trans)
        hmms[current_hmm_index] = {
            "match": match_probabilities,
            "insertion": insertion_probabilities,
            "transition": transition_probabilities,
        }
    return hmms

def get_sepp_probabilities(input_dir, backbone_alignment, mappings, bitscores, output_prefix):
    num_hmms = len(mappings)
    hmms = {}

    for current_hmm_index in range(num_hmms):
        for current_input_file in glob.glob(input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.model.*"):
            current_hmm =None
            with HMMFile(current_input_file) as hmm_f:
                current_hmm = next(hmm_f)

            match_probabilities = np.asarray(current_hmm.mat)
            insertion_probabilities = np.asarray(current_hmm.ins)
            transition_probabilities = np.asarray(current_hmm.trans)
            hmms[current_hmm_index] = {
                "match": match_probabilities,
                "insertion": insertion_probabilities,
                "transition": transition_probabilities,
            }
    return hmms

def get_probabilities_helper(input_dir, hmms, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix):
    # print("the input hmms are")
    # pp.pprint(hmms)

    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_columns = None
    for record in backbone_records:
        total_columns = len(backbone_records[record].seq)
        break

    cumulative_hmm = {}

    current_fragmentary_sequence = fragmentary_sequence
    current_hmm_bitscores = bitscores[fragmentary_sequence_id]

    output_hmm = {}
    for backbone_state_index in range(total_columns + 1):
        # print("hmm weights sum to: " + str(np.sum(hmm_weights.values())))
        # print(hmm_weights.values())
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

        for current_hmm_index,current_hmm in hmms.items():
            current_hmm_mapping = mappings[current_hmm_index]
            if(backbone_state_index in current_hmm_mapping):
                current_states_probabilities[current_hmm_index] = current_hmm
        # if(backbone_state_index == 2566):
            # print("In state " + str(backbone_state_index) + ", there are " + str(len(current_states_probabilities)) + " hmms that have a match state")
            # pp.pprint(current_states_probabilities.keys())

        for current_hmm_file in current_states_probabilities:
            current_sum = 0.0
            if(current_hmm_file in current_hmm_bitscores):
                for compare_to_hmm_file in current_states_probabilities:
                    if(compare_to_hmm_file in current_hmm_bitscores):
                        current_sum += 2**(float(current_hmm_bitscores[compare_to_hmm_file]) - float(current_hmm_bitscores[current_hmm_file]))
                hmm_weights[current_hmm_file] = 1 / current_sum
            else:
                hmm_weights[current_hmm_file] = 0

        is_hmm_weights_all_zero = True
        for hmm_weight_index in hmm_weights:
            if(hmm_weights[hmm_weight_index] != 0):
                is_hmm_weights_all_zero = False
        if(is_hmm_weights_all_zero):
            for hmm_weight_index in hmm_weights:
                hmm_weights[hmm_weight_index] = 1 / (len(hmm_weights))
        # print("hmm weights for " + str(fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
        # pp.pprint(hmm_weights)
        npt.assert_almost_equal(sum(hmm_weights.values()), 1)

        if(backbone_state_index == 0):
            # this is the begin state
            for current_hmm_file in hmm_weights:
                if(backbone_state_index not in mappings[current_hmm_file]):
                    raise Exception("Every mapping should have the state 0, the begin state")
                current_state_in_hmm = mappings[current_hmm_file][backbone_state_index]
                next_state_in_hmm = current_state_in_hmm + 1
                corresponding_next_backbone_state = None
                for state_index in mappings[current_hmm_file]:
                    if(mappings[current_hmm_file][state_index] == next_state_in_hmm):
                        corresponding_next_backbone_state = state_index
                # if(output_hmm[backbone_state_index]["insertion"] == []):
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
                # print(str(backbone_state_index) + " to " + str(corresponding_next_backbone_state))
        elif(backbone_state_index == total_columns):
            # this has a transition to the end state and is the last full column of states
            for current_hmm_file in hmm_weights:
                if(backbone_state_index not in mappings[current_hmm_file]):
                    continue
                current_state_in_hmm = mappings[current_hmm_file][backbone_state_index]
                # if(output_hmm[backbone_state_index]["match"] == []):
                if(len(output_hmm[backbone_state_index]["match"]) == 0):
                    output_hmm[backbone_state_index]["match"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]
                else:
                    output_hmm[backbone_state_index]["match"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]

                # if(output_hmm[backbone_state_index]["insertion"] == []):
                if(len(output_hmm[backbone_state_index]["insertion"]) == 0):
                    output_hmm[backbone_state_index]["insertion"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                else:
                    output_hmm[backbone_state_index]["insertion"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]

                # print(current_states_probabilities[current_hmm_file]["transition"])
                if(total_columns + 1 not in output_hmm[backbone_state_index]["transition"]):
                    output_hmm[backbone_state_index]["transition"][total_columns + 1] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                else:
                    output_hmm[backbone_state_index]["transition"][total_columns + 1] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                output_hmm[backbone_state_index]["insert_loop"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][4]
                output_hmm[backbone_state_index]["self_match_to_insert"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm][1]

                # print(str(backbone_state_index) + " to " + str(corresponding_next_backbone_state))
                # if(backbone_state_index == 2566):
                #     print("hmmfile: " + str(current_hmm_file))
                #     print("1whole row")
                #     pp.pprint(output_hmm[backbone_state_index]["transition"][total_columns + 1])
                #     print("1currently adding")
                #     pp.pprint(current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm])
                #     print("1with weights")
                #     pp.pprint(hmm_weights[current_hmm_file])
                #     print("1total weights is ")
                #     pp.pprint(hmm_weights)
        else:
            # print("backbone state index: " + str(backbone_state_index))
            for current_hmm_file in hmm_weights:
                # print("current hmm file: " + str(current_hmm_file))
                if(backbone_state_index not in mappings[current_hmm_file]):
                    continue
                current_state_in_hmm = mappings[current_hmm_file][backbone_state_index]
                next_state_in_hmm = current_state_in_hmm + 1
                corresponding_next_backbone_state = None
                for state_index in mappings[current_hmm_file]:
                    if(mappings[current_hmm_file][state_index] == next_state_in_hmm):
                        corresponding_next_backbone_state = state_index
                # if(corresponding_next_backbone_state == None):
                    # corresponding_next_backbone_state = total_columns + 1 # this is the skip to the end state
                # print("current state in hmm: " + str(current_state_in_hmm))
                # print("corresponding (tranisitioning to) next backbone state index: " + str(corresponding_next_backbone_state))

                # if(output_hmm[backbone_state_index]["match"] == []):
                if(len(output_hmm[backbone_state_index]["match"]) == 0):
                    output_hmm[backbone_state_index]["match"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]
                else:
                    output_hmm[backbone_state_index]["match"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]

                # if(output_hmm[backbone_state_index]["insertion"] == []):
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
                # DEBUG
                # if(backbone_state_index == 2566 and corresponding_next_backbone_state == 2567):
                #     print("hmmfile: " + str(current_hmm_file))
                #     print("2566-2567whole row")
                #     pp.pprint(output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state])
                #     print("2566-2567currently adding")
                #     pp.pprint(current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm])
                #     print("2566-2567with weights")
                #     pp.pprint(hmm_weights[current_hmm_file])
                #     print("2566-2567total weights is ")
                #     pp.pprint(hmm_weights)
                # elif(backbone_state_index == 2566):
                #     print("hmmfile: " + str(current_hmm_file))
                #     print("2566-endwhole row")
                #     pp.pprint(output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state])
                #     print("2566-endcurrently adding")
                #     pp.pprint(current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm])
                #     print("2566-endwith weights")
                #     pp.pprint(hmm_weights[current_hmm_file])
                #     print("2566-endtotal weights is ")
                #     pp.pprint(hmm_weights)

                # print(str(backbone_state_index) + " to " + str(corresponding_next_backbone_state))
                # print(-np.log(current_states_probabilities[current_hmm_file]["transition"][1]))
        # print(-np.log(output_hmm[0][]["transition"]))
    return output_hmm

def get_custom_bitscores(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix):
    num_hmms = len(list(glob.glob(input_dir + "/input_*.fasta")))
    for current_hmm_index in range(num_hmms):
        current_input_file = output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.profile"
        with open(output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.out", "w") as stdout_f:
            with open(output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.err", "w") as stderr_f:
                current_search_file = output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.output"
                subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.1/hmmsearch", "--noali", "--cpu", "1", "-o", current_search_file, "-E", "99999999999", "--max", current_input_file,fragmentary_sequence_file], stdout=stdout_f, stderr=stderr_f)
    return get_bitscores_helper(input_dir, num_hmms, backbone_alignment, fragmentary_sequence_file, output_prefix)

def get_sepp_bitscores(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix):
    num_hmms = len(list(glob.glob(input_dir + "/P_*")))
    for current_hmm_index in range(num_hmms):
        current_input_file = list(glob.glob(input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.model.*"))[0]
        with open(output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.out", "w") as stdout_f:
            with open(output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.err", "w") as stderr_f:
                current_search_file = output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.output"
                subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.1/hmmsearch", "--noali", "--cpu", "1", "-o", current_search_file, "-E", "99999999999", "--max", current_input_file, fragmentary_sequence_file], stdout=stdout_f, stderr=stderr_f)

    return get_bitscores_helper(input_dir, num_hmms, backbone_alignment, fragmentary_sequence_file, output_prefix)

def get_bitscores_helper(input_dir, num_hmms, backbone_alignment, fragmentary_sequence_file, output_prefix):
    hmm_bitscores = {}

    for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        current_fragmentary_sequence = fragmentary_sequence_record.seq
        current_hmm_bitscores = {}
        for current_hmm_index in range(num_hmms):
            current_search_file = output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.output"
            with open(current_search_file, "r") as f:
                count_from_sequence_id = 0
                current_sequence_id_encountered = False
                for line in f:
                    if("No") in line:
                        break
                    if(current_sequence_id_encountered):
                        count_from_sequence_id += 1
                    if(count_from_sequence_id == 3):
                        current_hmm_bitscores[current_hmm_index] = line.split()[2]
                        break
                    if(">> " + fragmentary_sequence_record.id in line):
                        current_sequence_id_encountered = True
        hmm_bitscores[fragmentary_sequence_record.id] = current_hmm_bitscores
    return hmm_bitscores

def create_custom_mappings(input_dir, backbone_alignment):
    num_hmms = len(list(glob.glob(input_dir + "/input_*.fasta")))
    input_fasta_filenames = []
    for current_hmm_index in range(num_hmms):
        current_input_file = input_dir + "/input_" + str(current_hmm_index) + ".fasta"
        input_fasta_filenames.append(current_input_file)
    return create_mappings_helper(input_fasta_filenames, backbone_alignment)

def create_sepp_mappings(input_dir, backbone_alignment):
    num_hmms = len(list(glob.glob(input_dir + "/P_*")))
    input_fasta_filenames = []
    for current_hmm_index in range(num_hmms):
        current_input_file = list(glob.glob(input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.input.*.fasta"))[0]
        input_fasta_filenames.append(current_input_file)
    return create_mappings_helper(input_fasta_filenames, backbone_alignment)

def create_mappings_helper(input_fasta_filenames, backbone_alignment):
    num_hmms = len(input_fasta_filenames)
    hmm_weights = {}
    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_columns = None
    for record in backbone_records:
        total_columns = len(backbone_records[record].seq)
        break

    match_state_mappings = {} # from hmm_index to {map of backbone match state index to current input hmm build fasta's match state index}
    for current_hmm_index,current_input_file in enumerate(input_fasta_filenames):
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
            for key_so_far in cumulative_mapping:
                if key_so_far in current_mapping:
                    assert cumulative_mapping[key_so_far] == current_mapping[key_so_far]
            cumulative_mapping.update(current_mapping)
        assert current_hmm_index not in match_state_mappings
        match_state_mappings[current_hmm_index] = cumulative_mapping

    for mapping_index,mapping in match_state_mappings.items():
        # pp.pprint(mapping)
        max_match_state_index = -1
        match_state_sum = 0
        for backbone_index,match_state_index in mapping.items():
            if(match_state_index > max_match_state_index):
                max_match_state_index = match_state_index
            match_state_sum += match_state_index
        assert match_state_sum == (max_match_state_index * (max_match_state_index + 1) / 2)
        # DEBUG TODO: comeback to this
        # DEBUG TODO: second note i think this is correct now
        match_state_mappings[mapping_index][total_columns + 1] = max_match_state_index + 1

    for mapping_index,mapping in match_state_mappings.items():
        mapping[0] = 0

    return match_state_mappings


if __name__ == "__main__":
    merge_hmms()
