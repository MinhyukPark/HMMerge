from contextlib import closing
import glob
import gc
import math
from multiprocessing import Pool
import os
import pprint
import subprocess
import sys

from Bio import SeqIO
import click
import numpy as np
import numpy.testing as npt
import pyhmmer
from pyhmmer.plan7 import HMM,HMMFile
from scipy.sparse import lil_matrix

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(threshold=np.inf)

DEBUG = False
VERBOSE = False

def run_align_wrapper(args):
    return run_align(*args)

def run_align(input_dir, hmms, support_value, input_sequence_files, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix, model, equal_probabilities):
    output_hmm = get_probabilities_helper(input_dir, hmms, support_value, input_sequence_files, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix)
    del hmms
    del bitscores
    gc.collect()
    # output_hmm = get_probabilities_top_1_helper(input_dir, hmms, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix)
    # print("the output hmm for sequence " + str(fragmentary_sequence_id) + " is")
    # pp.pprint(output_hmm)
    sparse_adjacency_matrix,sparse_emission_probabilities,sparse_transition_probabilities,alphabet = get_matrices(output_hmm, input_dir, backbone_alignment, model, output_prefix)
    if(equal_probabilities):
        sparse_adjacency_matrix,sparse_transition_probabilities = add_equal_entry_exit_probabilities(sparse_adjacency_matrix,sparse_transition_probabilities)
    # print("adjacency matrix for sequence " + str(fragmentary_sequence_id) + " is")
    # with np.printoptions(suppress=True, linewidth=np.inf):
        # print(adjacency_matrix)
    # print(adjacency_matrix[len(adjacency_matrix) - 2,:])
    # print("emission probabilities for sequence " + str(fragmentary_sequence_id) + " is")
    # with np.printoptions(suppress=True, linewidth=np.inf):
        # print(emission_probabilities)
    # print("tranistion probabilities for sequence " + str(fragmentary_sequence_id) + " is")
    # with np.printoptions(suppress=True, linewidth=np.inf):
        # print(transition_probabilities)
    print("starting viterbi for sequence " + str(fragmentary_sequence_id))
    with np.errstate(divide='ignore'):
        aligned_sequences,backtraced_states = run_viterbi_log_vectorized(np.asarray(sparse_adjacency_matrix.todense()), np.log2(np.asarray(sparse_emission_probabilities.todense())), np.log2(np.asarray(sparse_transition_probabilities.todense())), alphabet, fragmentary_sequence)
        del sparse_emission_probabilities
        del sparse_transition_probabilities
        gc.collect()
        return fragmentary_sequence_id,aligned_sequences,backtraced_states

@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="The input temp root dir of sepp that contains all the HMMs")
@click.option("--backbone-alignment", required=True, type=click.Path(exists=True), help="The input backbone alignment")
@click.option("--fragmentary-sequence-file", required=True, type=click.Path(exists=True), help="The input fragmentary sequence file to SEPP")
@click.option("--output-prefix", required=True, type=click.Path(), help="Output prefix")
@click.option("--input-type", required=False, default="custom", type=click.Choice(["custom", "sepp", "upp"]), help="The type of input")
@click.option("--num-processes", required=False, type=int, default=1, help="Number of Processes")
@click.option("--support-value", required=False, default=1.0, type=click.FloatRange(min=0.0, max=1.0), help="the weigt support of Top HMMs to choose for merge, 1.0 for all HMMs")
@click.option("--equal-probabilities", required=False, type=bool, default=True, help="Whether to have equal enty/exit probabilities")
@click.option("--model", required=True, type=click.Choice(["DNA", "RNA", "amino"]), help="DNA, RNA, or amino acid analysis")
@click.option("--output-format", required=True, default="FASTA", type=click.Choice(["FASTA", "A3M"]), help="FASTA or A3M format for the output alignment")
@click.option("--debug", required=False, is_flag=True, help="Whether to run in debug mode or not")
@click.option("--verbose", required=False, is_flag=True, help="Whether to run in verbose mode or not")
def merge_hmms(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix, input_type, num_processes, support_value, equal_probabilities, model, output_format, debug, verbose):
    global DEBUG
    global VERBOSE
    if(debug):
        DEBUG = True
    if(verbose):
        VERBOSE = True
    merge_hmms_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix, input_type, num_processes, support_value, model, output_format, equal_probabilities)

def custom_helper(input_dir, output_prefix):
    num_hmms = len(list(glob.glob(input_dir + "/input_*.fasta")))
    input_profile_files = {}
    input_sequence_files = {}
    for current_hmm_index in range(num_hmms):
        input_profile_files[current_hmm_index] = output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.profile"
        input_sequence_files[current_hmm_index] = input_dir + "/input_" + str(current_hmm_index) + ".fasta"
    return num_hmms,input_profile_files,input_sequence_files

def sepp_helper(input_dir):
    num_hmms = len(list(glob.glob(input_dir + "/P_*")))
    input_profile_files = {}
    input_sequence_files = {}
    for current_hmm_index in range(num_hmms):
        input_profile_files[current_hmm_index] = list(glob.glob(input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.model.*"))[0]
        input_sequence_files[current_hmm_index] = list(glob.glob(input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.input.*.fasta"))[0]
    return num_hmms,input_profile_files,input_sequence_files

def upp_helper(input_dir):
    num_hmms = len(list(glob.glob(input_dir + "/P_0/A_*")))
    input_profile_files = {}
    input_sequence_files = {}
    for current_hmm_index in range(num_hmms):
        input_profile_files[current_hmm_index] = list(glob.glob(input_dir + "/P_0/A_0_" + str(current_hmm_index) + "/hmmbuild.model.*"))[0]
        input_sequence_files[current_hmm_index] = list(glob.glob(input_dir + "/P_0/A_0_" + str(current_hmm_index) + "/hmmbuild.input.*.fasta"))[0]
    return num_hmms,input_profile_files,input_sequence_files

def generic_helper(input_dir, num_hmms, input_profile_files, input_sequence_files, backbone_alignment, fragmentary_sequence_file, mappings, output_prefix):
    bitscores = get_bitscores_helper(input_dir, num_hmms, input_profile_files, backbone_alignment, fragmentary_sequence_file, output_prefix)
    hmms = read_hmms(input_profile_files)
    return bitscores,hmms

def merge_hmms_helper(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix, input_type, num_processes, support_value, model, output_format, equal_probabilities):
    mappings = None
    bitscores = None
    aligned_sequences_dict = {}
    backtraced_states_dict = {}
    hmms = None
    num_hmms = None
    input_profile_files = None
    input_sequence_files = None

    if(input_type == "custom"):
        num_hmms,input_profile_files,input_sequence_files = custom_helper(input_dir, output_prefix)
    elif(input_type == "sepp"):
        num_hmms,input_profile_files,input_sequence_files = sepp_helper(input_dir)
    elif(input_type == "upp"):
        num_hmms,input_profile_files,input_sequence_files = upp_helper(input_dir)
    else:
        print(input_type)
        raise Exception("Unsupported mode")
    mappings = create_mappings_helper(input_sequence_files, backbone_alignment)
    if(VERBOSE):
        print("mappings")
        pp.pprint(mappings)

    if(input_type == "custom"):
        print("type is custom")
        build_hmm_profiles(input_dir, mappings, model, output_prefix)

    bitscores,hmms = generic_helper(input_dir, num_hmms, input_profile_files, input_sequence_files, backbone_alignment, fragmentary_sequence_file, mappings, output_prefix)

    run_align_args = []
    if(num_processes > 1):
        for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
            fragmentary_sequence = fragmentary_sequence_record.seq
            fragmentary_sequence_id = fragmentary_sequence_record.id
            run_align_args.append((input_dir, hmms, support_value, input_sequence_files, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix, model, equal_probabilities))
        aligned_results = None

        with closing(Pool(processes=num_processes, maxtasksperchild=1)) as pool:
            aligned_results = pool.imap_unordered(run_align_wrapper, run_align_args)

        for aligned_result in aligned_results:
            aligned_sequences_dict[aligned_result[0]] = aligned_result[1]
            backtraced_states_dict[aligned_result[0]] = aligned_result[2]
    else:
        for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
            fragmentary_sequence = fragmentary_sequence_record.seq
            fragmentary_sequence_id = fragmentary_sequence_record.id
            _,aligned_sequences,backtraced_states = run_align(input_dir, hmms, support_value, input_sequence_files, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix, model, equal_probabilities)
            aligned_sequences_dict[fragmentary_sequence_id] = aligned_sequences
            backtraced_states_dict[fragmentary_sequence_id] = backtraced_states



    # print("viterbi output sequences are")
    # pp.pprint(aligned_sequences_dict)
    # with np.printoptions(suppress=True, linewidth=np.inf):
        # print(aligned_sequences_dict)
    merged_alignment = get_merged_alignments(aligned_sequences_dict, backtraced_states_dict, backbone_alignment, output_format)
    # NOTE: top 1
    # merged_alignment = get_merged_alignments_top_1(aligned_sequences_dict, backtraced_states_dict, backbone_alignment, mappings)
    if(output_format == "FASTA"):
        with open(output_prefix + "HMMerge.aligned.fasta", "w") as f:
            for merged_aligned_sequence in merged_alignment:
                if(merged_aligned_sequence != "backbone_indices"):
                    f.write(">" + merged_aligned_sequence + "\n")
                    f.write(merged_alignment[merged_aligned_sequence] + "\n")

        print("merged alignment is written to " + str(output_prefix) + "HMMerge.aligned.fasta")
    elif(output_format == "A3M"):
        with open(output_prefix + "HMMerge.aligned.a3m", "w") as f:
            for merged_aligned_sequence in merged_alignment:
                f.write(">" + merged_aligned_sequence + "\n")
                f.write(merged_alignment[merged_aligned_sequence] + "\n")

        print("merged alignment is written to " + str(output_prefix) + "HMMerge.aligned.a3m")
    # pp.pprint(merged_alignment)
    # with np.printoptions(suppress=True, linewidth=np.inf):
        # print(merged_alignment)

    return merged_alignment

def get_merged_alignments_top_1(aligned_sequences_dict, backtraced_states_dict, backbone_alignment, mapping):
    merged_alignment = {}
    num_columns = None
    for backbone_sequence_record in SeqIO.parse(backbone_alignment, "fasta"):
        if(num_columns == None):
            num_columns = len(backbone_sequence_record)
        merged_alignment[backbone_sequence_record.id] = str(backbone_sequence_record.seq)
    merged_alignment["backbone_indices"] = list(range(1, num_columns + 1))

    for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
        current_backtraced_states = backtraced_states_dict[aligned_sequence_id]
        current_sequence_list = ["-" for _ in range(num_columns)]
        # print("fragmentary sequence aligned length is :" + str(len(aligned_sequence)))
        for aligned_sequence_index,fragmentary_state_index in current_backtraced_states[1:len(current_backtraced_states)-1]:
            if(not is_insertion(backbone_column_index)):
                # print("adding " + str(aligned_sequence[aligned_sequence_index]) + " at position " + str(backbone_column_index - 1))
                backbone_sequence_index = None
                for mapping_backbone_index in mapping:
                    if(mapping[mapping_backbone_index] == aligned_sequence_index):
                        backbone_sequence_index = mapping_backbone_index
                        break
                current_sequence_list[backbone_sequnece_index - 1] = aligned_sequence[aligned_sequence_index]
        merged_alignment[aligned_sequence_id] = "".join(current_sequence_list)

    for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
        current_backtraced_states = backtraced_states_dict[aligned_sequence_id]
        # print("aligning " + str(aligned_sequence_id))
        for aligned_sequence_index,fragmentary_state_index in current_backtraced_states[1:len(current_backtraced_states)-1]:
            if(is_insertion(fragmentary_state_index)):
                backbone_sequence_index = None
                for mapping_backbone_index in mapping:
                    if(mapping[mapping_backbone_index] == aligned_sequence_index):
                        backbone_sequence_index = mapping_backbone_index
                        break
                if(backbone_sequence_index == len(aligned_sequence)):
                    # print("making a new insertion column")
                    merged_alignment["backbone_indices"] = merged_alignment["backbone_indices"] + ["I"]
                    for merged_sequence_id,merged_sequence in merged_alignment.items():
                        if(merged_sequence_id not in ["backbone_indices", aligned_sequence_id]):
                            merged_alignment[merged_sequence_id] = merged_alignment[merged_sequence_id] + "-"
                    merged_alignment[aligned_sequence_id] += aligned_sequence[aligned_sequence_index]
                else:
                    insertion_index_in_backbone = merged_alignment["backbone_indices"].index(backbone_sequence_index + 1)
                    merged_alignment["backbone_indices"] = merged_alignment["backbone_indices"][:insertion_index_in_backbone] + ["I"] + merged_alignment["backbone_indices"][insertion_index_in_backbone:]
                    for merged_sequence_id,merged_sequence in merged_alignment.items():
                        if(merged_sequence_id != "backbone_indices"):
                            merged_alignment[merged_sequence_id] = merged_alignment[merged_sequence_id][:insertion_index_in_backbone] + "-" + merged_alignment[merged_sequence_id][insertion_index_in_backbone:]

                    current_alignment_list = list(merged_alignment[aligned_sequence_id])
                    if(DEBUG):
                        assert current_alignment_list[insertion_index_in_backbone] == "-"
                    current_alignment_list[insertion_index_in_backbone] = aligned_sequence[aligned_sequence_index]
                    merged_alignment[aligned_sequence_id] = "".join(current_alignment_list)
    return merged_alignment

def get_merged_alignments(aligned_sequences_dict, backtraced_states_dict, backbone_alignment, output_format):
    if(output_format == "FASTA"):
        return get_merged_alignments_fasta(aligned_sequences_dict, backtraced_states_dict, backbone_alignment)
    elif(output_format == "A3M"):
        return get_merged_alignments_a3m(aligned_sequences_dict, backtraced_states_dict, backbone_alignment)

def get_merged_alignments_a3m(aligned_sequences_dict, backtraced_states_dict, backbone_alignment):
    merged_alignment = {}
    num_columns = None
    for backbone_sequence_record in SeqIO.parse(backbone_alignment, "fasta"):
        if(num_columns == None):
            num_columns = len(backbone_sequence_record)
        merged_alignment[backbone_sequence_record.id] = str(backbone_sequence_record.seq)

    for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
        merged_alignment[aligned_sequence_id] = aligned_sequence
    return merged_alignment

def get_merged_alignments_fasta(aligned_sequences_dict, backtraced_states_dict, backbone_alignment):
    merged_alignment = {}
    num_columns = None
    for backbone_sequence_record in SeqIO.parse(backbone_alignment, "fasta"):
        if(num_columns == None):
            num_columns = len(backbone_sequence_record)
        merged_alignment[backbone_sequence_record.id] = str(backbone_sequence_record.seq)
    merged_alignment["backbone_indices"] = list(range(1, num_columns + 1))
    for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
        current_backtraced_states = backtraced_states_dict[aligned_sequence_id]
        # print(current_backtraced_states)
        current_backtraced_states = list(map(get_column_type_and_index, current_backtraced_states))
        # print(current_backtraced_states)
        if(DEBUG):
            assert current_backtraced_states[0] == ("M",0)
        current_sequence_list = ["-" for _ in range(num_columns)]
        # print("fragmentary sequence aligned length is :" + str(len(aligned_sequence)))
        for aligned_sequence_index,(column_type,backbone_column_index) in enumerate(current_backtraced_states[1:len(current_backtraced_states)-1]):
            if(column_type != "I"):
                # print(backbone_column_index)
                # print(current_sequence_list)
                # print("adding " + str(aligned_sequence[aligned_sequence_index]) + " at position " + str(backbone_column_index - 1))
                current_sequence_list[backbone_column_index - 1] = aligned_sequence[aligned_sequence_index]
        merged_alignment[aligned_sequence_id] = "".join(current_sequence_list)
    # pp.pprint(merged_alignment)

    if(DEBUG):
        alignment_length = None
        for aligned_sequence in merged_alignment:
            if(alignment_length == None):
                alignment_length = len(merged_alignment[aligned_sequence])
                # print("full: " + str(alignment_length))
            # print(aligned_sequence)
            # print(merged_alignment[aligned_sequence])
            # print(len(merged_alignment[aligned_sequence]))
                assert alignment_length == len(merged_alignment[aligned_sequence])

    # time to add insertion columns
    for aligned_sequence_id,aligned_sequence in aligned_sequences_dict.items():
        current_backtraced_states = backtraced_states_dict[aligned_sequence_id]
        current_backtraced_states = list(map(get_column_type_and_index, current_backtraced_states))
        # print("aligning " + str(aligned_sequence_id))
        for aligned_sequence_index,(column_type,backbone_column_index) in enumerate(current_backtraced_states[1:len(current_backtraced_states)-1]):
            if(column_type == "I"):
                if(backbone_column_index == num_columns):
                    # print("making a new insertion column")
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

    # pp.pprint(merged_alignment)
    if(DEBUG):
        alignment_length = None
        for aligned_sequence in merged_alignment:
            if(alignment_length == None):
                alignment_length = len(merged_alignment[aligned_sequence])
                # print("full: " + str(alignment_length))
            # print(aligned_sequence)
            # print(merged_alignment[aligned_sequence])
            # print(len(merged_alignment[aligned_sequence]))
            assert alignment_length == len(merged_alignment[aligned_sequence])
    return merged_alignment


def run_viterbi_log_vectorized(adjacency_matrix, emission_probabilities, transition_probabilities, alphabet, fragmentary_sequence):
    backtraced_states_dict = {}
    aligned_sequences_dict = {}

    backtraced_states = []
    aligned_sequence = ""
    current_fragmentary_sequence = fragmentary_sequence
    lookup_table = np.zeros((len(current_fragmentary_sequence) + 1, emission_probabilities.shape[0]), dtype="float32")
    lookup_table.fill(np.NINF)
    backtrace_table = np.empty(lookup_table.shape, dtype=object)
    backtrace_table.fill((-3,-3))
    lookup_table[0,0] = 0
    backtrace_table[0,0] = (-1,-1)
    num_states = emission_probabilities.shape[0]
    # [0,j] should be -inf but i'm making everything -inf in the initilazation
    for state_index in range(emission_probabilities.shape[0]):
        if(state_index % 2500 == 0):
            print("Viterbi in progress - state index: " + str(state_index) + "/" + str(num_states))
        for sequence_index in range(len(current_fragmentary_sequence) + 1):
            if(state_index == 0 and sequence_index == 0):
                # this is already handled by the base case
                continue
            if(np.sum(emission_probabilities[state_index,:]) > np.NINF):
                # it's an emission state
                lookup_add_transition = np.add(lookup_table[sequence_index-1,:state_index+1], transition_probabilities[:state_index+1,state_index])
                max_lookup_add_transition_index = np.argmax(lookup_add_transition)
                max_lookup_add_transition_value = lookup_add_transition[max_lookup_add_transition_index]
                if(current_fragmentary_sequence[sequence_index - 1] not in alphabet):
                    current_emission_probability = np.log2(1/len(alphabet))
                else:
                    current_emission_probability = emission_probabilities[state_index,alphabet.index(current_fragmentary_sequence[sequence_index - 1])]
                if(sequence_index == 0):
                    # this means emitting an empty sequence which has a zero percent chance
                    current_emission_probability = np.NINF
                backtrace_table[sequence_index,state_index] = (sequence_index - 1, max_lookup_add_transition_index)
                lookup_table[sequence_index,state_index] = current_emission_probability + max_lookup_add_transition_value
                if(DEBUG):
                    for search_state_index in range(state_index + 1):
                        if(adjacency_matrix[search_state_index,state_index] == 0):
                            if(transition_probabilities[search_state_index,state_index] > np.NINF): # was np.NINF
                                sys.stderr.write(str(search_state_index))
                                sys.stderr.write("\n")
                                sys.stderr.write(str(state_index))
                                sys.stderr.write("\n")
                                sys.stderr.write(str(transition_probabilities[search_state_index,state_index]))
                                sys.stderr.write("\n")
                                sys.stderr.flush()
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
                            if(transition_probabilities[search_state_index,state_index] > np.NINF): # was np.NINF
                                raise Exception("No edge but transition probability exists")
                            continue
    print("Viterbi in progress - state index: " + str(num_states) + "/" + str(num_states))
    if(VERBOSE):
        # note: debug with print
        readable_table = np.around(lookup_table, decimals=3)
        for state_index in range(emission_probabilities.shape[0]):
            pp.pprint(readable_table[:,state_index])
            pp.pprint(transition_probabilities[state_index,:])
        pp.pprint(backtrace_table)
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

    current_position = (len(current_fragmentary_sequence),emission_probabilities.shape[0] - 1)
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
            # the if statement is redundant since ==2 would always be an insertion state but not all insertion state is == 2 for instance m to i is not == 2
            if(adjacency_matrix[previous_state_in_sequence][current_state] == 2 or is_insertion(current_state)):
                # print("insertion in fragment")
                aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].lower()
            elif(adjacency_matrix[previous_state_in_sequence][current_state] == 1):
                aligned_sequence += current_fragmentary_sequence[current_sequence_index - 1].upper()
            else:
                sys.stderr.write(str(previous_state_in_sequence))
                sys.stderr.write("\n")
                sys.stderr.write(str(current_state))
                sys.stderr.write("\n")
                sys.stderr.write(str(adjacency_matrix[previous_state_in_sequence][current_state]))
                sys.stderr.write("\n")
                sys.stderr.flush()
                sys.stdout.flush()
                raise Exception("Illegal transition")
        elif(current_state != 0 and current_state != emission_probabilities.shape[0] - 1):
            aligned_sequence += "-"

    backtraced_states = backtraced_states[::-1]
    aligned_sequence = aligned_sequence[::-1]
    # pp.pprint(backtraced_states)
    # pp.pprint(aligned_sequence)
    return aligned_sequence,backtraced_states

def add_equal_entry_exit_probabilities(adjacency_matrix, transition_probabilities):
    new_adjacency_matrix = adjacency_matrix.copy()
    new_transition_probabilities = transition_probabilities.copy()

    num_states = adjacency_matrix.shape[0]
    num_match_states = ((num_states - 3) / 3)
    p_total_entry = 0.1
    p_entry = p_total_entry / num_match_states
    p_exit = 0.1

    for current_state in range(1,num_states):
        if(is_match(current_state)):
            new_adjacency_matrix[0,current_state] = 1
            new_transition_probabilities[0,current_state] = p_entry
        elif(current_state != get_index("I", 0) and current_state != get_index("D", 1)):
            new_adjacency_matrix[0,current_state] = 0
            new_transition_probabilities[0,current_state] = 0

    cumulative_sum_begin = 1 - p_total_entry
    old_transition_sum = transition_probabilities[0,get_index("I", 0)] + transition_probabilities[0,get_index("D", 1)]
    ratio_i0 = transition_probabilities[0,get_index("I", 0)] / old_transition_sum
    ratio_d1 = transition_probabilities[0,get_index("D", 1)] / old_transition_sum

    new_transition_probabilities[0,get_index("I", 0)] = cumulative_sum_begin * ratio_i0 # begin to I0
    new_transition_probabilities[0,get_index("D", 1)] = cumulative_sum_begin * ratio_d1 # begin to D1

    if(DEBUG):
        for row_index,row in enumerate(new_transition_probabilities[:num_states-1,:]):
            # print("rowindex: " + str(row_index))
            # print(row)
            npt.assert_almost_equal(np.sum(row), 1, decimal=2)


    for current_state in range(1,num_states - 1):
        if(not is_insertion(current_state)):
            if(transition_probabilities[current_state, num_states - 1] < 1):
                new_adjacency_matrix[current_state,num_states - 1] = 1
                new_transition_probabilities[current_state,num_states - 1] = p_exit
                cumulative_sum = 1 - p_exit
                old_transition_sum = 0.0
                original_destination_states_set = set()
                for destination_state in range(current_state + 1, num_states - 1):
                    if(transition_probabilities[current_state,destination_state] > 0):
                        old_transition_sum += transition_probabilities[current_state, destination_state]
                        original_destination_states_set.add(destination_state)
                for destination_state in original_destination_states_set:
                    new_transition_probabilities[current_state, destination_state] = cumulative_sum * (transition_probabilities[current_state, destination_state] / old_transition_sum)
                    if(math.isnan(new_transition_probabilities[current_state,destination_state])):
                        sys.stderr.write(f"{transition_probabilities[current_state,:]}\n")
                        sys.stderr.write(f"old transition sum: {old_transition_sum}\n")
                        sys.stderr.write(f"transition_probabilities[current_state,destination_state]: {transition_probabilities[current_state,destination_state]}\n")
                        sys.stderr.write(f"cumulative_sum: {cumulative_sum}\n")
                        sys.stderr.flush()

    if(DEBUG):
        for row_index,row in enumerate(new_transition_probabilities[:num_states-1,:]):
            # print("rowindex: " + str(row_index))
            # print(row)
            npt.assert_almost_equal(np.sum(row), 1, decimal=2)

    # exit(0)

    del adjacency_matrix
    del transition_probabilities
    gc.collect()
    return new_adjacency_matrix,new_transition_probabilities

def get_matrices(output_hmm, input_dir, backbone_alignment, model, output_prefix):
    alphabet = None
    if(model == "DNA"):
        alphabet = ["A", "C", "G", "T"]
    elif(model == "RNA"):
        alphabet = ["A", "C", "G", "U"]
    elif(model == "amino"):
        alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_columns = None
    for record in backbone_records:
        total_columns = len(backbone_records[record].seq)
        break
    num_states = (3 * total_columns) + 2 + 1
    num_characters = len(alphabet)

    # print("output_hmm:")
    # pp.pprint(output_hmm)

    # M = np.zeros((num_states, num_states), dtype=np.int8) # HMM adjacency matrix
    M = lil_matrix((num_states, num_states), dtype=np.int8)
    # T = np.zeros(M.shape, dtype=np.float32) # transition probability table
    T = lil_matrix(M.shape, dtype=np.float32) # transition probability table
    # P = np.zeros((num_states, num_characters), dtype=np.float32) # emission probability table
    P = lil_matrix((num_states, num_characters), dtype=np.float32) # emission probability table
    # current_emission_state_mask = np.zeros(num_states, dtype=np.float32)

    for current_column in output_hmm:
        for letter_index,letter in enumerate(alphabet):
            if(current_column == total_columns + 1):
                # this is the end state which has no emission probabilities or transitions going out
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
    if(DEBUG):
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


def build_hmm_profiles(input_dir, mappings, model, output_prefix):
    num_hmms = len(mappings)
    for current_hmm_index in range(num_hmms):
        with open(output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.out", "w") as stdout_f:
            with open(output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.err", "w") as stderr_f:
                current_input_file = input_dir + "/input_" + str(current_hmm_index) + ".fasta"
                current_output_file = output_prefix + "/" + str(current_hmm_index) + "-hmmbuild.profile"
                print("calling hmmbuild with output at " + current_output_file)
                subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.2/hmmbuild", "--cpu", "1", f"--{model.lower()}", "--ere", "0.59", "--symfrac", "0.0", "--informat", "afa", current_output_file, current_input_file], stdout=stdout_f, stderr=stderr_f)

def read_hmms(input_profile_files):
    hmms = {}
    for current_hmm_index,current_input_file in input_profile_files.items():
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

def get_probabilities_top_1_helper(input_dir, hmms, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix):
    hmm_weights = {}

    current_hmm_bitscores = bitscores[fragmentary_sequence_id]
    for current_hmm_file in hmms:
        current_sum = 0.0
        for compare_to_hmm_file in hmms:
            current_sum += 2**(float(current_hmm_bitscores[compare_to_hmm_file]) - float(current_hmm_bitscores[current_hmm_file]))
        if(current_sum != 0):
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
    # print("uncorrected hmm weights for " + str(fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
    # pp.pprint(hmm_weights)

    hmm_weights_tuple_arr = []
    for hmm_weight_index in hmm_weights:
        hmm_weights_tuple_arr.append((hmm_weight_index, hmm_weights[hmm_weight_index]))
    hmm_weights_tuple_arr.sort(key=lambda x: x[1], reverse=True)
    pp.pprint(hmm_weights_tuple_arr)
    top_hmm_index = hmm_weights_tuple_arr[0][0]
    print("top hmm for sequnece " + str(fragmentary_sequence_id) + " is " + str(top_hmm_index))
    top_hmm = hmms[top_hmm_index]

    output_hmm = {}
    for column_index in range(len(fragmentary_sequence) + 1):
        output_hmm[column_index] = {
            "match": [],
            "insertion": [],
            "insert_loop": 0,
            "self_match_to_insert": 0,
            "transition": {
            },
        }
        next_column = column_index + 1
        if(column_index == 0):
            output_hmm[column_index]["insertion"] = top_hmm["insertion"][column_index]
            output_hmm[column_index]["transition"][next_column] = top_hmm["transition"][column_index]
            output_hmm[column_index]["insert_loop"] = top_hmm["transition"][column_index][4]
            output_hmm[column_index]["self_match_to_insert"] = top_hmm["transition"][column_index][1]
        elif(column_index == len(fragmentary_sequence)):
            output_hmm[column_index]["match"] = top_hmm["match"][column_index]
            output_hmm[column_index]["insertion"] = top_hmm["insertion"][column_index]
            output_hmm[column_index]["transition"][total_columns + 1] = top_hmm["transition"][column_index]
            output_hmm[column_index]["insert_loop"] = top_hmm["transition"][column_index][4]
            output_hmm[column_index]["self_match_to_insert"] = top_hmm["transition"][column_index][1]
        else:
            output_hmm[column_index]["match"] = top_hmm["match"][column_index]
            output_hmm[column_index]["insertion"] = top_hmm["insertion"][column_index]
            output_hmm[column_index]["transition"][next_column] = top_hmm["transition"][column_index]
            output_hmm[column_index]["insert_loop"] = top_hmm["transition"][column_index][4]
            output_hmm[column_index]["self_match_to_insert"] = top_hmm["transition"][column_index][1]
    return output_hmm

def get_probabilities_helper(input_dir, hmms, support_value, input_sequence_files, backbone_alignment, fragmentary_sequence_id, fragmentary_sequence, mappings, bitscores, output_prefix):
    if(VERBOSE):
        print("the input hmms are")
        pp.pprint(hmms)
    hmm_freq_dict = {}
    input_alignment_sizes = {}
    for current_hmm_index,current_input_alignment in input_sequence_files.items():
        input_alignment_sizes[current_hmm_index] = 0
        for sequence_record in SeqIO.parse(current_input_alignment, "fasta"):
            input_alignment_sizes[current_hmm_index] += 1

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

        # current states probabilities is a map HMM index to the actual hmm
        # current hmm_bitscores contains a map of HMM index to bitscores
        for current_hmm_file in current_states_probabilities:
            current_sum = 0.0
            current_num_sequences = 0
            if(current_hmm_file in current_hmm_bitscores):
                # for compare_to_hmm_file in current_states_probabilities:
                    # if(compare_to_hmm_file in current_hmm_bitscores):
                        # current_num_sequences += input_alignment_sizes[compare_to_hmm_file]
                # print("current_num_sequences:" + str(current_num_sequences))
                for compare_to_hmm_file in current_states_probabilities:
                    if(compare_to_hmm_file in current_hmm_bitscores):
                        # print(str(compare_to_hmm_file) + " has " + str(current_num_sequences) + " sequences")
                        # print(str(compare_to_hmm_file) + " has " + str(current_hmm_bitscores[compare_to_hmm_file]) + " bitscore")
                        current_sum += 2**(float(current_hmm_bitscores[compare_to_hmm_file]) - float(current_hmm_bitscores[current_hmm_file]) + np.log2(input_alignment_sizes[compare_to_hmm_file] / input_alignment_sizes[current_hmm_file]))
                hmm_weights[current_hmm_file] = 1 / current_sum
            else:
                hmm_weights[current_hmm_file] = 0
        if(VERBOSE):
            print("uncorrected hmm weights for " + str(fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            pp.pprint(hmm_weights)

        is_hmm_weights_all_zero = True
        for hmm_weight_index in hmm_weights:
            if(hmm_weights[hmm_weight_index] != 0):
                is_hmm_weights_all_zero = False
        if(is_hmm_weights_all_zero):
            # for hmm_weight_index in current_states_probabilities:
                # hmm_weights[hmm_weight_index] = 1 / len(current_states_probabilities)
            for hmm_weight_index in hmm_weights:
                hmm_weights[hmm_weight_index] = 1 / (len(hmm_weights))
        if(VERBOSE):
            print("zero corrected hmm weights for " + str(fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            pp.pprint(hmm_weights)
            npt.assert_almost_equal(sum(hmm_weights.values()), 1)

        hmm_weights_tuple_arr = []
        for hmm_weight_index in hmm_weights:
            hmm_weights_tuple_arr.append((hmm_weight_index, hmm_weights[hmm_weight_index]))
        hmm_weights_tuple_arr.sort(key=lambda x: x[1], reverse=True)

        if(VERBOSE):
            print("hmm tuple arr for " + str(fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            pp.pprint(hmm_weights_tuple_arr)

        adjusted_support_value = support_value

        if(DEBUG):
            assert adjusted_support_value >= 0.0
            assert adjusted_support_value <= 1.0

        hmm_weights = {}
        hmm_weight_value_sum = 0.0
        # TODO: num hmms > len(hmm_weights_tuple_arr) then what?
        for hmm_weight_index,hmm_weight_value in hmm_weights_tuple_arr:
            hmm_weights[hmm_weight_index] = hmm_weight_value
            hmm_weight_value_sum += hmm_weight_value
            if(hmm_weight_value_sum > adjusted_support_value):
                break
        # note: 0.0 would happen if the top hit is zero
        # if the top hit is zero, then we should set everything to be 1 / len(hmm_weights)
        for hmm_weight_index in hmm_weights:
            if(hmm_weight_value_sum == 0.0):
                hmm_weights[hmm_weight_index] = 1 / len(hmm_weights)
            else:
                hmm_weights[hmm_weight_index] /= hmm_weight_value_sum
        if(VERBOSE):
            print("corrected hmm weights for " + str(fragmentary_sequence_id) + " at backbone state " + str(backbone_state_index))
            pp.pprint(hmm_weights)

        for hmm_weight_index in hmm_weights:
            if(hmm_weight_index not in hmm_freq_dict):
                hmm_freq_dict[hmm_weight_index] = 0
            hmm_freq_dict[hmm_weight_index] += 1

        if(DEBUG):
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
        elif(backbone_state_index == total_columns):
            for current_hmm_file in hmm_weights:
                if(backbone_state_index not in mappings[current_hmm_file]):
                    continue
                current_state_in_hmm = mappings[current_hmm_file][backbone_state_index]
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
                if(backbone_state_index not in mappings[current_hmm_file]):
                    continue
                current_state_in_hmm = mappings[current_hmm_file][backbone_state_index]
                next_state_in_hmm = current_state_in_hmm + 1
                corresponding_next_backbone_state = None
                for state_index in mappings[current_hmm_file]:
                    if(mappings[current_hmm_file][state_index] == next_state_in_hmm):
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
    if(VERBOSE):
        pp.pprint(hmm_freq_dict)
    return output_hmm

def get_bitscores_helper(input_dir, num_hmms, input_profile_files, backbone_alignment, fragmentary_sequence_file, output_prefix):
    hmm_bitscores = {}
    string_infinity = "9" * 100
    for current_hmm_index,current_input_file in input_profile_files.items():
        with open(output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.out", "w") as stdout_f:
            with open(output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.err", "w") as stderr_f:
                current_search_file = output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.output"
                subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.2/hmmsearch", "--noali", "--cpu", "1", "-o", current_search_file, "-E", string_infinity, "--domE", string_infinity, "--max", current_input_file,fragmentary_sequence_file], stdout=stdout_f, stderr=stderr_f)

    for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        current_fragmentary_sequence = fragmentary_sequence_record.seq
        current_hmm_bitscores = {}
        for current_hmm_index in range(num_hmms):
            current_search_file = output_prefix + "/" + str(current_hmm_index) + "-hmmsearch.output"
            with open(current_search_file, "r") as f:
                count_from_score_section_start = 0
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

        # if(len(current_hmm_bitscores) !=  num_hmms):
            # raise Exception("Not all hmms reported bitscores for fragmentary sequence " + fragmentary_sequence_record.id)
        hmm_bitscores[fragmentary_sequence_record.id] = current_hmm_bitscores
    return hmm_bitscores

def create_mappings_helper(input_fasta_filenames, backbone_alignment):
    # print("input fasta files")
    # print(input_fasta_filenames)
    # print("backbone alignment")
    # print(backbone_alignment)
    num_hmms = len(input_fasta_filenames)
    hmm_weights = {}
    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_columns = None
    for record in backbone_records:
        total_columns = len(backbone_records[record].seq)
        break

    match_state_mappings = {} # from hmm_index to {map of backbone match state index to current input hmm build fasta's match state index}
    # for current_hmm_index,current_input_file in enumerate(input_fasta_filenames):
    for current_hmm_index,current_input_file in input_fasta_filenames.items():
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
        # pp.pprint(mapping)
        max_match_state_index = -1
        match_state_sum = 0
        for backbone_index,match_state_index in mapping.items():
            if(match_state_index > max_match_state_index):
                max_match_state_index = match_state_index
            match_state_sum += match_state_index
        if(DEBUG):
            assert match_state_sum == (max_match_state_index * (max_match_state_index + 1) / 2)
        # DEBUG TODO: comeback to this
        # DEBUG TODO: second note i think this is correct now
        match_state_mappings[mapping_index][total_columns + 1] = max_match_state_index + 1

    for mapping_index,mapping in match_state_mappings.items():
        mapping[0] = 0
    return match_state_mappings


if __name__ == "__main__":
    merge_hmms()
