import glob
import math
import numpy as np
import os
import pprint
import subprocess
import sys

from Bio import SeqIO
import click
import pyhmmer
from pyhmmer.plan7 import HMM,HMMFile
pp = pprint.PrettyPrinter(indent=4)


@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="The input temp root dir of sepp that contains all the HMMs")
@click.option("--backbone-alignment", required=True, type=click.Path(exists=True), help="The input backbone alignment to SEPP")
@click.option("--fragmentary-sequence-file", required=True, type=click.Path(exists=True), help="The input fragmentary sequence file to SEPP")
@click.option("--output-prefix", required=True, type=click.Path(), help="Output prefix")
def merge_hmms(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix):
    mappings = create_mappings(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix)
    print("the mappings are")
    pp.pprint(mappings)
    bitscores = get_bitscores(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix)
    print("bitscores are")
    pp.pprint(bitscores)
    output_hmm = get_probabilities(input_dir, backbone_alignment, fragmentary_sequence_file, mappings, bitscores, output_prefix)
    print("output hmm is")
    pp.pprint(output_hmm)

    # match_probabilities,insertion_probabilities,transition_probabilities = get_probabilities(input_dir, backbone_alignment, fragmentary_sequence_file, bitscores, output_prefix)
    # print(mappings)


def get_probabilities(input_dir, backbone_alignment, fragmentary_sequence_file, mappings, bitscores, output_prefix):
    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_nodes = None
    for record in backbone_records:
        total_nodes = len(backbone_records[record].seq)
        break
    num_hmms = len(list(glob.glob(input_dir + "/P_*")))

    backbone_probabilities = {
        "match": [],
        "insertion": [],
        "transition": {
            "destination_state": ["m->m", "m->i", "m->d", "i->m", "i->i", "d->m", "d->d"],
        },
    }

    hmms = {}
    output_hmm = {}
    transition_origins = {}

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
    # print(mappings[0])
    print("the input hmms are")
    pp.pprint(hmms)
    for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        current_fragmentary_sequence = fragmentary_sequence_record.seq
        current_hmm_bitscores = bitscores[fragmentary_sequence_record.id]
        for backbone_state_index in range(total_nodes + 1):
            current_states_probabilities = {}
            hmm_weights = {}
            for current_hmm_index,current_hmm in hmms.items():
                current_hmm_mapping = mappings[current_hmm_index]
                if(backbone_state_index in current_hmm_mapping):
                    current_states_probabilities[current_hmm_index] = current_hmm
            # print("In state " + str(backbone_state_index) + ", there are " + str(len(current_states_probabilities)) + " hmms that have a match state")

            current_hmm_bitscores = bitscores[fragmentary_sequence_record.id]
            for current_hmm_file in current_states_probabilities:
                current_sum = 0.0
                for compare_to_hmm_file in current_states_probabilities:
                    current_sum += 2**(float(current_hmm_bitscores[compare_to_hmm_file]) - float(current_hmm_bitscores[current_hmm_file]))
                hmm_weights[current_hmm_file] = 1 / current_sum

            # print(hmm_weights)
            output_hmm[backbone_state_index] = {
                "match": [],
                "insertion": [],
                "transition": {
                },
            }
            if(backbone_state_index == 0):
                # this is the begin state
                for current_hmm_file in hmm_weights:
                    if(backbone_state_index not in mappings[current_hmm_file]):
                        continue
                    current_state_in_hmm = mappings[current_hmm_file][backbone_state_index]
                    next_state_in_hmm = current_state_in_hmm + 1
                    corresponding_next_backbone_state = None
                    for state_index in mappings[current_hmm_file]:
                        if(mappings[current_hmm_file][state_index] == next_state_in_hmm):
                            corresponding_next_backbone_state = state_index
                    if(output_hmm[backbone_state_index]["insertion"] == []):
                        output_hmm[backbone_state_index]["insertion"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["insertion"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]

                    output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                    # print(str(backbone_state_index) + " to " + str(corresponding_next_backbone_state))
            elif(backbone_state_index == total_nodes):
                # this is the end state
                for current_hmm_file in hmm_weights:
                    if(backbone_state_index not in mappings[current_hmm_file]):
                        continue
                    current_state_in_hmm = mappings[current_hmm_file][backbone_state_index]
                    if(output_hmm[backbone_state_index]["match"] == []):
                        output_hmm[backbone_state_index]["match"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["match"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]

                    if(output_hmm[backbone_state_index]["insertion"] == []):
                        output_hmm[backbone_state_index]["insertion"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["insertion"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]

                    # print(current_states_probabilities[current_hmm_file]["transition"])
                    output_hmm[backbone_state_index]["transition"]["END from file " + str(current_hmm_file)] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]

                    # print(str(backbone_state_index) + " to " + str(corresponding_next_backbone_state))
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
                    # print("current state in hmm: " + str(current_state_in_hmm))
                    # print("corresponding (tranisitioning to) next backbone state index: " + str(corresponding_next_backbone_state))

                    if(output_hmm[backbone_state_index]["match"] == []):
                        output_hmm[backbone_state_index]["match"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["match"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["match"][current_state_in_hmm]

                    if(output_hmm[backbone_state_index]["insertion"] == []):
                        output_hmm[backbone_state_index]["insertion"] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]
                    else:
                        output_hmm[backbone_state_index]["insertion"] += hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["insertion"][current_state_in_hmm]

                    output_hmm[backbone_state_index]["transition"][corresponding_next_backbone_state] = hmm_weights[current_hmm_file] * current_states_probabilities[current_hmm_file]["transition"][current_state_in_hmm]
                    # print(str(backbone_state_index) + " to " + str(corresponding_next_backbone_state))
                    # print(-np.log(current_states_probabilities[current_hmm_file]["transition"][1]))
            # print(-np.log(output_hmm[0][]["transition"]))
    return output_hmm


def get_bitscores(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix):
    # DEBUG
    return {
        "x": {
            0: 1.1,
            1: 1.1,
        }
    }
    hmm_bitscores = {}
    num_hmms = len(list(glob.glob(input_dir + "/P_*")))
    for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        current_fragmentary_sequence = fragmentary_sequence_record.seq
        for current_hmm_index in range(num_hmms):
            for current_input_file in glob.glob(input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.model.*"):
                with open(output_prefix + "/" + str(current_hmm_index) + "-" + fragmentary_sequence_record.id + "-hmmsearch.out", "w") as stdout_f:
                    with open(output_prefix + "/" + str(current_hmm_index) + "-" + fragmentary_sequence_record.id + "-hmmsearch.err", "w") as stderr_f:
                        subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.1/hmmsearch", "--noali", "--cpu", "1", "-o", output_prefix + "/" + str(current_hmm_index) + "-" + fragmentary_sequence_record.id + "-hmmsearch.output", "-E", "99999999999", "--max", current_input_file, fragmentary_sequence_file], stdout=stdout_f, stderr=stderr_f)
                        # subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.1/hmmsearch", "--noali", "--cpu", "1", "-o", output_prefix + "/" + str(current_hmm_index) + "-" + fragmentary_sequence_record.id + "-hmmsearch.output", "-T", "-100000000", "--max", current_input_file, fragmentary_sequence_file], stdout=stdout_f, stderr=stderr_f)

    for fragmentary_sequence_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        current_fragmentary_sequence = fragmentary_sequence_record.seq
        current_hmm_bitscores = {}
        for current_hmm_index in range(num_hmms):
            current_search_file = output_prefix + "/" + str(current_hmm_index) + "-" + fragmentary_sequence_record.id + "-hmmsearch.output"
            with open(current_search_file, "r") as f:
                count_from_evalue = 0
                evalue_encountered = False
                for line in f:
                    if(count_from_evalue == 1):
                        current_hmm_bitscores[current_hmm_index] = line.split()[1]
                        break
                    if(evalue_encountered):
                        count_from_evalue += 1
                    if("E-value" in line and "score" in line):
                        evalue_encountered = True
        hmm_bitscores[fragmentary_sequence_record.id] = current_hmm_bitscores
    return hmm_bitscores



def create_mappings(input_dir, backbone_alignment, fragmentary_sequence_file, output_prefix):
    num_hmms = len(list(glob.glob(input_dir + "/P_*")))
    input_fasta_filenames = []
    for current_hmm_index in range(num_hmms):
        for current_input_file in glob.glob(input_dir + "/P_" + str(current_hmm_index) + "/A_" + str(current_hmm_index) + "_0/hmmbuild.input.*.fasta"):
            input_fasta_filenames.append(current_input_file)
    return create_mappings_helper(input_fasta_filenames, num_hmms, backbone_alignment, fragmentary_sequence_file, output_prefix)

def create_mappings_helper(input_fasta_filenames, num_hmms, backbone_alignment, fragmentary_sequence_file, output_prefix):
    hmm_weights = {}
    backbone_records = SeqIO.to_dict(SeqIO.parse(backbone_alignment, "fasta"))
    total_nodes = None
    for record in backbone_records:
        total_nodes = len(backbone_records[record].seq)
        break

    fragment_length = None
    for fragmentary_record in SeqIO.parse(fragmentary_sequence_file, "fasta"):
        fragment_length = len(fragmentary_record.seq)
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
        pp.pprint(match_state_mappings)
        assert current_hmm_index not in match_state_mappings
        match_state_mappings[current_hmm_index] = cumulative_mapping

    for mapping_index,mapping in match_state_mappings.items():
        max_match_state_index = -1
        match_state_sum = 0
        for backbone_index,match_state_index in mapping.items():
            if(match_state_index > max_match_state_index):
                max_match_state_index = match_state_index
            match_state_sum += match_state_index
        assert match_state_sum == ((max_match_state_index * (max_match_state_index + 1)) / 2)

    for mapping_index,mapping in match_state_mappings.items():
        mapping[0] = 0

    return match_state_mappings


if __name__ == "__main__":
    merge_hmms()
