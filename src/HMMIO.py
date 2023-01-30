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


class HMMIO:
    num_hmms = None
    input_profile_files = None
    input_sequence_files = None
    bitscores = None
    hmms = None
    mappings = None
    def __init__(self, input_type, backbone_alignment, input_dir, model, fragmentary_sequence_file, output_prefix, logger, debug, verbose):
        self.input_type = input_type
        self.backbone_alignment = backbone_alignment
        self.input_dir = input_dir
        self.model = model
        self.fragmentary_sequence_file = fragmentary_sequence_file
        self.output_prefix = output_prefix
        self.logger = logger
        self.debug = debug
        self.verbose = verbose
        self.process()

    def process(self):
        if(self.input_type == "custom"):
            self.custom_helper()
        elif(self.input_type == "sepp"):
            self.sepp_helper()
        elif(self.input_type == "upp"):
            self.upp_helper()
        else:
            self.logger.errror(f"Unsupported mode: {self.input_type}")
            raise Exception(f"Unsupported mode: {self.input_type}")

        self.mappings = self.create_mappings_helper()
        self.logger.verbose("mappings")
        self.logger.verbose_pprint(self.mappings)

        if(self.input_type == "custom"):
            self.logger.info("type is custom")
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
                if(self.debug):
                    for key_so_far in cumulative_mapping:
                        if key_so_far in current_mapping:
                            assert cumulative_mapping[key_so_far] == current_mapping[key_so_far]
                cumulative_mapping.update(current_mapping)
            if(self.debug):
                assert current_hmm_index not in match_state_mappings
            match_state_mappings[current_hmm_index] = cumulative_mapping

        for mapping_index,mapping in match_state_mappings.items():
            max_match_state_index = -1
            match_state_sum = 0
            for backbone_index,match_state_index in mapping.items():
                if(match_state_index > max_match_state_index):
                    max_match_state_index = match_state_index
                match_state_sum += match_state_index
            if(self.debug):
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
                    self.logger.info("calling hmmbuild with output at " + current_output_file)
                    subprocess.call(["/usr/bin/time", "-v", "/opt/sepp/.sepp/bundled-v4.5.2/hmmbuild", "--cpu", "1", f"--{self.model.lower()}", "--ere", "0.59", "--symfrac", "0.0", "--informat", "afa", current_output_file, current_input_file], stdout=stdout_f, stderr=stderr_f)
