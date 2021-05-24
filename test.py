import main
import os
import shutil

TEMP_OUTPUT_PREFIX = "./tmp/"

def setup_mapping_inputs(test_root, num_subsets):
    input_fasta_filenames = []
    for i in range(num_subsets):
        input_fasta_filenames.append(test_root + "input_" + str(i) + ".fasta")
    backbone_alignment = test_root + "backbone.fasta"
    return input_fasta_filenames,backbone_alignment

def setup_merge_inputs(test_root, test_suffix):
    backbone_alignment = test_root + "backbone.fasta"
    fragment_sequences_file = test_root + "fragment.fasta"
    if os.path.exists(TEMP_OUTPUT_PREFIX + test_suffix):
        shutil.rmtree(TEMP_OUTPUT_PREFIX + test_suffix)
    os.mkdir(TEMP_OUTPUT_PREFIX + test_suffix)
    return test_root,backbone_alignment,fragment_sequences_file,test_root

# def test_merge_1_insertion_10_fragment_5_subsets_6():
#     test_root = "./test/merge_test/merge_1_insertion_10_fragment_5_subsets_6/"
#     test_suffix = "merge_1_insertion_10_fragment_5_subsets_6"
#     input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
#     output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
#     for num_threads in range(1, 12):
#         merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", num_threads, "DNA", "FASTA", False)
#         assert merged_alignment == {
#             "s1":        "AAAAA---------------------------------------------",
#             "s2":        "AAAAA---------------------------------------------",
#             "s3":        "AAAAA---------------------------------------------",
#             "s4":        "AAAAA---------------------------------------------",
#             "s5":        "AAAAA---------------------------------------------",
#             "s6":        "AAAAA---------------------------------------------",
#             "s7":        "AAAAA---------------------------------------------",
#             "s8":        "AAAAA---------------------------------------------",
#             "s9":        "AAAAA---------------------------------------------",
#             "s10":       "AAAAA---------------------------------------------",
#             "fragment0": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment1": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment2": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment3": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment4": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment5": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment6": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment7": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment8": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat",
#             "fragment9": "AAAAAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaat"
#             # "backbone_indices": [1, 2, 3, 4, 5, ],
#         }

def test_merge_1_insertion_1_fragment_5_subsets_0():
    test_root = "./test/merge_test/merge_1_insertion_1_fragment_5_subsets_0/"
    test_suffix = "merge_1_insertion_1_fragment_5_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "tAAAAA",
        "s1":  "-AAAAA",
        "s2":  "-AAAAA",
        "s3":  "-AAAAA",
        "s4":  "-AAAAA",
        "s5":  "-AAAAA",
        "s6":  "-AAAAA",
        "s7":  "-AAAAA",
        "s8":  "-AAAAA",
        "s9":  "-AAAAA",
        "s10": "-AAAAA",
        "backbone_indices": ["I", 1, 2, 3, 4, 5],
    }

def test_merge_1_insertion_1_fragment_5_subsets_1():
    test_root = "./test/merge_test/merge_1_insertion_1_fragment_5_subsets_1/"
    test_suffix = "merge_1_insertion_1_fragment_5_subsets_1/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "AtAAAA",
        "s1":  "A-AAAA",
        "s2":  "A-AAAA",
        "s3":  "A-AAAA",
        "s4":  "A-AAAA",
        "s5":  "A-AAAA",
        "s6":  "A-AAAA",
        "s7":  "A-AAAA",
        "s8":  "A-AAAA",
        "s9":  "A-AAAA",
        "s10": "A-AAAA",
        "backbone_indices": [1, "I", 2, 3, 4, 5],
    }
def test_merge_1_insertion_1_fragment_5_subsets_2():
    test_root = "./test/merge_test/merge_1_insertion_1_fragment_5_subsets_2/"
    test_suffix = "merge_1_insertion_1_fragment_5_subsets_2/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "AAtAAA",
        "s1":  "AA-AAA",
        "s2":  "AA-AAA",
        "s3":  "AA-AAA",
        "s4":  "AA-AAA",
        "s5":  "AA-AAA",
        "s6":  "AA-AAA",
        "s7":  "AA-AAA",
        "s8":  "AA-AAA",
        "s9":  "AA-AAA",
        "s10": "AA-AAA",
        "backbone_indices": [1, 2, "I", 3, 4, 5],
    }
def test_merge_1_insertion_1_fragment_5_subsets_3():
    test_root = "./test/merge_test/merge_1_insertion_1_fragment_5_subsets_3/"
    test_suffix = "merge_1_insertion_1_fragment_5_subsets_3/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "AAAtAA",
        "s1":  "AAA-AA",
        "s2":  "AAA-AA",
        "s3":  "AAA-AA",
        "s4":  "AAA-AA",
        "s5":  "AAA-AA",
        "s6":  "AAA-AA",
        "s7":  "AAA-AA",
        "s8":  "AAA-AA",
        "s9":  "AAA-AA",
        "s10": "AAA-AA",
        "backbone_indices": [1, 2, 3, "I", 4, 5],
    }
def test_merge_1_insertion_1_fragment_5_subsets_4():
    test_root = "./test/merge_test/merge_1_insertion_1_fragment_5_subsets_4/"
    test_suffix = "merge_1_insertion_1_fragment_5_subsets_4/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "AAAAtA",
        "s1":  "AAAA-A",
        "s2":  "AAAA-A",
        "s3":  "AAAA-A",
        "s4":  "AAAA-A",
        "s5":  "AAAA-A",
        "s6":  "AAAA-A",
        "s7":  "AAAA-A",
        "s8":  "AAAA-A",
        "s9":  "AAAA-A",
        "s10": "AAAA-A",
        "backbone_indices": [1, 2, 3, 4, "I", 5],
    }
def test_merge_1_insertion_1_fragment_5_subsets_5():
    test_root = "./test/merge_test/merge_1_insertion_1_fragment_5_subsets_5/"
    test_suffix = "merge_1_insertion_1_fragment_5_subsets_5/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "AAAAAt",
        "s1":  "AAAAA-",
        "s2":  "AAAAA-",
        "s3":  "AAAAA-",
        "s4":  "AAAAA-",
        "s5":  "AAAAA-",
        "s6":  "AAAAA-",
        "s7":  "AAAAA-",
        "s8":  "AAAAA-",
        "s9":  "AAAAA-",
        "s10": "AAAAA-",
        "backbone_indices": [1, 2, 3, 4, 5, "I"],
    }

def test_merge_7_insertion_1_fragment_5_subsets_0():
    test_root = "./test/merge_test/merge_7_insertion_1_fragment_5_subsets_0/"
    test_suffix = "merge_7_insertion_1_fragment_5_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "tttttttAAAAA",
        "s1":  "-------AAAAA",
        "s2":  "-------AAAAA",
        "s3":  "-------AAAAA",
        "s4":  "-------AAAAA",
        "s5":  "-------AAAAA",
        "s6":  "-------AAAAA",
        "s7":  "-------AAAAA",
        "s8":  "-------AAAAA",
        "s9":  "-------AAAAA",
        "s10": "-------AAAAA",
        "backbone_indices": ["I", "I", "I", "I", "I", "I", "I", 1, 2, 3, 4, 5],
    }

def test_merge_5_insertion_1_fragment_5_subsets_0():
    test_root = "./test/merge_test/merge_5_insertion_1_fragment_5_subsets_0/"
    test_suffix = "merge_5_insertion_1_fragment_5_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "AAtttttAAA",
        "s1":  "AA-----AAA",
        "s2":  "AA-----AAA",
        "s3":  "AA-----AAA",
        "s4":  "AA-----AAA",
        "s5":  "AA-----AAA",
        "s6":  "AA-----AAA",
        "s7":  "AA-----AAA",
        "s8":  "AA-----AAA",
        "s9":  "AA-----AAA",
        "s10": "AA-----AAA",
        "backbone_indices": [1, 2, "I", "I", "I", "I", "I", 3, 4, 5],
    }

def test_merge_1_insertion_2_fragments_2_subsets_0():
    test_root = "./test/merge_test/merge_1_insertion_2_fragments_2_subsets_0/"
    test_suffix = "merge_1_insertion_2_fragments_2_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment-0": "A-AAAtAAAA",
        "fragment-1": "AtAAA-AAAA",
        "s1":         "A-AAA-AAAA",
        "s2":         "A-AAA-AAAA",
        "s3":         "A-AAA-AAAA",
        "s4":         "A-AAA-AAAA",
        "s5":         "A-AAA-AAAA",
        "s6":         "A-AAA-AAAA",
        "s7":         "A-AAA-AAAA",
        "s8":         "A-AAA-AAAA",
        "backbone_indices": [1, "I", 2, 3, 4, "I", 5, 6, 7, 8],
    }

def test_merge_1_insertion_2_fragments_2_subsets_1():
    test_root = "./test/merge_test/merge_1_insertion_2_fragments_2_subsets_1/"
    test_suffix = "merge_1_insertion_2_fragments_2_subsets_1/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment-0": "A-AAAtAAAAGGGGGGGGGGGGGGGGGGGG",
        "fragment-1": "TaTTT-TTTTGGGGGGGGGGGGGGGGGGGG",
        "s1":         "A-AAA-AAAAGGGGGGGGGGGGGGGGGGGG",
        "s2":         "A-AAA-AAAAGGGGGGGGGGGGGGGGGGGG",
        "s3":         "A-AAA-AAAAGGGGGGGGGGGGGGGGGGGG",
        "s4":         "A-AAA-AAAAGGGGGGGGGGGGGGGGGGGG",
        "s5":         "T-TTT-TTTTGGGGGGGGGGGGGGGGGGGG",
        "s6":         "T-TTT-TTTTGGGGGGGGGGGGGGGGGGGG",
        "s7":         "T-TTT-TTTTGGGGGGGGGGGGGGGGGGGG",
        "s8":         "T-TTT-TTTTGGGGGGGGGGGGGGGGGGGG",
        "backbone_indices": [1, "I", 2, 3, 4, "I", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    }

def test_merge_1_insertion_2_fragments_2_subsets_2():
    test_root = "./test/merge_test/merge_1_insertion_2_fragments_2_subsets_2/"
    test_suffix = "merge_1_insertion_2_fragments_2_subsets_2/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment-0": "AAAAt----AAAAGGGGGGGGGGGGGGGGGGGG",
        "fragment-1": "TTTT-atttTTTTGGGGGGGGGGGGGGGGG---",
        "s1":         "AAAA-----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s2":         "AAAA-----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s3":         "AAAA-----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s4":         "AAAA-----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s5":         "TTTT-----TTTTGGGGGGGGGGGGGGGGGGGG",
        "s6":         "TTTT-----TTTTGGGGGGGGGGGGGGGGGGGG",
        "s7":         "TTTT-----TTTTGGGGGGGGGGGGGGGGGGGG",
        "s8":         "TTTT-----TTTTGGGGGGGGGGGGGGGGGGGG",
        "backbone_indices": [1, 2, 3, 4, "I", "I", "I", "I", "I", 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
    }


def test_merge_1_insertion_2_subsets_0():
    test_root = "./test/merge_test/merge_1_insertion_2_subsets_0/"
    test_suffix = "merge_1_insertion_2_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)
    assert merged_alignment == {
        "fragment": "AAAAtAAAA",
        "s1":       "AAAA-AAAA",
        "s2":       "AAAA-AAAA",
        "s3":       "AAAA-AAAA",
        "s4":       "AAAA-AAAA",
        "s5":       "AAAA-AAAA",
        "s6":       "AAAA-AAAA",
        "s7":       "AAAA-AAAA",
        "s8":       "AAAA-AAAA",
        "backbone_indices": [1, 2, 3, 4, "I", 5, 6, 7, 8],
    }

def test_merge_2_subsets_0():
    test_root = "./test/merge_test/merge_2_subsets_0/"
    test_suffix = "merge_2_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    merged_alignment = main.merge_hmms_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix, "custom", 1, "DNA", "FASTA", False)

    assert merged_alignment == {
        "fragment": "AAAAAAAA----------------------",
        "s1":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "s2":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "s3":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "s4":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "s5":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "s6":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "s7":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "s8":       "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "backbone_indices": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    }

def test_identical_backbone_3_subsets():
    ''' in this test case, there are 3 subsets and all the sequences are identical
    with no all gap columns
    '''
    test_root = "./test/mapping_test/identical_3_subsets_0/"
    mapping = main.create_mappings_helper(*setup_mapping_inputs(test_root, 3))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
        2: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10},
    }

def test_identical_backbone_2_subsets():
    ''' in this test case, there are 2 subsets and all the sequences are identical
    with no all gap columns
    '''
    test_root = "./test/mapping_test/identical_2_subsets_0/"
    mapping = main.create_mappings_helper(*setup_mapping_inputs(test_root, 2))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    }

def test_middle_missing_3_subsets():
    ''' in this test case, there are 3 subsets but three columns in the middle are missing
    '''
    test_root = "./test/mapping_test/middle_missing_3_subsets_0/"
    mapping = main.create_mappings_helper(*setup_mapping_inputs(test_root, 3))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6, 10: 7},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6, 10: 7},
        2: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6, 10: 7},
    }

def test_front_missing_2_subsets():
    ''' in this test case, there are 2 subsets but three columns in the front are missing
    '''
    test_root = "./test/mapping_test/front_missing_2_subsets_0/"
    mapping = main.create_mappings_helper(*setup_mapping_inputs(test_root, 2))
    assert mapping == {
        0: {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7},
        1: {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7},
    }

def test_end_missing_2_subsets():
    ''' in this test case, there are 2 subsets but three columns in the end are missing
    '''
    test_root = "./test/mapping_test/end_missing_2_subsets_0/"
    mapping = main.create_mappings_helper(*setup_mapping_inputs(test_root, 2))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 10: 7},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 10: 7},
    }


def test_front_middle_end_missing_3_subsets():
    ''' in this test case, there are 3 subsets but three columns in the front are missing
    for the first input file, three columns in the middle are missing in the second input
    file, and three columns in the end are missing in the third input file
    '''
    test_root = "./test/mapping_test/front_middle_end_missing_3_subsets_0/"
    mapping = main.create_mappings_helper(*setup_mapping_inputs(test_root, 3))
    assert mapping == {
        0: {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6, 10: 7},
        2: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 10: 7},
    }


