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

def test_merge_1_insertion_2_fragments_2_subsets_0():
    test_root = "./test/merge_test/merge_1_insertion_2_fragments_2_subsets_0/"
    test_suffix = "merge_1_insertion_2_fragments_2_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    cumulative_hmm = main.custom_merge_hmm_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
    merged_alignment = main.compute_alignment(cumulative_hmm, input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
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
    }

def test_merge_1_insertion_2_fragments_2_subsets_1():
    test_root = "./test/merge_test/merge_1_insertion_2_fragments_2_subsets_1/"
    test_suffix = "merge_1_insertion_2_fragments_2_subsets_1/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    cumulative_hmm = main.custom_merge_hmm_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
    merged_alignment = main.compute_alignment(cumulative_hmm, input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
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
    }

def test_merge_1_insertion_2_fragments_2_subsets_2():
    test_root = "./test/merge_test/merge_1_insertion_2_fragments_2_subsets_2/"
    test_suffix = "merge_1_insertion_2_fragments_2_subsets_2/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    cumulative_hmm = main.custom_merge_hmm_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
    merged_alignment = main.compute_alignment(cumulative_hmm, input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
    assert merged_alignment == {
        "fragment-0": "AAAAt---AAAAGGGGGGGGGGGGGGGGGGGG",
        "fragment-1": "TTTTatttTTTTGGGGGGGGGGGGGGGGG---",
        "s1":         "AAAA----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s2":         "AAAA----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s3":         "AAAA----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s4":         "AAAA----AAAAGGGGGGGGGGGGGGGGGGGG",
        "s5":         "TTTT----TTTTGGGGGGGGGGGGGGGGGGGG",
        "s6":         "TTTT----TTTTGGGGGGGGGGGGGGGGGGGG",
        "s7":         "TTTT----TTTTGGGGGGGGGGGGGGGGGGGG",
        "s8":         "TTTT----TTTTGGGGGGGGGGGGGGGGGGGG",
    }


def test_merge_1_insertion_2_subsets_0():
    test_root = "./test/merge_test/merge_1_insertion_2_subsets_0/"
    test_suffix = "merge_1_insertion_2_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    cumulative_hmm = main.custom_merge_hmm_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
    merged_alignment = main.compute_alignment(cumulative_hmm, input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
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
    }

def test_merge_2_subsets_0():
    test_root = "./test/merge_test/merge_2_subsets_0/"
    test_suffix = "merge_2_subsets_0/"
    input_dir,backbone_alignment,fragment_sequences_file,test_root = setup_merge_inputs(test_root, test_suffix)
    output_prefix = TEMP_OUTPUT_PREFIX + test_suffix
    cumulative_hmm = main.custom_merge_hmm_helper(input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
    merged_alignment = main.compute_alignment(cumulative_hmm, input_dir, backbone_alignment, fragment_sequences_file, output_prefix)
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

