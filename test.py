import main


def setup_inputs(test_root, num_subsets):
    input_fasta_filenames = []
    for i in range(num_subsets):
        input_fasta_filenames.append(test_root + "input_" + str(i) + ".fasta")
    backbone_alignment = test_root + "backbone.fasta"
    return input_fasta_filenames,backbone_alignment

def test_identical_backbone_3_subsets():
    ''' in this test case, there are 3 subsets and all the sequences are identical
    with no all gap columns
    '''
    test_root = "./test/mapping_test/identical_3_subsets_0/"
    mapping = main.create_mappings_helper(*setup_inputs(test_root, 3))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
        2: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9},
    }

def test_identical_backbone_2_subsets():
    ''' in this test case, there are 2 subsets and all the sequences are identical
    with no all gap columns
    '''
    test_root = "./test/mapping_test/identical_2_subsets_0/"
    mapping = main.create_mappings_helper(*setup_inputs(test_root, 2))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
    }

def test_middle_missing_3_subsets():
    ''' in this test case, there are 3 subsets but three columns in the middle are missing
    '''
    test_root = "./test/mapping_test/middle_missing_3_subsets_0/"
    mapping = main.create_mappings_helper(*setup_inputs(test_root, 3))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6},
        2: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6},
    }

def test_front_missing_2_subsets():
    ''' in this test case, there are 2 subsets but three columns in the front are missing
    '''
    test_root = "./test/mapping_test/front_missing_2_subsets_0/"
    mapping = main.create_mappings_helper(*setup_inputs(test_root, 2))
    assert mapping == {
        0: {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6},
        1: {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6},
    }

def test_end_missing_2_subsets():
    ''' in this test case, there are 2 subsets but three columns in the end are missing
    '''
    test_root = "./test/mapping_test/end_missing_2_subsets_0/"
    mapping = main.create_mappings_helper(*setup_inputs(test_root, 2))
    assert mapping == {
        0: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
    }


def test_front_middle_end_missing_3_subsets():
    ''' in this test case, there are 3 subsets but three columns in the front are missing
    for the first input file, three columns in the middle are missing in the second input
    file, and three columns in the end are missing in the third input file
    '''
    test_root = "./test/mapping_test/front_middle_end_missing_3_subsets_0/"
    mapping = main.create_mappings_helper(*setup_inputs(test_root, 3))
    assert mapping == {
        0: {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6},
        1: {0: 0, 1: 1, 2: 2, 3: 3, 7: 4, 8: 5, 9: 6},
        2: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
    }
