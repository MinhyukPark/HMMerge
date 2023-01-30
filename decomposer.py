'''
Created on Oct 7, 2019

@author: Vlad
modified by Min (minpark815@gmail.com)
'''
import sys

import dendropy
from dendropy.utility import bitprocessing


def decomposeTree(tree, max_subset_size, mode):
    num_leaves = len(tree.leaf_nodes())
    return decompose_tree_helper(num_leaves, tree, max_subset_size, mode)

def decompose_tree_helper(num_taxa, tree, max_subset_size, mode):
    numLeaves = len(tree.leaf_nodes())
    if(mode != "centroid"):
        sys.exit("Non-centroid decomposition not supported")
    if numLeaves > max_subset_size:
        e = getCentroidEdge(tree)
        t1, t2 = bipartitionByEdge(tree, e)
        return decompose_tree_helper(num_taxa, t1, max_subset_size, mode) + decompose_tree_helper(num_taxa, t2, max_subset_size, mode)
    elif numLeaves >= 1:
        return [tree]
    else:
        sys.exit("tree has fewer than 1 leaves!")

def bipartitionByEdge(tree, edge):
    newRoot = edge.head_node
    edge.tail_node.remove_child(newRoot)
    newTree = dendropy.Tree(seed_node=newRoot, taxon_namespace = tree.taxon_namespace)
    tree.update_bipartitions()
    newTree.update_bipartitions()
    return tree, newTree

def getCentroidEdge(tree):
    numLeaves = bitprocessing.num_set_bits(tree.seed_node.tree_leafset_bitmask)
    bestBalance = float('inf')
    for edge in tree.postorder_edge_iter():
        if edge.tail_node is None:
            continue
        balance = abs(numLeaves/2 - bitprocessing.num_set_bits(edge.bipartition.leafset_bitmask))
        if balance < bestBalance:
            bestBalance = balance
            bestEdge = edge
    return bestEdge
