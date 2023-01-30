import click
import dendropy
from Bio import SeqIO

import decomposer

@click.command()
@click.option("--input-tree", required=True, type=click.Path(exists=True), help="The input tree file in newick format")
@click.option("--sequence-file", required=True, type=click.Path(exists=True), help="Aligned sequence file on the full taxa")
@click.option("--output-prefix", required=True, type=str, help="Output file prefix for each subset")
@click.option("--maximum-size", required=True, type=int, help="Maximum size of output subsets")
def decompose_tree(input_tree, sequence_file, output_prefix, maximum_size):
    '''This script decomposes the input tree and outputs induced alignments on the subsets.
    '''
    guide_tree = dendropy.Tree.get(path=input_tree, schema="newick")
    namespace = guide_tree.taxon_namespace
    guide_tree.is_rooted = False
    guide_tree.resolve_polytomies(limit=2)
    guide_tree.collapse_basal_bifurcation()
    guide_tree.update_bipartitions()

    trees = None
    mode="centroid"
    trees = decomposer.decomposeTree(guide_tree, maximum_size, mode=mode)
    clusters = []
    for tree in trees:
        keep = [n.taxon.label.replace("_"," ") for n in tree.leaf_nodes()]
        clusters.append(set(keep))
    print(len(clusters))

    files = [output_prefix + str(i) + ".fasta" for i in range(len(clusters))]
    sequence_partitions = [[] for _ in range(len(clusters))]

    for sequence in SeqIO.parse(open(sequence_file), "fasta"):
        for cluster_index,cluster in enumerate(clusters):
            if(sequence.id.replace("_"," ") in cluster):
                sequence_partitions[cluster_index].append(sequence)

    for sequence_partition_index,sequence_partition in enumerate(sequence_partitions):
        SeqIO.write(sequence_partition, files[sequence_partition_index], "fasta")

if __name__ == "__main__":
    decompose_tree()
