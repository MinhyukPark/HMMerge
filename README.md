# HMMerge
---
Do you want to merge multiple profile HMMs into one?

## How to Run
---
There are 2 ways to run HMMerge depending on the input data available. Regardless of which way is used, the output file will be available at `<Output directory>/HMMerge.aligned.fasta`.


1. Most users fall into this category.
Required Inputs:
- Backbone alignment in FASTA format
- Backbone tree in NEWICK format
- Sequences to align in FASTA format
```
python <git root>/decompose_fasta_version.py --input-tree <Backbone tree> --sequence-file <Backbone alignment> --output-prefix <Output folder for decomposed alignments>/input_ --maximum-size <Decompose size>

pushd <Folder with decomposed alignments>
for f in *.fasta
do
    trimal -in ${f} -out ${f} -noallgaps
done
popd

python <git root>/main.py --input-dir <Folder with decomposed alignments> --backbone-alignment <Backbone alignment> --query-sequence-file <Query sequences> --output-prefix <Output directory> --num-processes <Num cpus> --model {dna|amino}
```

2. No backbone tree, only backbone alignment and query sequences
Required Inputs:
- Backbone alignment in FASTA format
- Sequences to align in FASTA format
- Directory containing decomposed backbone alignment in FASTA format

Note: Make sure that the directory containing decomposed backbone alignment follows these two rules.
a. The files need to follow the naming format input\_<integer starting from 0>.fasta
b. The fasta files must not have columns that are all gaps. If so, use trimal or other equivalent utilities to remove columns with all gaps.

```
python <git root>/main.py --input-dir <Folder with decomposed alignments> --backbone-alignment <Backbone alignment> --query-sequence-file <Query sequences> --output-prefix <Output prefix> --num-processes <num cpus> --model {dna|amino}
```

## Requirements
---
* biopython
* click
* dendropy (if using a backbone tree)
* numpy
* pyhmmer-sepp
* pytest (for testing)
* scipy

## Testing
---
```
git clone https://github.com/MinhyukPark/HMMerge.git
cd HMMerge
pytest test.py
```

