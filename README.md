# HMMerge
---
Do you want to merge multiple profile HMMs into one?

## How to Run
---
### The Mapping Generator
```
input_fasta_filenames = ["/path/to/fasta/file/with/all/gap/columns/removed", "/path/to/another/fasta/file/with/all/gap/columns/removed"]
backbone_alignment = "/path/to/backbone/alignment"
mapping = create_mappings_helper(input_fasta_filenames, backbone_alignment)
```
Mapping will be a map of input fasta filename indices to a secondary map

The secondary map is a mapping of column index in the backbone alignment to the corresponding column index in the input fasta filename

For some example input files and expected outputs, look at [the test directories](test/merge_test) and [the test file](test.py)


## Requirements
---
* biopython
* click
* pyhmmer-sepp
* pytest

## Testing
---
```
git clone https://github.com/MinhyukPark/HMMerge.git
cd HMMerge
pytest test.py
```

