# CogNet Cognate Data

This directory contains the automatically downloaded CogNet v2.0 dataset for cognate analysis.

## Automatic Download

The system automatically downloads the CogNet v2.0 dataset from:
https://github.com/kbatsuren/CogNet/blob/master/CogNet-v2.0.zip

On first use, the following files will be downloaded and extracted:
- `CogNet-v2.0.tsv` - Complete cognate dataset (~100MB)

## Supported Language Pairs

The system supports cognate analysis for any language pair available in the CogNet dataset, including but not limited to:
- Italian (ita)
- English (eng) 
- Spanish (spa)
- French (fra)

The dataset contains cognate information for many more language pairs beyond these four.

## Dataset Format

The CogNet TSV file contains columns:
- `lang 1`: First language code (3-letter ISO 639-3)
- `word 1`: Word in the first language
- `lang 2`: Second language code (3-letter ISO 639-3)  
- `word 2`: Word in the second language
- `cognacy`: "1" for cognates, "0" for non-cognates

## Example Usage

```python
from core.cognate_lookup import cognate_loader

# Check if two words are cognates
is_cognate = cognate_loader.is_cognate("explain", "explicar", "eng", "spa")

# Find all cognates for a word
cognates = cognate_loader.find_cognates_for_word("animal", "spa", "eng")
```

## Git Ignore

The TSV files are automatically added to `.gitignore` to avoid committing large data files to the repository.

## Data Source

CogNet v2.0 dataset by Khuyagbaatar Batsuren et al.
Original repository: https://github.com/kbatsuren/CogNet

## Note

If the dataset cannot be downloaded automatically, the system will gracefully fall back to frequency-only scoring without cognate boosting.