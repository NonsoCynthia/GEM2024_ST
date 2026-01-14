# GEM2024_ST

DCU-ADAPT-modPB at the GEM’24 Data-to-Text Generation Task: Model Hybridisation for Pipeline Data-to-Text Natural Language Generation  
https://aclanthology.org/2024.inlg-genchal.7/

## Summary

This work describes our system for the GEM 2024 multilingual data-to-text shared task, where the input is a set of WebNLG RDF triples and the output is a natural language text in multiple languages, including low-resource languages. We combine end-to-end and pipeline neural approaches for English generation, then extend to Hindi, Korean, Arabic, and Swahili using a neural machine translation step.

## Contents

- Core scripts and configuration for the shared task experiments
- Prompts and templates used for generation and post-processing (where applicable)
- Translation utilities for non-English runs (where applicable)
- Evaluation scripts plus saved outputs and score summaries
- `data/` and `results/` folders containing inputs, generations

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use API-based models, set your key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

## Run

Check the `job/` folder and run the scripts used to produce generations and evaluations.

## Citation

```bibtex
@inproceedings{osuji-etal-2024-dcu,
    title = "{DCU}-{ADAPT}-mod{PB} at the {GEM}{'}24 Data-to-Text Generation Task: Model Hybridisation for Pipeline Data-to-Text Natural Language Generation",
    author = "Osuji, Chinonso Cynthia  and
      Huidrom, Rudali  and
      Adebayo, Kolawole John  and
      Castro Ferreira, Thiago  and
      Davis, Brian",
    editor = "Mille, Simon  and
      Clinciu, Miruna-Adriana",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference: Generation Challenges",
    month = sep,
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-genchal.7/",
    doi = "10.18653/v1/2024.inlg-genchal.7",
    pages = "66--75",
}
```
