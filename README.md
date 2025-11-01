# PlasRAG

PlasRAG is a deep learning-based tool specifically designed for analyzing plasmids, which serves __two purposes: (1) property characterization for plasmid DNA sequences, and (2) plasmid DNA sequence retrieval.__ Users can easily input their interested plasmid sequences as queries. Then, PlasRAG can (1) describe the query plasmids based on predicted properties and information from relevant literature, (2) retrieve eligible plasmids based on selected property queries in Boolean expression form.


### E-mail: yongxinji2-c@my.cityu.edu.hk


# Install (Linux or Ubuntu only)
## Dependencies
* [Python 3.x](https://www.python.org/downloads/)
* [NumPy](https://pypi.org/project/numpy/) (pip install numpy==1.25.2)
* [bidict](https://pypi.org/project/bidict/) (pip install bidict)
* [PyTorch](https://pytorch.org/get-started/previous-versions/)>1.8.0
* [Prodigal](https://anaconda.org/bioconda/prodigal) (conda install bioconda::prodigal)
* [biopython](https://pypi.org/project/biopython/) (pip install biopython==1.81)
* [transformers 4.46.1](https://github.com/huggingface/transformers) (pip install transformers==4.46.1)
* [ESM](https://github.com/facebookresearch/esm) (pip install fair-esm)
* [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (pip install bitsandbytes==0.42.0)
* [accelerate](https://github.com/huggingface/accelerate) (pip install accelerate==0.27.2)
* [datasets](https://github.com/huggingface/datasets) (pip install datasets)
* [einops](https://github.com/arogozhnikov/einops) (pip install einops)
* [einops_exts](https://github.com/lucidrains/einops-exts) (pip install einops_exts)
* [pyparsing](https://github.com/pyparsing/pyparsing) (pip install pyparsing)

If you want to use the GPU to accelerate the program:
- CUDA
- PyTorch-GPU
- For CPU version PyTorch: ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```
- For GPU version PyTorch: search [PyTorch](https://pytorch.org/get-started/previous-versions/) to find the correct CUDA version according to your computer
    - For example, in my own server (CUDA 11.3), I installed PyTorch with the Pip command: ```pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu113```


## Prepare the environment
After cloning this repository (```git clone https://github.com/Orin-beep/PlasRAG```), you can use Anaconda to install ```environment.yaml```. This will install all packages you need in GPU mode (make sure you have installed CUDA on your system to use the GPU version; otherwise, PlasRAG will run in CPU mode). The installation command is: 
```
git clone https://github.com/Orin-beep/PlasRAG
cd PlasRAG/
conda env create -f environment.yaml -n plasrag
conda activate plasrag
```
If Anaconda fails to work, you can prepare the environment by individually installing the packages listed in the __Dependencies__ section.


## Download models
### The pretrained 10-faceted PlasRAG models (__required__):
```
wget https://zenodo.org/records/15605555/files/models.tgz
tar zxvf models.tgz
rm models.tgz
```

### The ESM-2 model (esm2_t33_650M_UR50D, __required__):
```
python download_esm.py
mv ~/.cache/torch/hub/checkpoints/ ./esm_models/ 
```

### Optional: LLM Generative Module for Text Summarization and Reasoning.
  
The LLM (Large Language Model) module in PlasRAG is optional.
It enables automatic text summarization and context‑aware question answering, enhancing plasmid characterization reports with natural‑language explanations.
If you have limited GPU memory or do not need textual summaries, you can safely skip this component — the core analytical pipeline is unaffected. We have adopted two potions for using the LLM.

#### Option 1: Use Custom Hugging Face Models (Local Inference):
  
 you can download and load any Hugging Face language model for use in text summarization or reasoning within PlasRAG. For examples:

- Open models (no token needed) — such as Qwen, Mistral, Gemma, etc.
- Gated models — such as Llama 3, which require you to provide your Hugging Face access token.

Run the script with the following syntax:
```
python load_llm.py --model <model_name> [--token <hf_token>]
--model (required): the model name on Hugging Face.
--token (optional): your Hugging Face access token (only needed for gated models).
```

##### Examples
- Open Model (no token required)
```
python load_llm.py --model Qwen/Qwen3-30B-A3B-Instruct-2507
```
- Gated Model (requires Hugging Face token)
```
python load_llm.py --model meta-llama/Llama-3.3-70B-Instruct --token hf_xxxxxxxxxxxxxxxx
```

After the command completes, the model will be downloaded and loaded automatically.
This may take several minutes, depending on the model size and your network speed.

#### Option 2: Use API‑Based LLM Services
If you prefer not to host large models locally, PlasRAG can also connect to API‑based LLM providers (e.g., OpenAI, Anthropic, Google Gemini, etc.).
In this case, users must provide their own API credentials via environment variables. For example:
```
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
export OPENAI_BASE_URL="https://xxxxxx"
```

# Usage
Before employing PlasRAG's two purposes, you should first run ```preprocessing.py``` for your interested plasmid DNA sequences in FASTA format (e.g., 'example_data/test_plasmids.fasta'). 

```
python preprocessing.py --fasta example_data/test_plasmids.fasta --model_path models/ --esm esm_models/esm2_t33_650M_UR50D.pt
```

You can then run ```characterize.py``` for plasmid characterization (describing plasmid properties based on model predictions and literature), or ```retrieve.py``` for plasmid retrieval (filtering eligible plasmids based on your selected queries).

## ```characterize.py```: plasmid property characterization
- __Mode 1__: without LLMs for question answering
    - ```
      python characterize.py --llm False --out results/
      ```
    - The results (plasmid properties predicted by the 10 multi-modal models) will be saved in the 'results/' folder. For example, the characterization result of the plasmid 'NC_005024.1' will be saved in 'results/NC_005024.1.tsv':

| Item | Content |
| ------------- | ------------- |
| __AMR__  | The plasmid encodes ARGs that confer resistance to amikacin, aminoglycoside, aminoglycoside antibiotic, bleomycin, gentamicin, kanamycin, quaternary ammonium, tobramycin. The associated resistance mechanisms include Aminoglycoside Modifying Enzyme, Phosphotransferase. |
| __Virulence Factor__  | The plasmid does not encode any virulence factors. |
| __Metal Resistance__  | The plasmid does not encode any metal resistance genes.  |
| __Host Range__  | The plasmid is hosted by bacteria in the Staphylococcus genus.  |
| __Ecosystem__ | The plasmid can be found in hosts of species Homo sapiens, and ecosystems associated with Birds habitat, Human habitat, Mammal habitat, and engineered, host associated, modeled, simulated communities (contig mixture) ecosystems. |
| __Mobility__  | The plasmid is a conjugative plasmid, which encodes a complete conjugation system and belongs to the mating-pair-formation type MPF_FATA, MPF_T. Additionally, it encodes T4SS ATPase virb4, Type IV coupling protein t4cp2 within the T4SS conjugation system. |
| __Incompatibility Group__ | The plasmid does not belong to any incompatibility groups.  |
| __Risk Index__ | The plasmid has a Combined Minimal risk level, categorized as follows: 1) Moderate risk based on insertion sequences, 2) Low risk based on its distribution across habitats, 3) Minimal based on virulence factor genes, 4) Low based on all encoded ARGs, 5) Minimal risk based on ARGs from WHO priority list, 6) Low risk based on its host range breadth. |
| __Basic Properties__ | The plasmid is a circular, Complete plasmid, sourced from Isolate, and hosted by Gram-positive bacteria.  |
| __Reference plasmid__ | plasmid pSK41 |
| __Literature citation__ | Berg, T, et al. "Complete nucleotide sequence of pSK41: evolution of staphylococcal conjugative multiresistance plasmids." J Bacteriol. 1998 Sep;180(17):4350-9. doi: 10.1128/JB.180.17.4350-4359.1998. |
| __Literature abstract__ | The 46.4-kb nucleotide sequence of pSK41, a prototypical multiresistance plasmid from ... |
- __Mode 2__: with local LLMs for question answering, you should provide the location of your downloaded model, like:
     ```
      python characterize.py --query 'Please summarize key information from the most relevant literature.' --llm True --out results/ --llm_backend local  --model Qwen/Qwen3-30B-A3B-Instruct-2507
   ```
- __Mode 3__: with API LLMs for question answering, you should provide the model name and the API information:
  ```
     python characterize.py --query 'Please summarize key information from the most relevant literature.' --llm True --out results/ --llm_backend api --model gpt-4o-mini
  ```

- The results (response to the query question generated by the Llama-3 model) will be saved in the 'results/' folder. For example, the response for the plasmid 'NC_005024.1' will be saved in 'results/NC_005024.1_ans.tsv':

| Item | Content |
| ------------- | ------------- |
| __Question__ | Please summarize key information from the most relevant literature. |
| __Answer__  | The most relevant literature abstract regarding the plasmid in question is related to the reference plasmid pSK41. pSK41 is a 46.4-kb multiresistance plasmid from Staphylococcus aureus that has been completely sequenced. Analysis of the sequence has provided insights into the evolution of this clinically significant group of plasmids. 1) The plasmid encodes several resistance determinants, including Aminoglycoside Modifying Enzyme and Phosphotransferase, which confer resistance to various antibiotics such as amikacin, aminoglycoside, ... |

## ``` retrieve.py ```: Plasmid DNA Retrieval Module
The retrieval module allows users to search for plasmid sequences that match specific biological properties or combinations of properties.
It supports two retrieval modes:

- Local retrieval — search within your own plasmid sequence collection.
- Database retrieval — search against the plasmid reference database included with PlasRAG.
  
To make querying easier, we provide an ID‑to‑property vocabulary list in
Query_vocabulary.pdf, which maps standardized property IDs (e.g., host range, mobility, virulence factors) to their biological meanings.
Users can combine these IDs into Boolean expressions using logical operators and, or, and not.

### Usage
You can run the retrieval script as follows:
```
python retrieve.py --query "<Boolean_expression>" --database <database_directory>
```
#### Example 1 — Retrieve from Your Own Plasmid Database
Suppose you want to find complete plasmids whose host range is within the Enterobacteriaceae family but that do not encode any adherence‑related virulence factors, from your own custom database. The default my_dir is the temp directory. 
Use the following command:
```
python retrieve.py --query "(not VF1000) and CH2000 and HO4013" 
or
python retrieve.py --query "(not VF1000) and CH2000 and HO4013" --retrieve_db my_dir
```
#### Example 2 — Retrieve from the Built‑in PlasRAG Database
You can also search within the PlasRAG reference plasmid database by simply specifying its name:

```
python retrieve.py --query "(not VF1000) and CH2000 and HO4013" --retrieve_db PlasRAG
``` 
### Output
The matching plasmid IDs will be printed directly in the terminal.
For example:

The eligible plasmids aligning with the query expression '(not VF1000) and CH2000 and HO4013' are the NC_010378.1 sequence.

## Full command-line options
preprocessing.py:
```
Usage of preprocessing.py:
        [--fasta FASTA] FASTA file of the input plasmid DNA sequences (either complete sequences or contigs) to be characterized or retrieved by the PlasRAG tool, default: example_data/test_plasmids.fasta
        [--model_path MODEL_PATH] path of the folder storing the downloaded models, default: models
        [--midfolder MIDFOLDER] folder to store the intermediate files for prediction, default: temp
        [--esm ESM] path of the ESM-2 model (esm2_t33_650M_UR50D.pt), which can be downloaded at: https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt, default: esm_models/esm2_t33_650M_UR50D.pt
        [--batch_size BATCH_SIZE] batch size for prediction, default: 64
        [--database DATABASE] path of the PlasRAG database folder, default: ./database
        [--threads THREADS] number of threads utilized for prediction if 'cpu' is detected ('cuda' not found), default: 8
```

characterize.py
```
Usage of characterize.py:
        [--query QUERY] question or instruction regarding the query plasmids, default: 'Please summarize key information from the most relevant literature.'
        [--out OUT] path to store the prediction results, default: results
        [--llm LLM] whether to enable LLM for result summarization and question answering, default: 'True'
        [--llm_backend LLM_BACKEND] choose llm backend: 'local' or 'api'
        [--model Model] the downloaded model ID, default: 'Qwen/Qwen3-30B-A3B-Instruct-2507'
        [--midfolder MIDFOLDER] the intermediate folder generated by preprocessing.py, default: ./temp
        [--database DATABASE] path of the PlasRAG database folder, default: ./database
        [--threads THREADS] number of threads utilized for prediction if 'cpu' is detected ('cuda' not found), default: 8
```

retrieve.py
```
Usage of retrieve.py:
        [--query QUERY] query boolean expression combined with property IDs and logical operators ('and', 'or', 'not'), e.g., 'CH1000 and (AM3000 or AM3002 or AM3016)', default: 'CH1000'
        [--retrieve_db RETRIEVE_DB] path of the database folder you want to retrieve the plasmid sequences, default: ./temp
        [--database DATABASE] path of the PlasRAG database folder, default: ./database
        [--output_dir OUTPUT_DIR] path saved the results, default: ./temp
```
