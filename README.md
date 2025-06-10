# PlasRAG

PlasRAG is a deep learning-based tool specifically designed for analyzing plasmids, which serves two purposes: (1) plasmid property characterization, and (2) plasmid DNA sequence retrieval. Users can easily input their interested plasmid sequences as queries. Then, PlasRAG can (1) describe the query plasmids based on predicted properties and information from relevant literature, (2) retrieve eligible plasmids based on selected property queries in Boolean expression form.


### E-mail: yongxinji2-c@my.cityu.edu.hk


# Install (Linux or Ubuntu only)
## Dependencies
* [Python 3.x](https://www.python.org/downloads/)
* [NumPy](https://pypi.org/project/numpy/) (pip install numpy==1.25.2)
* [bidict](https://pypi.org/project/bidict/) (pip install bidict)
* [PyTorch](https://pytorch.org/get-started/previous-versions/)>1.8.0
* [Prodigal](https://anaconda.org/bioconda/prodigal) (conda install prodigal)
* [biopython](https://pypi.org/project/biopython/) (pip install biopython==1.81)
* [transformers 4.46.1](https://github.com/huggingface/transformers) (pip install transformers==4.46.1)
* [ESM](https://github.com/facebookresearch/esm) (pip install fair-esm)
* [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (pip install bitsandbytes==0.42.0)
* [accelerate](https://github.com/huggingface/accelerate) (pip install accelerate==0.27.2)

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
- The pretrained 10-faceted PlasRAG models (__required__):
```
wget https://zenodo.org/records/15605555/files/models.tgz
tar zxvf models.tgz
rm models.tgz
```

- The ESM-2 model (esm2_t33_650M_UR50D, __required__):
```
python download_esm.py
mv ~/.cache/torch/hub/checkpoints/ ./esm_models/ 
```

- The Llama-3 generative model (__optional__):
    - If you have a powerful GPU, we recommend downloading the Llama-3 model for text summarization and question answering to support plasmid characterization. Depending on your GPU resources, you can choose to download one of the following two Llama-3 models:
    - The __lightweight__ [Llama-3.2-3B-Instruct model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct):
      ```
      python download_llama_3b.py
      mv ~/.cache/torch/hub/checkpoints/ ./esm_models/ 
      ```

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
        [--llama LLAMA] the downloaded Llama3 model id, default: 'meta-llama/Llama-3.2-3B-Instruct'
        [--quantize QUANTIZE] whether to load the Llama-3 model in 8-bit or 4-bit, which can largely decrease memory usage ('False', '8bit', or '4bit'), default: '8bit'
        [--midfolder MIDFOLDER] the intermediate folder generated by preprocessing.py, default: ./temp
        [--database DATABASE] path of the PlasRAG database folder, default: ./database
        [--threads THREADS] number of threads utilized for prediction if 'cpu' is detected ('cuda' not found), default: 8
```

retrieve.py
```
Usage of retrieve.py:
        [--query QUERY] query boolean expression combined with property IDs and logical operators ('and', 'or', 'not'), e.g., 'CH1000 and (AM3000 or AM3002 or AM3016)', default: 'CH1000'
        [--midfolder MIDFOLDER] the intermediate folder generated by preprocessing.py, default: ./temp
        [--database DATABASE] path of the PlasRAG database folder, default: ./database
```
