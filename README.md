# PlasRAG

PlasRAG is a deep learning-based tool specifically designed for analyzing plasmids, which serves two purposes: (1) plasmid property characterization, and (2) plasmid DNA sequence retrieval. Users can easily input their interested plasmid sequences as queries. Then, PlasRAG can (1) describe the query plasmids based on predicted properties and information from relevant literature, (2) retrieve eligible plasmids based on selected property queries in Boolean expression form.

### E-mail: yongxinji2-c@my.cityu.edu.hk



## Full command-line options
preprocessing.py:
```
Usage of preprocessing.py:
        [--fasta FASTA] FASTA file of the plasmid DNA sequences to be predicted (either complete sequences or contigs), default: multiple_plasmids.fasta
        [--database DATABASE]   path of the downloaded database folder, which consists of the sequences of PC proteins, MOB/MPF proteins, and replicons, default: database
        [--model_path MODEL_PATH]   path of the folder storing the downloaded or your customized models, default: models
        [--midfolder]   folder to store the intermediate files for prediction, default: temp
        [--len LEN] minimum length of plasmid DNA sequences, default: 1500
        [--threads THREADS] number of threads utilized for preprocessing, default: 2
```
