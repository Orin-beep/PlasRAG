import argparse
from datasets import load_dataset
import os, sys, subprocess
from collections import defaultdict
import numpy as np
from esm import pretrained, MSATransformer, FastaBatchedDataset
from model import retriever, retrieverConfig, get_vram
from transformers import Trainer, TrainingArguments
import pickle as pkl
import time
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import torch
start = time.time()


#############################################################
########################  Parameters  #######################
#############################################################
parser = argparse.ArgumentParser(description="""PlasRAG is a deep learning-based tool specifically designed for analyzing plasmids, which serves two purposes: (1) plasmid property characterization, and (2) plasmid DNA retrieval. Users can easily input their interested plasmid sequences. Then, PlasRAG can (1) describe the query plasmids with predicted properties and information from relevant literature, (2) retrieve eligible plasmids based on input property queries in Boolean expression form.""")
parser.add_argument('--fasta', help='FASTA file of the input plasmid DNA sequences (either complete sequences or contigs) to be characterized or retrieved by the PlasRAG tool, default: example_data/test_plasmids.fasta', default = 'example_data/test_plasmids.fasta')
parser.add_argument('--model_path', help='path of the folder storing the downloaded models, default: models', type=str, default='models')
parser.add_argument('--esm', help="Path of the ESM-2 model (esm2_t33_650M_UR50D.pt), which can be downloaded with the 'download_esm.py' script, default: esm_models/esm2_t33_650M_UR50D.pt", default = 'esm_models/esm2_t33_650M_UR50D.pt')
parser.add_argument('--midfolder', help='folder to store the intermediate files for prediction, default: temp', type=str, default='temp')
parser.add_argument('--batch_size', help="batch size for prediction, default: 64", type=int, default=64)
parser.add_argument('--database', help='path of the PlasRAG database folder, default: ./database', type=str, default='./database')
parser.add_argument('--threads', help="number of threads utilized for prediction if 'cpu' is detected ('cuda' not found), default: 8", type=int, default=8)
inputs = parser.parse_args()


#############################################################
########################  Help info  ########################
#############################################################
def help_info():
    print('')
    print("""Usage of preprocessing.py:
        [--fasta FASTA] FASTA file of the input plasmid DNA sequences (either complete sequences or contigs) to be characterized or retrieved by the PlasRAG tool, default: example_data/test_plasmids.fasta
        [--model_path MODEL_PATH] path of the folder storing the downloaded models, default: models
        [--midfolder MIDFOLDER] folder to store the intermediate files for prediction, default: temp
        [--esm ESM] path of the ESM-2 model (esm2_t33_650M_UR50D.pt), which can be downloaded at: https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt, default: esm_models/esm2_t33_650M_UR50D.pt
        [--batch_size BATCH_SIZE] batch size for prediction, default: 64
        [--database DATABASE] path of the PlasRAG database folder, default: ./database
        [--threads THREADS] number of threads utilized for prediction if 'cpu' is detected ('cuda' not found), default: 8
""")


#############################################################
#########################  Prodigal  ########################
#############################################################
out_fn = inputs.midfolder
if not os.path.isdir(out_fn):
    os.makedirs(out_fn)
fasta_path = inputs.fasta
raw_plasmids = set()
for s in SeqIO.parse(fasta_path, 'fasta'):
    raw_plasmids.add(s.id)
print("Running Prodigal ...")
prodigal_cmd = f'prodigal -i {fasta_path} -a {out_fn}/plasmids.faa -f gff -p meta'
_ = subprocess.check_call(prodigal_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# read translated proteins
input_ids = defaultdict(list)
protein_idx = []
idx = 1
proteins = []
for s in SeqIO.parse(f'{out_fn}/plasmids.faa', 'fasta'):
    pls = s.id[:s.id.rfind('_')]
    protein_idx.append(s.id)
    if(len(input_ids[pls])<300):
        input_ids[pls].append(str(idx))
    idx+=1
    rec = SeqRecord(Seq(str(s.seq)[:-1]), id=s.id, description='')
    proteins.append(rec)
SeqIO.write(proteins, f'{out_fn}/plasmids.faa', 'fasta')
discarded = raw_plasmids - set(input_ids)
if(len(discarded)>1):
    discarded = ', '.join(list(discarded))
    print(f'{len(discarded)} plasmids have been discarded because they are too short for gene translation: {discarded}.')
elif(len(discarded)==1):
    discarded = ', '.join(list(discarded))
    print(f'1 plasmid has been discarded because it is too short for gene translation: {discarded}.')


#############################################################
#################  ESM-2 protein embedding  #################
#############################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == torch.device('cpu'):
    torch.set_num_threads(inputs.threads)    
    print(f"Running with CPU {inputs.threads} threads ...")
else:
    print(f"Running with GPU ...")
esm_path = inputs.esm
model, alphabet = pretrained.load_model_and_alphabet(esm_path)
model.eval()
if isinstance(model, MSATransformer):
    raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
#model = model.cuda()
model = model.to(device)
truncation_seq_length = 1022
toks_per_batch = 4096
dataset = FastaBatchedDataset.from_file(f'{out_fn}/plasmids.faa')
batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
data_loader = torch.utils.data.DataLoader(
    dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches)

return_contacts = False
repr_layers = [33]
assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

esm_dict = {}
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        toks = toks.to(device="cuda", non_blocking=True)
        out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

        representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
        representations = representations[33]
        for i, label in enumerate(labels):
            truncate_len = min(truncation_seq_length, len(strs[i]))
            res = representations[i, 1 : truncate_len + 1].mean(0).clone().numpy()
            esm_dict[label] = res

esm_embed = []
for prot in protein_idx:
    esm_embed.append(torch.from_numpy(esm_dict[prot]))
esm_embed = torch.stack(esm_embed)
zero_row = torch.zeros(1, 1280, dtype=torch.float32)
esm_embed = torch.cat((zero_row, esm_embed), dim=0)
#torch.save(esm_embed, f'{out_fn}/esm_embed.pt')


#############################################################
#########################  Dataset  #########################
#############################################################
plasmids = sorted(list(input_ids))
f=open(f'{out_fn}/plasmids.csv', 'w')
f.write(f'plasmid_id,proteins\n')
for plasmid in plasmids:
    tmp = ';'.join(input_ids[plasmid])
    f.write(f'{plasmid},{tmp}\n')
f.close()


#############################################################
######################  Run the model  ######################
#############################################################
batch_size = inputs.batch_size
plasmid_max_length = 300
#esm_path = f'{out_fn}/esm_embed.pt'
test_file = f'plasmids.csv'
data_files = {'test': test_file}
dataset = load_dataset(out_fn, data_files=data_files)

# preprocessing the input data
def preprocess_data(examples):
    encoding = {}
    # protein idxes
    proteins = examples["proteins"]
    protein_idxes = []
    for sentence in proteins:
        prots = sentence.split(';')
        prots = list(map(int, prots))
        prots += [0]*(plasmid_max_length-len(prots))
        protein_idxes.append(prots)
    encoding['proteins'] = protein_idxes
    return encoding

# preprocess dataset
dataset = dataset.map(preprocess_data, batched=True, num_proc=inputs.threads, remove_columns=dataset['test'].column_names, keep_in_memory=True)
dataset.set_format("torch")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def return_risk(p):
    # Minimal: 0-0.3, Low: 0.3-0.5, Moderate: 0.5-0.7, High: 0.7-1
    if(p<=0.3):
        return 'Minimal risk '
    elif(p<=0.5):
        return 'Low risk '
    elif(p<=0.7):
        return 'Moderate risk '
    else:
        return 'High risk '

def check_complete(res):
    for i,j in res:
        if(i=='Complete plasmid'):
            return 1
    return 0

weight = torch.tensor([0.15, 0.1, 0.15, 0.05, 0.05, 0.05, 0.1, 0.15, 0.05, 0.15])
model_path = inputs.model_path
vocab_size_dict = {'amr':124, 'host':199, 'eco':58, 'ecohost':43, 'vf':205, 'risk':6, 'mob':41, 'inc':59, 'metal':231, 'char':15}
db_path = inputs.database
embeddings = []
res_dict = defaultdict(dict)
for domain in ['amr', 'vf', 'host', 'ecohost', 'eco', 'char', 'metal', 'inc', 'risk', 'mob']:
    if(domain=='amr'):
        prob_cutoff=0.4
    elif(domain=='eco'):
        prob_cutoff=0.6
    else:
        prob_cutoff=0.5
    vocab = pkl.load(open(f'{db_path}/vocab_list/{domain}.list', 'rb'))
    OUTPUT_dir = f'{model_path}/{domain}/'
    vocab_size = vocab_size_dict[domain]
    text_embed_path = f'{db_path}/text_embed/text_embed_{domain}.pt'
    # load model
    config = retrieverConfig()
    model = retriever.from_pretrained(
        OUTPUT_dir,
        config=config,
        raw_embed=esm_embed,
        tokenizer=None,
        vocab=vocab_size,
        max_length=-1,
        text_embed_path=text_embed_path,
        device=device,
        domain=domain).to(device)
    #print(domain, model.num_parameters())
    #get_vram()

    test_args = TrainingArguments(
        output_dir = inputs.midfolder,
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = batch_size,
        dataloader_drop_last = False
    )

    # init trainer
    trainer = Trainer(
        model = model,
        args = test_args,#compute_metrics = compute_metrics
    )

    test_results = trainer.predict(dataset['test'])
    logits, embed = test_results[0]
    probs = sigmoid(logits)
    embeddings.append(embed)
    
    if(domain=='risk'):
        idx = 0
        for prob in probs:
            pls = plasmids[idx]
            idx+=1
            # Minimal: 0-0.3, Low: 0.3-0.5, Moderate: 0.5-0.7, High: 0.7-1
            res = []
            for j in range(6):
                risk = vocab[j]
                x = risk[11:]
                p = prob[j]
                x = return_risk(p)+x 
                res.append((x, p))

            combine = np.mean(prob)
            #combine = (sum(prob)+prob[4])/7
            level = return_risk(combine)
            res.append((f'Combined {level}level', combine))
            res_dict[pls][domain] = res
        continue
   
    if(domain=='vf'):
        new = []
        for i in vocab:
            if('Virulence factor' in i):
                new.append(i[17:])
            else:
                new.append(i)
        vocab = new

    # Besides risk
    idx = 0 
    for prob in probs:
        pls = plasmids[idx]
        idx+=1
        indexes = np.where(prob >= prob_cutoff)[0]
        res = []
        for i in indexes:
            if(domain=='char' and vocab[i]=='Direct terminal repeat' and not check_complete(res)):
                res.append((vocab[i], prob[i]))
                res.append(('Complete plasmid', prob[i]))
            else:
                res.append((vocab[i], prob[i]))
        res_dict[pls][domain] = res

pkl.dump(res_dict, open(f'{out_fn}/res.dict', 'wb'))

# final plasmid embeddings (plas_num * 512), order: "plasmids"
query_embed = sum(w * t for w, t in zip(weight, embeddings))
#torch.save(embeddings, f'{out_fn}/embeddings.pt')

# pinpoint closest reference plasmids
reference_embed = torch.load(f'{db_path}/reference_embed.pt')
ref_list = pkl.load(open(f'{db_path}/reference.list', 'rb'))
ref2info = pkl.load(open(f'{db_path}/ref2info.dict', 'rb'))
distances = torch.norm(reference_embed.unsqueeze(1)-query_embed.unsqueeze(0), dim=2)
closest_indices = torch.argmin(distances, dim=0)
idx = 0
query2info = {}
for i in closest_indices:
    pls = plasmids[idx]
    idx+=1
    ref = ref_list[i]
    info = ref2info[ref]
    query2info[pls] = info
pkl.dump(query2info, open(f'{out_fn}/pub.dict', 'wb'))

end = time.time()
print(f'Total time for running preprocessing.py is {end-start}s.')
