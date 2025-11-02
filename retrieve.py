import os
import csv
from pyparsing import infixNotation, opAssoc, Word, alphas
import pickle as pkl
import re
import argparse
from collections import defaultdict
import gzip
from Bio import SeqIO

#############################################################
########################  Parameters  #######################
#############################################################
parser = argparse.ArgumentParser(
    description="PlasRAG is a deep learning-based tool specifically designed for analyzing plasmids, which serves two purposes: (1) plasmid property characterization, and (2) plasmid DNA retrieval. Users can easily input their interested plasmid sequences. Then, PlasRAG can (1) describe the query plasmids with predicted properties and information from relevant literature, (2) retrieve eligible plasmids based on input property queries in Boolean expression form.""")
parser.add_argument('--query',
                    help="query boolean expression combined with property IDs and logical operators ('and', 'or', 'not'), e.g., 'CH1000 and (AM3000 or AM3002 or AM3016)', default: 'CH1000'",
                    type=str, default='CH1000')
parser.add_argument('--database', help='path of the PlasRAG database folder, default: ./database', type=str,
                    default='./database')
parser.add_argument('--retrieve_db',
                    help='path of the database folder you want to retrieve the plasmid sequences, default: ./temp',
                    type=str, default='./temp')
parser.add_argument('--output_dir',
                    help='path saved the results, default: ./results',
                    type=str, default='./results')
inputs = parser.parse_args()
db_path = inputs.database
retrieve_db = inputs.retrieve_db
out_fn = inputs.output_dir

if not os.path.exists(out_fn):
    os.makedirs(out_fn)

#############################################################
########################  Help info  ########################
#############################################################
def help_info():
    print('')
    print("""Usage of retrieve.py:
        [--query QUERY] query boolean expression combined with property IDs and logical operators ('and', 'or', 'not'), e.g., 'CH1000 and (AM3000 or AM3002 or AM3016)', default: 'CH1000'
        [--retrieve_db retrieve_db] path of the database folder you want to retrieve the plasmid sequences, default: ./temp
        [--database DATABASE] path of the PlasRAG database folder, default: ./database
        [--output_dir OUTPUT_DIR] path saved the results, default: ./results
""")


#############################################################
####################  Check predictions  ####################
#############################################################
id2pro = pkl.load(open(f'{db_path}/id2pro.dict', 'rb'))
if retrieve_db=="PlasRAG":
    res = pkl.load(open(f'{db_path}/PlasRAG_res.dict', 'rb'))
else:
    res = pkl.load(open(f'{retrieve_db}/res.dict', 'rb'))
items = defaultdict(set)
for pls in res:
    for domain in res[pls]:
        for i, j in res[pls][domain]:
            items[pls].add(id2pro.inv[i])


#############################################################
#################### Analyze expression  ####################
#############################################################
def evaluate_boolean_expression(expression, item_phrases):
    def replace_phrases(match):
        phrase_id = match.group(0)
        return str(phrase_id in item_phrases)

    raw_expression = expression
    expression = re.sub(r'[A-Z]{2}\d{4}', replace_phrases, expression)
    try:
        return eval(expression)
    except Exception as e:
        print(f"Error evaluating expression: {raw_expression}")
        return False


def find_matching_items(boolean_expression, items):
    matching_items = []
    for item_id, item_phrases in items.items():
        if evaluate_boolean_expression(boolean_expression, item_phrases):
            matching_items.append(item_id)
    return matching_items


query = inputs.query
result = find_matching_items(query, items)
if (result != []):
    # result = ', '.join(result)
    # print(f"The eligible plasmids aligning with the query expression '{query}' are: ")
    # for res in result:
    #     print(res)

    target_ids = set(result)

    count = 0  # 统计成功写入的序列数

    if retrieve_db == "PlasRAG":
        output_fasta = f"{out_fn}/align_query_plasmid_from_PlasRAG.fasta"
        output_csv = f"{out_fn}/align_query_plasmid_from_PlasRAG.csv"
        with open(output_fasta, "w") as handle:
            zip_path = f"{db_path}/PlasRAG_plasmids.fasta.gz"
            with gzip.open(zip_path, "rt") as fasta_file:  # "t" 文本模式
                for record in SeqIO.parse(fasta_file, "fasta"):
                    if record.id in target_ids:
                        count += SeqIO.write(record, handle, "fasta")

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows([[item] for item in result])

    else:
        output_fasta = f"{out_fn}/align_query_plasmid_from_own.fasta"
        with open(output_fasta, "w") as handle:
            path = os.path.join(retrieve_db, "split_fasta")
            for fname in os.listdir(path):
                if not fname.endswith(".fasta"):
                    continue
                file_path = os.path.join(path, fname)
                for record in SeqIO.parse(file_path, "fasta"):
                    if record.id in target_ids:
                        count += SeqIO.write(record, handle, "fasta")
        output_csv = f"{out_fn}/align_query_plasmid_from_own.csv"
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows([[item] for item in result])

    print(f"✅ Done! Extracted {count} sequences which satisfy the query to {output_fasta} and {output_csv} ")

else:
    print(f"No eligible plasmid aligning with the query expression '{query}'")
    exit(0)
