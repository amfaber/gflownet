#%%
import gflownet
import gzip
import pickle
import torch
import rdkit.Chem as Chem
import os
import argparse
from pathlib import Path
from futils import ROOT
p = argparse.ArgumentParser()
p.add_argument("--model", default = "results/first_run")
p.add_argument("--output", default = None)
p.add_argument("--device", default = "cpu")
p.add_argument("-n", "--number", default = 1000, type = int)
p.add_argument("--sample-random", action="store_true")
inference_args = p.parse_args()

modelpath = Path(inference_args.model)
if inference_args.output is None:
    inference_args.output = modelpath / "_0/mols_sampled_from_trained"
os.makedirs(inference_args.output, exist_ok = True)
# sys.path.append("/home/qzj517/POR-DD/EquiBind/datasets")
# from multiple_ligands import get_rdkit_coords

#%%
with gzip.open(modelpath / "_0/info.pkl.gz", "rb") as file:
    args = pickle.load(file)["args"]

#%%

dataset = gflownet.Dataset(args, ROOT.ds / "gflownet/mols/data/blocks_PDB_105.json", inference_args.device)

model = gflownet.make_model(args, dataset.mdp)
model.to(inference_args.device)

if inference_args.sample_random:
    dataset.set_sampling_model(model, None)
    sampled_mols = [dataset._get_sample_model()[-1][-2].mol for i in range(inference_args.number)]
    with Chem.SmilesWriter(str(inference_args.output / "sampled_from_random.smi"), includeHeader = False) as w:
        [w.write(mol) for mol in sampled_mols]


with gzip.open(str(modelpath / "_0/params.pkl.gz")) as file:
    params = pickle.load(file)

currparams = model.parameters()
for old, new in zip(currparams, params):
    old.data = torch.tensor(new, dtype = dataset.mdp.floatX, device = inference_args.device)

dataset.set_sampling_model(model, None)

sampled_mols = [dataset._get_sample_model()[-1][-2].mol for i in range(inference_args.number)]
with Chem.SmilesWriter(str(inference_args.output / "sampled.smi"), includeHeader = False) as w:
    [w.write(mol) for mol in sampled_mols]

# %%
