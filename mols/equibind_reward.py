import sys
from futils import ROOT
sys.path.append(ROOT.ds / "EquiBind")
sys.path.append(ROOT.ds / "EquiBind/datasets")
import multiligand_inference
import multiple_ligands
from multiple_ligands import get_rdkit_coords
from torch.utils.data import DataLoader
import rdkit.Chem as Chem
import os
import tempfile
import numpy as np
import subprocess
import time
import tempfile
from pathlib import Path
import multiprocessing as mp
from multiple_ligands import get_rdkit_coords
from rdkit.Chem import Descriptors

class RewardDataset(multiple_ligands.Ligands):
    def __init__(self, rec_graph, args, rdkit_seed = None):
        self.rec_graph = rec_graph
        self.args = args
        self.dp = args.dataset_params
        self.use_rdkit_coords = args.use_rdkit_coords
        self.device = args.device
        self.rdkit_seed = rdkit_seed
        self.ligs = None
        self._len = None
        self.skips = None
        self.addH = args.addH
        self.generate_conformer = args.use_rdkit_coords #generate_conformer
        self.processed = None

    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict["pool"]
    #     return self_dict
    @staticmethod
    def embed(lig):
        Chem.AddHs(lig)
        return get_rdkit_coords(lig)
    
    def set_ligs(self, ligs):
        self.ligs = ligs
        self._len = len(self.ligs)
        self.processed = None

    def __getitem__(self, i):
        if self.ligs is None:
            raise IndexError("Attempted indexing before ligands have been added")
        if self.processed is None:
            return super().__getitem__(i)
        else:
            # print("getting from cache")
            return self.processed[i]

class Rewarder:
    def __init__(self, reward_type, arglist, tempdir = True,
        gnina_verbosity = False, default_score = None,
        equibind_save = None, gnina_save = None, use_lipinski = True):

        self.equibind_save = equibind_save
        self.gnina_save = gnina_save
        self.default_score = default_score
        self.gnina_verbosity = gnina_verbosity
        self.use_lipinski = use_lipinski
        args, cmd = multiligand_inference.parse_arguments(arglist)
        args = multiligand_inference.get_default_args(args, cmd)
        args.skip_in_output = False
        if tempdir and args.output_directory is None:
            self.tempdir = tempfile.TemporaryDirectory()
            args.output_directory = self.tempdir.name
        self.args = args
        rec_graph = multiligand_inference.load_rec(args)
        self.dataset = RewardDataset(rec_graph, args)
        self.loader = DataLoader(self.dataset, batch_size = args.batch_size, collate_fn=self.dataset.collate,
                                #  num_workers = 4,
                                #  persistent_workers = True,
                                 )
        self.model = multiligand_inference.load_model(args)
        self.reward_type = reward_type
        self.gnina_in = os.path.join(self.args.output_directory, "output.sdf")
        self.gnina_out = os.path.join(self.args.output_directory, "gnina.sdf")
        device = self.args.device.replace('cuda:', '')
        device = 0 if device == "cpu" else device
        with open(self.gnina_in, "w") as file:
            pass
        self.gnina_cmd_str = f"gnina -r {self.args.rec_pdb} -l {self.gnina_in} -o {self.gnina_out} \
--device {device} --minimize --continuous_operation"
        # print(self.gnina_cmd_str)
        self.start_gnina()
        # os.makedirs(args.output_directory, exist_ok = True)
    
    def start_gnina(self):
        self.subproc = subprocess.Popen(self.gnina_cmd_str.split(" "), 
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        return self.wait_for_gnina()
    
    def wait_for_gnina(self):
        output = ""
        prev_was_blank = False
        n_blanks = 0
        # i = 0
        everything_alright = True
        while output != b'--Chunk finished--\n':
            output = self.subproc.stdout.readline()
            if output == b"":
                prev_was_blank = True
                if prev_was_blank:
                    n_blanks += 1
            else:
                prev_was_blank = False
                n_blanks = 0
            if self.gnina_verbosity:
                print(output)
            if n_blanks > 3:
                everything_alright = False
                break
        return everything_alright
    
    def signal_ready_to_gnina(self):
        self.subproc.stdin.write(b"Ready\n")
        self.subproc.stdin.flush()

    def shutdown_gnina(self):
        self.subproc.stdin.write(b'quit\n')
        try:
            self.subproc.stdin.flush()
        except BrokenPipeError:
            pass
        self.subproc.kill()
        self.subproc.terminate()
    
    def lipinski_score(self, mol):
        if self.use_lipinski:
            violations = (Descriptors.ExactMolWt(mol) > 500) +\
            (Descriptors.MolLogP(mol) > 5) +\
            (Descriptors.NumHDonors(mol) > 5) +\
            (Descriptors.NumHAcceptors(mol) > 10)
            score = max(-violations**2, -8)
            return score
        else:
            return 0


    def get_reward(self, rdmols, pool = None):
        with open(Path(self.args.output_directory) / "failed.txt", "w+") as file:
            pass
        with open(Path(self.args.output_directory) / "success.txt", "w+") as file:
            pass
        self.dataset.set_ligs(rdmols)
        if pool is not None:
            pool.map(RewardDataset.embed, self.dataset.ligs)
        multiligand_inference.write_while_inferring(self.loader, self.model, self.args)
        self.signal_ready_to_gnina()
        ok = self.wait_for_gnina()
        gnina_restarted_times = 0
        while not ok:
            if gnina_restarted_times >= 10:
                break
            print(f"gnina restart number: {gnina_restarted_times}")
            # try:
            self.shutdown_gnina()
            # except BrokenPipeError:
            #     pass
            ok = self.start_gnina()
            if not ok:
                gnina_restarted_times += 1
                continue
            self.signal_ready_to_gnina()
            ok = self.wait_for_gnina()
            gnina_restarted_times += 1
        
        if self.gnina_save:
            with open(self.gnina_save, "a") as total, open(self.gnina_out, "r") as new:
                total.write(new.read())
        
        if self.equibind_save:
            with open(self.equibind_save, "a") as total, open(self.gnina_in, "r") as new:
                total.write(new.read())

        supp = Chem.SDMolSupplier(self.gnina_out)
        mols = [mol for mol in supp]

        prop_names = {"affinity": "CNNaffinity", "vs": "CNN_VS"}
        if not self.reward_type in prop_names:
            raise ValueError("Unknown reward_type")
        rewards = [float(mol.GetProp(prop_names[self.reward_type])) + self.lipinski_score(mol) if mol is not None else None for mol in mols]
        # if self.reward_type == "affinity":
        #     rewards = [float(mol.GetProp("CNNaffinity")) + self.lipinski_score(mol) if mol is not None else None for mol in mols]
        # elif self.reward_type == "vs":
        #     rewards = [float(mol.GetProp("CNN_VS")) + self.lipinski_score(mol) if mol is not None else None for mol in mols]
        # else:
        #     raise ValueError("Unknown reward_type")
        
        with open(os.path.join(self.args.output_directory, "failed.txt")) as file:
            lines = file.readlines()
        idx_of_failed = [int(line.split(" ")[0]) for line in lines]
        for idx in idx_of_failed:
            rewards.insert(idx, None)
        return [reward if reward is not None else self.default_score for reward in rewards]
    
    def __call__(self, rdmols, pool = None):
            return self.get_reward(rdmols, pool)

if __name__ == "__main__":
    mols = [mol for mol in Chem.SDMolSupplier("/home/qzj517/POR-DD/data/raw_data/FDA_drugs/test_3D_opt_1216.sdf")]

    with tempfile.TemporaryDirectory() as tmpdir:
        arglist = [
            "-r", "/home/qzj517/POR-DD/data/raw_data/por_structures/3QE2_1_reduced.pdb",
            "-o", tmpdir
        ]
        rewarder = Rewarder("vs", arglist)
        rewarder.shutdown_gnina()
        # rewarder.start_gnina()
        # rewarder.signal_ready_to_gnina()
        print(rewarder.get_reward(mols[:10]))
        print(rewarder.get_reward(mols[10:20]))
    # print(test)