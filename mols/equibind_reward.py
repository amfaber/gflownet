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

class RewardDataset(multiple_ligands.Ligands):
    def __init__(self, rec_graph, args, generate_conformer = False, addH = False, rdkit_seed = None):
        self.rec_graph = rec_graph
        self.args = args
        self.dp = args.dataset_params
        self.use_rdkit_coords = args.use_rdkit_coords
        self.device = args.device
        self.rdkit_seed = rdkit_seed
        self.ligs = None
        self._len = None
        self.skips = None
        
        # if addH is None:
        #     addH = True
        self.addH = addH
        
        self.generate_conformer = generate_conformer

    def set_ligs(self, ligs):
        self.ligs = ligs
        self._len = len(self.ligs)

    def __getitem__(self, i):
        if self.ligs is None:
            raise IndexError("Attempted indexing before ligands have been added")
        return super().__getitem__(i)

class Rewarder:
    def __init__(self, reward_type, arglist):
        args, cmd = multiligand_inference.parse_arguments(arglist)
        args = multiligand_inference.get_default_args(args, cmd)
        args.skip_in_output = False
        self.args = args
        rec_graph = multiligand_inference.load_rec(args)
        self.dataset = RewardDataset(rec_graph, args)
        self.loader = DataLoader(self.dataset, batch_size = args.batch_size, collate_fn=self.dataset.collate,
                                #  num_workers = 4,
                                #  persistent_workers = True,
                                 )
        self.model = multiligand_inference.load_model(args)
        self.reward_type = reward_type
        gnina_in = os.path.join(self.args.output_directory, "output.sdf")
        self.gnina_out = os.path.join(self.args.output_directory, "gnina.sdf")
        device = self.args.device.replace('cuda:', '')
        device = 0 if device == "cpu" else device
        with open(gnina_in, "w") as file:
            pass
        self.gnina_cmd_str = f"gnina -r {self.args.rec_pdb} -l {gnina_in} -o {self.gnina_out} \
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
        self.wait_for_gnina()
    
    def wait_for_gnina(self, verbose = False):
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
            if verbose:
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
        self.subproc.stdin.flush()
        self.subproc.kill()
        self.subproc.terminate()
    
    def get_reward(self, rdmols):
        self.dataset.set_ligs(rdmols)
        multiligand_inference.write_while_inferring(self.loader, self.model, self.args)
        self.signal_ready_to_gnina()
        verbose = False
        ok = self.wait_for_gnina(verbose)
        gnina_restarted_times = 0
        while not ok:
            if gnina_restarted_times >= 2:
                break
            print("Had to restart gnina")
            self.shutdown_gnina()
            self.start_gnina()
            self.signal_ready_to_gnina()
            ok = self.wait_for_gnina(verbose)
            gnina_restarted_times += 1
        
        # print("gnina output ", os.path.exists(self.gnina_out))
        # print("equi output ", os.path.exists(self.args.output_directory))
        # with open(self.gnina_out) as file:
        #     print(file.read())
        supp = Chem.SDMolSupplier(self.gnina_out)
        mols = [mol for mol in supp]
        # cnnvar = [float(mol.GetProp("CNNaffinity_variance")) for mol in mols]
        if self.reward_type == "affinity":
            rewards = [float(mol.GetProp("CNNaffinity")) for mol in mols]
        elif self.reward_type == "vs":
            # try:
            rewards = [float(mol.GetProp("CNN_VS")) for mol in mols]
            # except AttributeError:
            #     print("We got problems")
            #     print(self.gnina_out)
            #     self.shutdown_gnina()
            #     time.sleep(100000)
            # except KeyError:
            #     print("We got problems")
            #     print(self.gnina_out)
            #     self.shutdown_gnina()
            #     time.sleep(100000)
        else:
            raise ValueError("Unknown reward_type")
        
        with open(os.path.join(self.args.output_directory, "failed.txt")) as file:
            lines = file.readlines()
        idx_of_failed = [line.split(" ")[0] for line in lines]
        for idx in idx_of_failed:
            rewards.insert(idx, None)
        return rewards
    
    def __call__(self, rdmols):
        return self.get_reward(rdmols)

if __name__ == "__main__":
    mols = [mol for mol in Chem.SDMolSupplier("/home/qzj517/POR-DD/data/raw_data/cyp_screen/test_3D_opt_1216.sdf")]

    with tempfile.TemporaryDirectory() as tmpdir:
        arglist = [
            "-r", "/home/qzj517/POR-DD/data/raw_data/por_structures/3ES9_1_reduced.pdb",
            "-o", tmpdir
        ]
        rewarder = Rewarder("vs", arglist)
        print(rewarder.get_reward(mols[:10]))
        print(rewarder.get_reward(mols[10:20]))
    # print(test)