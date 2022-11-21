from re import A
from futils import gflownet_utils
from futils import ROOT
import argparse
def parseargs(arglist = None):
    p = argparse.ArgumentParser()
    
    p.add_argument("input")
    p.add_argument("-r", dest = "receptor")

    return p.parse_args(arglist)


if __name__ == "__main__":
    args = parseargs()
    if args.receptor is None:
        args.receptor = ROOT.ds / "data/raw_data/por_structures/3QE2_1_reduced.pdb"
    print(args)
    gflownet_utils.characterize(args.input, por_structure = args.receptor)
