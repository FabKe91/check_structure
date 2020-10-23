#!/bin/env python3

import sys
import logging
import argparse

import numpy as np
import MDAnalysis as mda


######################### DEFINITIONS ###################################

def check_cistrans(A, B, C, D, bondtype=None):
    '''
        Checks whether A is trans to B
        A
         \
          B == C
                \
                 D
        Using cross products CP: In cis (and trans vice versa), (AB x BC) and (BC x CD) must point
                                 to similar directions or at least have an angle of <90 with regard
                                 to the ABC plane.
    '''
    if bondtype not in ["trans", "cis"]:
        print("ERROR: Unknown bondtype: ", bondtype)
        sys.exit()

    AB = A.position - B.position
    BC = B.position - C.position
    CD = C.position - D.position

    c1 = np.cross(AB, BC)
    c2 = np.cross(BC, CD)
    dotp = np.dot(c1, c2)

    if bondtype == "trans" and dotp < 0:
        correct_isomerism = True
    elif bondtype == "cis" and dotp >= 0:
        correct_isomerism = True
    else:
        correct_isomerism = False

    LOGGER.debug("with vectors %s %s %s, and CP %s %s, getting dotp %s, cistrans=%s",
                 AB, BC, CD, c1, c2, dotp, correct_isomerism)

    return correct_isomerism


def check_chirality(A, B, C, D, E):
    '''
        checks whether chirality is correctly set with following configuration:

                A
                |
                |
                B --- E
               / \\
              /   \\
             C     D

        using the plane of ABD, whole C and E must lie at opposite sides of ABD
    '''
    BA = B.position - A.position
    BC = B.position - C.position
    BE = B.position - D.position
    BD = B.position - E.position

    cABD = np.cross(BA, BD)

    dot_C_ABD = np.dot(BC, cABD)
    dot_E_ABD = np.dot(BE, cABD)

    if dot_C_ABD <= 0 and dot_E_ABD >= 0:
        correct_chirality = True
    else:
        correct_chirality = False

    LOGGER.debug("Vectors %s %s with CP %s and vectors %s %s to DPs %s %s",
                  BA, BD, cABD, BC, BE, dot_C_ABD, dot_E_ABD )

    return correct_chirality


def read_mapping_file(mapfilename):
    ''' read martini mapping file and return entries:
            - chiral
            - cis
            - trans
    '''
    output = []
    molecule = None
    with open(mapfilename, "r") as fobj:

        identifier = None

        for line in fobj:

            if ";" in line:
                line, comment = line.split(";")[0], ";" + ''.join(line.split(";")[1:])
            else:
                comment = ""

            if not line.replace("\n", ""):
                continue

            if "[" in line and "]" in line:
                identifier = line.replace("[", "").replace("]", "").strip()
                continue

            if identifier in ["chiral", "cis", "trans"]:
                atomnames = line.split()
                output.append((identifier, atomnames))
            elif identifier == "molecule":
                molecule = line

    return molecule, output


#####################         LOGGER            #########################

LOGGER = logging.getLogger("check_structure")
LOGGER.setLevel("INFO")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel("INFO")
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
LOGGER.addHandler(ch)

fh = logging.FileHandler("check_structure_debug.log")
fh.setLevel("DEBUG")
fh.setFormatter(formatter)



######################### ARGPARSE ARGUMENTS ############################

PARSER = argparse.ArgumentParser()

# Non optional parameters
PARSER.add_argument('-f', action="store", metavar='backmapped.gro', required=True,
                    help="Backmapped structure file")
PARSER.add_argument('-s', action="store", metavar='mol.ff.map', required=True,
                    help="Martini backwards mapping file")
# optional arguments
PARSER.add_argument('-o', action="store", metavar='check.log', nargs='?', required=False,
                    help="output file name", default="structure_check.log")
# On/off flags
PARSER.add_argument('--debug', action="store_true",
                    help="Logs all debug information to check_structure_debug.log")


ARGS = PARSER.parse_args()

STRUCTFNAME    = ARGS.f
OUTPUTFILENAME = ARGS.o
MAPFILENAME    = ARGS.s

if ARGS.debug:
    LOGGER.setLevel("DEBUG")
    LOGGER.addHandler(fh)

if STRUCTFNAME == OUTPUTFILENAME:
    print("ERROR: Input and output file must be named differently")
    sys.exit()

#########################################################################


def main():
    u = mda.Universe(STRUCTFNAME)
    molname, bonds_to_check = read_mapping_file(MAPFILENAME)

    LOGGER.info("Checking chirality and cis/trans bonds for %s", molname)
    LOGGER.debug("Bonds to check are:\n%s", bonds_to_check)

    residues = u.atoms.select_atoms("resname {}".format(molname)).residues
    fails = []

    for residue in residues:
        LOGGER.debug("check residue %s", residue)
        for bondtype, atomnames in bonds_to_check:

            sortdict = {atmn:i for i,atmn in enumerate(atomnames)}
            atoms = residue.atoms.select_atoms("name {}".format(' '.join(atomnames)))
            atoms = sorted(atoms, key=lambda atm: sortdict[atm.name])
            if len(atoms) != len(atomnames):
                print("ERROR: Not all atoms found in system.")
                print("Missing atoms:", set(atomnames)-set([atm.name for atm in atoms]))
                sys.exit()

            LOGGER.debug("at bond %s for atoms %s", bondtype, [atm.name for atm in atoms])

            if bondtype == "chiral":
                correct = check_chirality(*atoms)
            elif bondtype in ["cis", "trans"]:
                correct = check_cistrans(*atoms, bondtype=bondtype)
            else:
                print("ERROR: Bond type {} unknown".format(bondtype))
                sys.exit()
            if not correct:
                LOGGER.debug("appending fail: %s", (residue.resid, bondtype))
                fails.append((residue.resid, bondtype, atoms))

    with open(OUTPUTFILENAME, "w") as outp:
        outp.write("# List of incorrect configurations:\n")
        outp.write("{: <15}{: <10}{: <20}\n".format("bondtype", "resid", "atoms"))
        fails = sorted(fails, key=lambda tup: tup[1])
        #fails = sorted(fails, key=lambda tup: tup[0])
        for resid, bondtype, atoms in fails:
            atoms = ' '.join([atm.name for atm in atoms])
            outp.write("{: <15}{: <10}{: <20}\n".format(bondtype, resid, atoms))

if __name__ == "__main__":
    main()
