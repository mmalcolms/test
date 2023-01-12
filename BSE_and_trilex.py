# Calculation of the DMFT charge and spin lattice susceptibility and the lattice trilex vertex
# via the Bethe-Salpeter equation.
import triqs.utility.mpi as mpi
import numpy as np
import os
from sys import exit
from h5 import *
import gc
import ladderdgammaa.pytools as pytools
from ladderdgammaa.dse import chiqw_and_trilex_vertex_from_chi0q_and_gamma_PH
U    = 2.0
beta = 15.0
nk   = 50
Niwb = 20

parent_dir  = "/home/deoliveira/Desktop/BSE_test"
direc_imp   = "2p_imp_qnts-U%.2f-b%.2f-niwb%s.h5"%(U, beta, Niwb)
input_imp   = os.path.join(parent_dir, direc_imp)

with HDFArchive(input_imp, "r") as A:
    #Gamma_ch_imp = A["Gamma_ch_imp"]
    Gamma_sp_imp = A["Gamma_sp_imp"]

fileBubble  = "lattice_qnts-b%.2f-nk%s.h5"%(beta,nk)
inputBubble = os.path.join(parent_dir, fileBubble)
output_file = os.path.join(parent_dir, fileBubble)

with HDFArchive(inputBubble, 'r') as A:
    chi0_wnq_sp = A["chi0_wnq_sp"]
    #chi0_wnq_ch = A["chi0_wnq_ch"]

# Calculation of DMFT lattice spin susceptibility (BSE) and trilex vertex gamma(i\omega, i\nu, q)
"""
chi_ch_qW, trilex_ch_Wnq = chiqw_and_trilex_vertex_from_chi0q_and_gamma_PH(chi0_wnq_ch,\
                                                                           Gamma_ch_imp,  U)
if mpi.is_master_node():
    with HDFArchive(output_file, "a") as A:
        A["chi_ch_qW"]     = chi_ch_qW
        A["trilex_ch_Wnq"] = trilex_ch_Wnq

del Gamma_ch_imp, chi0_wnq_ch, chi_ch_qW, trilex_ch_Wnq
gc.collect()
"""
chi_sp_qW, trilex_sp_Wnq = chiqw_and_trilex_vertex_from_chi0q_and_gamma_PH(chi0_wnq_sp,\
                                                                           Gamma_sp_imp, -U)

if mpi.is_master_node():
    with HDFArchive(output_file, "a") as A:
        A["chi_sp_qW"]     = chi_sp_qW
        A["trilex_sp_Wnq"] = trilex_sp_Wnq
del Gamma_sp_imp, chi0_wnq_sp, chi_sp_qW, trilex_sp_Wnq
gc.collect()
mpi.barrier()
