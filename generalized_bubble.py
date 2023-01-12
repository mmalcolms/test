from triqs.gf import Gf, BlockGf, Block2Gf, Idx
from triqs.lattice import BravaisLattice, BrillouinZone
from h5 import HDFArchive
import triqs.utility.mpi as mpi
from triqs.gf import MeshBrillouinZone, MeshImFreq, MeshProduct, MeshCyclicLattice
import numpy as np
import os
from sys import exit
from h5 import *
import gc

from triqs_tprf.lattice import lattice_dyson_g_wk
from triqs_tprf.bse import get_chi0_wnk

U    = 2.0
beta = 15.0
nk_range = [50]
Niwb =  20
parent_dir  = "/home/deoliveira/Desktop/BSE_test"
direc_imp   = "2p_imp_qnts-U%.2f-b%.2f-niwb%s.h5"%(U, beta, Niwb)

input_imp   = os.path.join(parent_dir, direc_imp)

with HDFArchive(input_imp, "r") as A:
    Sigma_imp  = A["Sigma_imp"]
    chi_ch_imp = A["chi_ch_imp"]
    chi_sp_imp = A["chi_sp_imp"]

mu_dmft = U/2.

nwf_sp = len(chi_sp_imp.mesh[1]) // 2
nW_sp = (len(chi_sp_imp.mesh[0]) + 1) // 2

nwf_ch = len(chi_ch_imp.mesh[1]) // 2
nW_ch = (len(chi_ch_imp.mesh[0]) + 1) // 2

del chi_ch_imp, chi_sp_imp
gc.collect()


for nk in nk_range:
    direc_disp  = "dispersion-nk%s.h5"%nk
    input_disp  = os.path.join(parent_dir, direc_disp)
    direc_latt  = "lattice_qnts-b%.2f-nk%s.h5"%(beta,nk)
    output_file =  os.path.join(parent_dir, direc_latt)


    with HDFArchive(input_disp, "r") as A:
        eps_k = A['dispersion']

    print("Lattice DMFT Green's function")
    # DMFT lattice Green's function
    G_wk = lattice_dyson_g_wk(mu=mu_dmft, e_k=eps_k, sigma_w=Sigma_imp)
    with HDFArchive(output_file, 'a') as A:
        A["G_wk-beta%.2f"%beta]          = G_wk


    if nW_sp == nW_ch:
        chi0_wnq = get_chi0_wnk(G_wk, nw=nW_sp, nwf=nwf_sp)
        with HDFArchive(output_file, 'a') as A:
            A["chi0_wnq_sp"]   = chi0_wnq
            A["chi0_wnq_ch"]   = chi0_wnq
    else:
        chi0_wnq_sp = get_chi0_wnk(G_wk, nw=nW_sp, nwf=nwf_sp)
        with HDFArchive(output_file, 'a') as A:
            A["chi0_wnq_sp"]   = chi0_wnq_sp
        del chi0_wnq_sp
        gc.collect()

        chi0_wnq_ch = get_chi0_wnk(G_wk, nw=nW_ch, nwf=nwf_ch)

        with HDFArchive(output_file, 'a') as A:
            A["chi0_wnq_ch"]   = chi0_wnq_ch

        del chi0_wnq_ch
        gc.collect()
