r"""DESCRIPTION:
In this script all the impurity vertices are calculated based on the usable bosonic frequency range
of the charge and spin impurity susceptibility.
r"""

from triqs.gf import Gf, BlockGf, Block2Gf, Idx
from triqs.lattice import BravaisLattice, BrillouinZone
from h5 import HDFArchive
import triqs.utility.mpi as mpi
from triqs.gf import MeshBrillouinZone, MeshImFreq, MeshProduct, MeshCyclicLattice
import numpy as np
import os
from sys import exit
from h5 import *


from triqs_tprf.linalg import inverse_PH
from triqs_tprf.tight_binding import TBLattice
from triqs_tprf.chi_from_gg2 import chi0_from_gg2_PH
import ladderdgammaa.pytools as pytools
from ladderdgammaa.dse import sum_iw_iw, make_f_channel_from_chi
from ladderdgammaa.dse import make_chi_usable_frequencies, make_tensor_valued_chi

from ladderdgammaa.dse import sum_iw_iw, make_chi_usable_frequencies, make_tensor_valued_chi
from triqs_ctint.post_process import chi_tilde_ph_from_G2c
beta_range = [15]
Niwb       = 20
measure    = 0
nden       = 1
U          = 2.0
parent_dir  = "/home/deoliveira/Desktop/BSE_test"
for j in range(len(beta_range)):
    beta = beta_range[j]
    direc_dmft =  "dmft_U%.1f_b%.1f_n1.0.h5"%(U, beta)
    direc_2pGf = "G2c_ph_imp-U%.2f-b%.2f-measure0.h5"%(U, beta)
    direc_out  = "2p_imp_qnts-U%.2f-b%.2f-niwb%s.h5"%(U, beta, Niwb)

    input_dmft  = os.path.join(parent_dir, direc_dmft)
    input_G2c   = os.path.join(parent_dir, direc_2pGf)
    output_file = os.path.join(parent_dir, direc_out)

    with HDFArchive(input_dmft, "r") as A:
        loop_input = max([int(k.replace('G_local-', '')) for k in [k for k in list(A.keys()) if 'G_local-' in k]])
        Sigma_imp = A['Sigma_param-%i' %loop_input]
        G_imp     =A['G_local-%i' %loop_input]

    G_imp["up"]   = 0.5 * (G_imp["up"] + G_imp["down"])
    G_imp["down"] = G_imp["up"]

    with HDFArchive(input_G2c, 'r') as A:
        G2c_Wnnp = A['G2c_iw']
    G2c_Wnnp["up","up"]     = 0.5 * (G2c_Wnnp["up","up"] + G2c_Wnnp["down","down"])
    G2c_Wnnp["up","down"]   = 0.5 * (G2c_Wnnp["up","down"] + G2c_Wnnp["down","up"])
    G2c_Wnnp["down","down"] = G2c_Wnnp["up","up"]
    G2c_Wnnp["down","up"]   = G2c_Wnnp["up","down"]

    beta      = G_imp.mesh.beta
    gf_struct = [('up', 1), ('down', 1)]
    n_imp     = 2. * G_imp["up"][0,0].density().real
    n_iwf_G2c = len(G2c_Wnnp['up', 'up'].mesh[0])//2

    print(f"U = {U} and beta = {beta}")
    print(f"Impurity density : {n_imp}")
    print(f"Number of positive bosonic frequencies = {Niwb}")
    print(f"Number of positive fermionic frequencies in the 2P-GF = {n_iwf_G2c}")
    print(f"\n")

    #Niwb = 20
    n_iwf_vertex = 50#n_iwf_G2c - Niwb

    print(f"Number of fermionic frequencies in the vertex: {n_iwf_vertex}")
    print("\n")
    print("Calculation of the impurity generalized susceptibility.")

    chi_imp_wnnp = chi_tilde_ph_from_G2c(G2c_Wnnp, G_imp, gf_struct, Niwb, n_iwf_vertex)
    chi_uu = 0.5 * (chi_imp_wnnp['up', 'up'] + chi_imp_wnnp['down', 'down'])
    chi_ud = 0.5 * (chi_imp_wnnp['up', 'down'] + chi_imp_wnnp['down', 'up'])
    # Generalized impurity charge and spin susceptibility
    chi_ch_wnnp = chi_uu + chi_ud
    chi_sp_wnnp = chi_uu - chi_ud

    print("Calculation of the vertex functions for the usable bosonic range.")
    # Generalized charge and spin susceptibility for usable bosonic frequencies
    chi_ch_wnnp_usable = make_tensor_valued_chi(make_chi_usable_frequencies(chi_ch_wnnp[0,0,0,0]))
    chi_sp_wnnp_usable = make_tensor_valued_chi(make_chi_usable_frequencies(chi_sp_wnnp[0,0,0,0]))

    chi_ch_imp = chi_ch_wnnp_usable
    chi_sp_imp = chi_sp_wnnp_usable
    print(f"Usable bosonic frequencies in the charge channel: {(len(chi_ch_imp.mesh[0])+1)//2}")
    print(f"Usable bosonic frequencies in the spin channel: {(len(chi_sp_imp.mesh[0])+1)//2}")
    print("\n")
    print("calculation of the impurity generalized bubble.")
    print("\n")
    # local generalized bubble
    chi0_wnnp_sp = chi0_from_gg2_PH(G_imp["up"] , chi_sp_imp)
    chi0_wnnp_ch = chi0_from_gg2_PH(G_imp["up"] , chi_ch_imp)

    print("Calculation of the impurity irreducible vertex in the charge and spin channel.")
    print("\n")
    # irreducible vertex in the charge and spin channnels
    Gamma_ch_imp   = inverse_PH(chi0_wnnp_ch) - inverse_PH(chi_ch_imp)
    Gamma_sp_imp   = inverse_PH(chi0_wnnp_sp) - inverse_PH(chi_sp_imp)

    print("Calculation of the impurity full vertex in the charge and spin channel.")
    print("\n")
    # full local vertex in the charge and spin channels
    F_ch_imp = make_f_channel_from_chi(chi_ch_imp[0,0,0,0], G_imp["up"][0,0])
    F_sp_imp = make_f_channel_from_chi(chi_sp_imp[0,0,0,0], G_imp["up"] [0,0])

    if mpi.is_master_node():
        with HDFArchive(output_file, "a") as A:
            A["chi_ch_imp"]     = chi_ch_imp
            A["chi_sp_imp"]     = chi_sp_imp
            A["Gamma_ch_imp"]   = Gamma_ch_imp
            A["Gamma_sp_imp"]   = Gamma_sp_imp
            A["F_ch_imp"]       = F_ch_imp
            A["F_sp_imp"]       = F_sp_imp
            A["chi_imp_tensor"] = chi_imp_wnnp
            A["Sigma_imp"]      = Sigma_imp
            A["G_imp"]          = G_imp
