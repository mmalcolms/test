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

U    = 2.0
beta = 15.0
nden = 1.0
input_imp   = "/home/deoliveira/Desktop/triangular_lattice/test_square/dmft_solution/dmft_U%.1f_b%.1f_n%.1f.h5"%(U, beta,nden)
with HDFArchive(input_imp, "r") as A:
    chi_ud    = A["chi_updn_ph_imp-9"]
    chi_uu    = A["chi_upup_ph_imp-9"]
    G_imp     = A["G_imp-9"]
    Sigma_imp = A["Sigma_imp-9"]

Sigma_imp_rank2 = Gf(mesh=Sigma_imp.mesh, target_shape = [1, 1])
Sigma_imp_rank2[0,0].data[:] = Sigma_imp.data[:]

n_imp = 2.*G_imp[0,0].density().real                                                                                                                                 
Niwb  = (len(chi_ud.mesh[0])+1)//2
Niwf  = len(chi_ud.mesh[1])//2

beta = chi_ud.mesh[1].beta
output_file = "/home/deoliveira/Desktop/triangular_lattice/test_square/2p_imp_qnts-U%.2f-b%.2f.h5"%(U, beta)

print("Half-filled square lattice Hubbard model")
print(f"U = {U} and beta = {beta}")
print(f"Impurity density : {n_imp}")
print(f"Number of positive bosonic frequencies = {Niwb}")
print(f"Number of positive fermionic frequencies = {Niwf}")
print(f"\n")

chi_ch = chi_uu + chi_ud 
chi_sp = chi_uu - chi_ud 
print("Calculation of the vertex functions for the usable bosonic range.")
# Generalized charge and spin susceptibility for usable bosonic frequencies                                                                                     
chi_ch_wnnp_usable = make_tensor_valued_chi(make_chi_usable_frequencies(chi_ch[0,0,0,0]))                                                                                          
chi_sp_wnnp_usable = make_tensor_valued_chi(make_chi_usable_frequencies(chi_sp[0,0,0,0]))

chi_ch_imp = chi_ch_wnnp_usable
chi_sp_imp = chi_sp_wnnp_usable
Niwb_usable = (len(chi_ch_imp.mesh[0])+1)//2

print(f"Number of usable bosonic frequencies  = {Niwb_usable}")
print("\n")
print("calculation of the impurity generalized bubble.")
print("\n")

# local generalized bubble   
chi0_wnnp_sp = chi0_from_gg2_PH(G_imp, chi_sp_imp)
chi0_wnnp_ch = chi0_from_gg2_PH(G_imp, chi_ch_imp)

print("Calculation of the impurity irreducible vertex in the charge and spin channel.")
print("\n")
# irreducible vertex in the charge and spin channnels
Gamma_ch_imp   = inverse_PH(chi0_wnnp_ch) - inverse_PH(chi_ch_imp)
Gamma_sp_imp   = inverse_PH(chi0_wnnp_sp) - inverse_PH(chi_sp_imp)

print("Calculation of the impurity full vertex in the charge and spin channel.")
print("\n")
# full local vertex in the charge and spin channels
F_ch_imp = make_f_channel_from_chi(chi_ch_imp[0,0,0,0], G_imp[0,0])
F_sp_imp = make_f_channel_from_chi(chi_sp_imp[0,0,0,0], G_imp[0,0])

if mpi.is_master_node():
    with HDFArchive(output_file, "a") as A:
        A["chi_ch_imp"]     = chi_ch_imp
        A["chi_sp_imp"]     = chi_sp_imp
        A["Gamma_ch_imp"]   = Gamma_ch_imp
        A["Gamma_sp_imp"]   = Gamma_sp_imp
        A["F_ch_imp"]       = F_ch_imp
        A["F_sp_imp"]       = F_sp_imp
        A["Sigma_imp"]      = Sigma_imp_rank2
        A["G_imp"]          = G_imp
