from triqs.gf import Gf, BlockGf, Block2Gf, Idx
from triqs.gf import MeshImFreq, MeshProduct
from h5 import HDFArchive

from triqs.lattice.tight_binding import TightBinding, dos, TBLattice
from triqs.lattice import BravaisLattice, BrillouinZone

# define the Bravais lattice: a square lattice in 2d
units = [(1, 0, 0), (0, 1, 0)]
BL = BravaisLattice(units)

# define the tight-binding model, i.e., the hopping parameters
t = -1.0
hop= {  (1, 0)  :  [[ t]],
        (-1, 0) :  [[ t]],
        (0, 1)  :  [[ t]],
        (0, -1) :  [[ t]]
     }

n_k = 50
H = TBLattice(units=units, hopping=hop)
k_mesh        = H.get_kmesh(n_k = (n_k, n_k, 1))
eps_k         = Gf(mesh=k_mesh, target_shape = [1, 1])

eps_k[0,0].data[:] = H.dispersion(k_mesh)[0].data[:]

output_file = "/home/deoliveira/Desktop/BSE_test/dispersion-nk%s.h5"%n_k

with HDFArchive(output_file, "a") as A:
    A["dispersion"] = eps_k
