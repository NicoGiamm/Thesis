from qiskit.aqua.algorithms import VQE, NumPyEigensolver
import numpy as np
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.operators import Z2Symmetries
from qiskit import BasicAer, Aer
from qiskit.chemistry import FermionicOperator
from qiskit.aqua import QuantumInstance
 
backend = BasicAer.get_backend("statevector_simulator")
optimizer = SLSQP(maxiter=5)

#Mantenere le parti commentate solo per simulazioni di H-H
def Vqe(molecule, dist):
		
	freeze_list = [0]
	remove_list = [-3,-2]
	repulsion_energy = molecule.nuclear_repulsion_energy
	num_particles = molecule.num_alpha + molecule.num_beta
	num_spin_orbitals = molecule.num_orbitals * 2
	remove_list = [x % molecule.num_orbitals for x in remove_list]
	freeze_list = [x % molecule.num_orbitals for x in freeze_list]
	remove_list = [x - len(freeze_list) for x in remove_list]
	remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
	freeze_list += [x + molecule.num_orbitals for x in freeze_list]
	ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
	ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
	num_spin_orbitals -= len(freeze_list)
	num_particles -= len(freeze_list)
	ferOp = ferOp.fermion_mode_elimination(remove_list)
	num_spin_orbitals -= len(remove_list)
	qubitOp = ferOp.mapping(map_type='parity', threshold=0.00000001)
	qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
	shift = repulsion_energy + energy_shift

	initial_state = HartreeFock(
		num_spin_orbitals,
		num_particles,
		qubit_mapping='parity'
	) 
	
	var_form = UCCSD(
		num_orbitals=num_spin_orbitals,
		num_particles=num_particles,
		initial_state=initial_state,
		qubit_mapping='parity'
	)
	
	vqe = VQE(qubitOp, var_form, optimizer)
	vqe_result = np.real(vqe.run(backend)['eigenvalue'] + shift)
	
	result = NumPyEigensolver(qubitOp).run()
	exact_energies = np.real(result.eigenvalues) + shift
	
	return vqe_result
