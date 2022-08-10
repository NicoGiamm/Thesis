from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import matplotlib.pyplot as plt
import numpy as np
import pyscf
from pyscf import fci
from VarQE import Vqe
import time
import scipy.stats as st
from scipy.optimize import curve_fit

vqe_energies = []
full_ci_energies = []
full_ci_energies2 = []
full_ci_energies3 = []

distances = np.arange(0.5, 4.5, 0.1)

start = time.time()

for dist in distances:
	
	driver = PySCFDriver(atom="Li .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
	mol1 = pyscf.M(atom = 'Li .0 .0 .0; H .0 .0 ' + str(dist), basis = 'sto3g', symmetry = True)
	molecule = driver.run()
	rhf1 = mol1.RHF().run()
#	mol2 = pyscf.M(atom = 'H 0 0 0; H 0 0 ' + str(dist), basis = '6-31G*', symmetry = True)
#	rhf2 = mol2.RHF().run()
#	mol3 = pyscf.M(atom = 'H 0 0 0; H 0 0 ' + str(dist), basis = '6-31G**', symmetry = True)
#	rhf3 = mol3.RHF().run()
	
	#Risultati Full Configuration interaction
	fci_energy = pyscf.fci.FCI(rhf1).kernel()[0]
	full_ci_energies.append(fci_energy)
#	fci_energy2 = pyscf.fci.FCI(rhf2).kernel()[0]
#	full_ci_energies2.append(fci_energy2)
#	fci_energy3 = pyscf.fci.FCI(rhf3).kernel()[0]
#	full_ci_energies3.append(fci_energy3)
	
	#Risultati VQE
	vqe = Vqe(molecule, dist)
	vqe_energies.append(vqe)

	
for i in range(len(distances)):
	print("			", np.round(distances[i], 2), "&", np.round(vqe_energies[i],10), "&", np.round(full_ci_energies[i],10), "\\""\\")

print('Tempo di esecuzione', time.time()-start, 's')

plt.plot(distances, full_ci_energies, label = "Full-CI")
#plt.plot(distances, full_ci_energies2, label = "6-31G*")
#plt.plot(distances, full_ci_energies3, label = "6-31G**")

plt.plot(distances, vqe_energies, label = "VQE")
plt.scatter(distances, full_ci_energies, s = 110, marker='o' )
plt.scatter(distances, vqe_energies, s = 100, marker='*', alpha = 0.7 )
plt.xlabel('Atomic distance [Angstrom]')
plt.ylabel('Energy [Hartree]')
plt.legend()
plt.show()	
