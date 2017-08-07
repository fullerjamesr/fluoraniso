import numpy as np
import matplotlib.pyplot as plt
import fluoraniso

# A sample dataset
ligand_concentration = 6.0
protein_concentrations = np.array([0.0, 0.48828125, 0.9765625, 1.953125, 3.90625, 7.8125, 15.625, 31.25, 62.5, 125.0,
                                   250.0, 500.0])
anisotropies = np.array([155.6044841, 155.6604261, 156.1728258, 155.0651991, 156.1683244, 161.1155556, 167.7940681,
                         183.3580883, 186.3442391, 187.3740944, 189.5407683, 189.7413338])
anisotropies_std = np.array([1.236084738, 0.809102889, 0.850551731, 0.703078516, 1.263584263, 0.788232401, 0.905118557,
                             0.561789379, 1.322522261, 1.788106265, 0.72623112, 0.966320949])

# Let's compare the fit using the normal vs cooperative models
noncooperative_opt, _ = fluoraniso.do_noncooperative_fit(protein_concentrations, anisotropies, anisotropies_std,
                                                         ligand_concentration)
cooperative_opt, _ = fluoraniso.do_hill_fit(protein_concentrations, anisotropies, anisotropies_std,
                                            ligand_concentration)

plt.xscale("log")
plt.xlabel("Protein concentration (nM)")
plt.ylabel("Anisotropy (mA)")
plt.errorbar(protein_concentrations, anisotropies, yerr=anisotropies_std, label="Data")
sim_protein_conc = np.arange(protein_concentrations[1], protein_concentrations[-1],
                             (protein_concentrations[-1] - protein_concentrations[1]) / 500)
plt.plot(sim_protein_conc,
         fluoraniso.simple_binding_function(sim_protein_conc, *noncooperative_opt, ligand_concentration),
         label="Non-cooperative fit")
plt.plot(sim_protein_conc, fluoraniso.simulate_cooperative_binding(sim_protein_conc, *cooperative_opt,
                                                                   ligand_concentration),
         label="Cooperative fit")
plt.legend()
