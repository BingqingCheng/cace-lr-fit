import numpy as np
from ase.io import read, write


atoms_SR_list = read('./OOD-SR.xyz', index=':')
atoms_LR_list = read('./OOD-LR.xyz', index=':')

F_CACE_all = []
F_DFT_all = []

F_std_x_all = []
F_std_y_all = []
F_std_z_all = []
F_std_atom_all = []

F_diff_x_all = []
F_diff_y_all = []
F_diff_z_all = []
F_diff_atom_all = []

for index in [-1]:
    
    atoms = atoms_SR_list[index]
    F_std = atoms.get_array('F_std')

    F_std_atom = np.sqrt(np.sum(F_std ** 2, axis=1))

    f_CACE = atoms.get_array('F_CACE')
    f_DFT = atoms.get_array('F_DFT')

    F_std_x = F_std[:, 0]
    F_std_y = F_std[:, 1]
    F_std_z = F_std[:, 2]

    F_diff = (f_CACE - f_DFT) * 1000

    F_diff_x = F_diff[:, 0]
    F_diff_y = F_diff[:, 1]
    F_diff_z = F_diff[:, 2]

    F_diff_atom = np.sqrt(np.sum(F_diff ** 2, axis=1))

    F_std_x_all += F_std_x.tolist()
    F_std_y_all += F_std_y.tolist()
    F_std_z_all += F_std_z.tolist()

    F_std_atom_all += F_std_atom.tolist()
    F_diff_atom_all += F_diff_atom.tolist()

    F_diff_x_all += F_diff_x.tolist()
    F_diff_y_all += F_diff_y.tolist()
    F_diff_z_all += F_diff_z.tolist()

    F_CACE_all += f_CACE.reshape(-1).tolist()
    F_DFT_all += f_DFT.reshape(-1).tolist()

    F_CACE_all += f_CACE.reshape(-1).tolist()
    F_DFT_all += f_DFT.reshape(-1).tolist()

    atoms.set_array('F_diff_atom', F_diff_atom)
    atoms.set_array('F_std_atom', F_std_atom)

F_std_x_all_SR = F_std_x_all.copy()
F_std_y_all_SR = F_std_y_all.copy()
F_std_z_all_SR = F_std_z_all.copy()
F_std_atom_all_SR = F_std_atom_all.copy()

F_diff_x_all_SR = F_diff_x_all.copy()
F_diff_y_all_SR = F_diff_y_all.copy()
F_diff_z_all_SR = F_diff_z_all.copy()
F_diff_atom_all_SR = F_diff_atom_all.copy()

F_CACE_all_SR = F_CACE_all.copy()
F_DFT_all_SR = F_DFT_all.copy()

# %%
F_CACE_all = []
F_DFT_all = []

F_std_x_all = []
F_std_y_all = []
F_std_z_all = []
F_std_atom_all = []

F_diff_x_all = []
F_diff_y_all = []
F_diff_z_all = []

F_diff_atom_all = []

for index in [-1]:
    
    atoms = atoms_LR_list[index]
    F_std = atoms.get_array('F_std')

    F_std_atom = np.sqrt(np.sum(F_std ** 2, axis=1))

    f_CACE = atoms.get_array('F_CACE')
    f_DFT = atoms.get_array('F_DFT')

    F_std_x = F_std[:, 0]
    F_std_y = F_std[:, 1]
    F_std_z = F_std[:, 2]

    F_diff = (f_CACE - f_DFT) * 1000

    F_diff_x = F_diff[:, 0]
    F_diff_y = F_diff[:, 1]
    F_diff_z = F_diff[:, 2]

    F_diff_atom = np.sqrt(np.sum(F_diff ** 2, axis=1))

    F_std_x_all += F_std_x.tolist()
    F_std_y_all += F_std_y.tolist()
    F_std_z_all += F_std_z.tolist()

    F_std_atom_all += F_std_atom.tolist()
    F_diff_atom_all += F_diff_atom.tolist()

    F_diff_x_all += F_diff_x.tolist()
    F_diff_y_all += F_diff_y.tolist()
    F_diff_z_all += F_diff_z.tolist()

    F_CACE_all += f_CACE.reshape(-1).tolist()
    F_DFT_all += f_DFT.reshape(-1).tolist()

    F_CACE_all += f_CACE.reshape(-1).tolist()
    F_DFT_all += f_DFT.reshape(-1).tolist()

    atoms.set_array('F_diff_atom', F_diff_atom)
    atoms.set_array('F_std_atom', F_std_atom)

F_std_x_all_LR = F_std_x_all.copy()
F_std_y_all_LR = F_std_y_all.copy()
F_std_z_all_LR = F_std_z_all.copy()
F_std_atom_all_LR = F_std_atom_all.copy()


F_diff_x_all_LR = F_diff_x_all.copy()
F_diff_y_all_LR = F_diff_y_all.copy()
F_diff_z_all_LR = F_diff_z_all.copy()
F_diff_atom_all_LR = F_diff_atom_all.copy()

F_CACE_all_LR = F_CACE_all.copy()
F_DFT_all_LR = F_DFT_all.copy()

F_xyz_diff_LR = np.array([F_diff_x_all_LR, F_diff_y_all_LR, F_diff_z_all_LR]).T
F_xyz_diff_SR = np.array([F_diff_x_all_SR, F_diff_y_all_SR, F_diff_z_all_SR]).T


########## Plot the correlation ##########
import matplotlib.pyplot as plt
import matplotlib

font = {'weight': 'normal',
        'size': 25}
matplotlib.rc('font', **font)

plt.figure(figsize=(5, 6))  # Create a square figure


x_plot = np.array(F_std_atom_all_SR)
y_plot = np.abs(F_diff_atom_all_SR)


plt.plot(x_plot, y_plot, 'o', alpha=1, label=f'SR-SR')

#################

x_plot = np.array(F_std_atom_all_LR)
y_plot = np.abs(F_diff_atom_all_LR)


plt.plot(x_plot, y_plot, 'o', alpha=1 , label=f'LR-LR')


##################################

x_plot = np.array(F_std_atom_all_LR) 
y_plot = np.abs(F_diff_atom_all_SR) 

plt.plot(x_plot, y_plot, 'o', alpha=1 , color='C2', label=f'LR-SR')
# Get the maximum range to set equal limits
max_range = 450 # max(max(x_plot), max(y_plot))
plt.xlim(0, 300)
plt.ylim(0, max_range)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

plt.xlabel('Force Uncertainty [meV/Å]')
plt.ylabel('Force Error [meV/Å]')

plt.xticks([0, 100, 200, 300], fontsize=20)
plt.yticks([0, 150, 300, 450], fontsize=20)

plt.legend(fontsize=20, bbox_to_anchor=(0.12, 0.6605))  # This puts it in the upper right area
plt.show()