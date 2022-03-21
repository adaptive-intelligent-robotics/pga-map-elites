import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
my_cmap = mpl.cm.viridis


############################ Main-plot function

def plot_cartesian_map(archive_file, cell_ids_file, cell_boundaries_file, save_path, case_name, \
		       min_fit=False, max_fit=False, verbose=False, colormap=my_cmap):
    """
    Plot a cartesian archive.
    Inputs:
    	- archive_file {str} - file containing the archive
	- cell_ids_file {str} - file containing the corresponding cell ids
	- cell_boundaries_file {str} - file containing the corresponding cell boundaries
	- save_path {str} - path to save the resultant archive
	- case_name {str} - template of filename to save the archive
	- min_fit {float} - min fitness value for colormap
	- max_fit {float} - max fitness value for colormap
	- verbose {bool}
	- colormap {mpl.colormap}
    Outputs: {bool} - has the archive succesfully been ploted
    """

    # Read file
    fit, desc, cells = load_grid(archive_file)

    if len(fit) == 0:
        print("\n!!!WARNING!!! Empty archive:", archive_file, ".\n")
        return True
    if len(desc[0]) > 2:
        print("\n!!!WARNING!!! Archive has too many dimensions.")
        return False

    # Cell ids
    with open(cell_ids_file, 'rb') as f:
        cell_ids = pickle.load(f)

    # Cell boundaries
    with open(cell_boundaries_file, 'rb') as f:
        cell_boundaries = pickle.load(f)
    cell_boundaries[0][0] = 0
    cell_boundaries[0][-1] = 1
    cell_boundaries[1][0] = 0
    cell_boundaries[1][-1] = 1

    # In the case of QR-RL, additional step 
    if "QD-RL" in archive_file:
        for i, bd in enumerate(desc):
            cell_id = find_cell_id(bd, cell_boundaries, cell_ids)
            cells[i] = cell_id

    # Fitness range
    verbose and print("Fitness max : ", max(fit))
    index = np.argmax(fit)
    verbose and print("Average fit:", fit.sum() / fit.shape[0])
    verbose and print("Associated desc : " , desc[index] )
    verbose and print("Associated cell : " , cells[index] )
    verbose and print("Index : ", index)
    verbose and print("total len ",len(fit))

    # If no min and max fitness given, use min and max fitness of the file
    if not min_fit:
        min_fit = min(fit)
    elif min_fit > min(fit):
        print("!!!Warning!!! Fitness minimum chosen greater than actual minimum fitness:", min(fit))
    if not max_fit:
        max_fit = max(fit)
    elif max_fit < max(fit):
        print("!!!Warning!!! Fitness maximum chosen lower than actual maximum fitness:", max(fit))
    verbose and print("Min = {} Max={}".format(min_fit, max_fit))

    # Set plot params
    params = {
        'axes.labelsize': 18,
        'legend.fontsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'text.usetex': False,
        'figure.figsize': [6, 6]}
    mpl.rcParams.update(params)

    # Plot
    fig, ax = plt.subplots(facecolor='white', edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set(adjustable='box', aspect='equal')
    norm = mpl.colors.Normalize(vmin=min_fit, vmax=max_fit)

    all_bounds = []
    for cell in np.nditer(cell_ids):
        idx = np.where(cell_ids == cell)
        index = []
        for i, dim in enumerate(idx):
            index.append(dim[0])
        bound_1 = [cell_boundaries[0][index[0]], cell_boundaries[1][index[1]]]
        bound_2 = [cell_boundaries[0][index[0]+1], cell_boundaries[1][index[1]]]
        bound_3 = [cell_boundaries[0][index[0]+1], cell_boundaries[1][index[1]+1]]
        bound_4 = [cell_boundaries[0][index[0]], cell_boundaries[1][index[1]+1]]

        bounds =  np.array([bound_1, bound_2, bound_3, bound_4])
        all_bounds.append(bounds)
                    
                    
    for idx, cell in enumerate(np.nditer(cell_ids)):
        polygon = all_bounds[idx]
        ax.fill(*zip(*polygon), alpha=0.5, edgecolor='black', facecolor='white', lw=1)

    for i in range(0, len(desc)):
        cell_id = cells[i][0]
        polygon = all_bounds[cell_id]
        ax.fill(*zip(*polygon), alpha=0.9, color=colormap(norm(fit[i]))[0])

    # Add axis name and colorbar
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"Leg 1 proportion of contact-time")
    ax.set_ylabel(f"Leg 2 proportion of contact-time")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax)
    cbar.set_label("Fitness", size=24, labelpad=5)
    cbar.ax.tick_params(labelsize=18) 

    # Save figure
    fig.savefig(f"{save_path}/{case_name}.png", bbox_inches = 'tight')
    plt.close()
    return True


############################ Files load functions

def load_grid(filename):
    """ Load a cartesian archive file and transform it into a usable dataframe. """
    data = np.loadtxt(filename)
    if len(data) == 0:
      return [], [], []
    if len(data.shape) == 1:
      fit = np.array([np.array([data[0]])])
      cells = np.array([data[1:2].astype(int)])
      desc = np.array([data[2::]])
      return fit, desc, cells
    fit = data[:,0:1]
    cells = data[:,1:2].astype(int)
    desc = data[:,2::]
    return fit, desc, cells


def find_cell_id(BD, boundaries, cell_ids):
    """Find cell identifier of the BD map corresponding to bd. """
    coords = []
    for j in range(len(BD)):
        inds = np.atleast_1d(np.argwhere(boundaries[j] < BD[j]).squeeze())
        if len(inds) == 0: coords.append(0)
        else: coords.append(inds[-1])
    coords = tuple(coords)
    cell_id = cell_ids[coords]
    return cell_id

