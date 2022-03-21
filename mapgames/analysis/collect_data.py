import numpy as np
import pandas as pd
import os
from . import get_files, find_min_max

pd.options.mode.chained_assignment = None # Disable warning when setting value directly in a dataframe

####################### DATA COLLECTION

def collect_data(*, data_path, variant_names, environment_names, 
                 batch_size = 100, stat_period = 20000, filename = "*", verbose = False):
    """
    Read all progress data files and fill corresponding dataframe.
    """

    ########################
    # Start data collection #

    data = pd.DataFrame()
    n_repl = 1 # Unique identifier to separate replications
    for exp in environment_names:
        verbose and print("\nWorking on environment : ", exp)

        # If variant list empty, plot all possible algos
        if variant_names == []:
            verbose and print("Collecting data")
            data, n_repl = sub_collect_data(data, n_repl, data_path, "", exp, filename, verbose = verbose)

        # If variant list non empty, go through the different algo variants encountered
        for variant in variant_names:
            verbose and print("Collecting data for variant : ", variant)
            data, n_repl = sub_collect_data(data, n_repl, data_path, variant, exp, filename, verbose = verbose)


    verbose and print("End of the data collection")
    if data.empty: return data


    #########################
    # Start data processing #

    # Correct names
    data = data.rename(columns={'n_eval' : 'Evaluations'})
    data = data.rename(columns=lambda x: " ".join([s[0].upper() + s[1:] for s in x.split("_")]))

    # Avoid duplicated columns names
    data = remove_duplicate_columns(data)
    if data.empty: return data

    # Uniformise evaluations across replications
    if "Evaluations" in data.columns: data = uniformise_evaluations(data)

    ###########################
    # Start data augmentation #

    for exp in data["Experiment"].drop_duplicates().values:
        # Display some infos on datas
        min_fit = find_min_max(data, exp)
        max_fit = find_min_max(data, exp, use_min = False)
        if "Coverage" in data.columns:
            for algo in data[data['Experiment'] == exp]['Algorithm'].drop_duplicates().values:
                print("    -> Found", \
	              len(data[(data['Experiment'] == exp) & \
		               (data['Algorithm'] == algo) & \
			       (data['Coverage'] == data['Coverage'])]['Replication'].drop_duplicates().values), \
	              "replications for", exp, "and", algo)
        # Env-specific data augmentation
        data = add_qd_score(data, exp, "", min_fit)
        data = add_qd_score(data, exp, "Reeval ", min_fit)
        data = add_qd_score(data, exp, "Robust ", min_fit)

    # Other data augmentation
    data = add_variation(data, stat_period, batch_size)

    # Sort by algorithms
    if "Evaluations" in data.columns: 
        data = data.sort_values(["Algorithm", "Replication", "Evaluations"], ignore_index=True)
    return data

################## Data collection

def sub_collect_data(data, n_repl, data_path, variant, exp, filename, verbose = False):

    # Get the names of the files associated to this variant
    files = get_files(data_path=data_path, variant=variant, env=exp, filetype=".csv", 
                      prefixe=filename, verbose=verbose)
    if len(files) == 0:
        verbose and print("No files", filename, "for experiment", exp)
        return data, n_repl
    verbose and print("\nNumber of files found : ", len(files))
    verbose and print("Files:\n", files)

    # For each file
    for f in files:
        verbose and print("\nCollecting file : ", f)
                
        # Read it
        data_tmp = pd.read_csv(f, index_col=False)
        if data_tmp.empty: continue

        # Add meta data
        if variant == "":
            variant_name = f[f.rfind("/")+1:]
            variant_name = variant_name[variant_name.find("_")+1:]
            variant_name = variant_name[:variant_name.find(exp)-1]
        else:
            variant_name = variant
        data_tmp['Algorithm'] = variant_name
        data_tmp['Experiment'] = exp
        data_tmp['Replication'] = n_repl

        # Remove None and reset index
        data_tmp = data_tmp.replace("None", 0).reset_index(drop = True)

        # Aggregate this data in the general dataframe
        data = data.append(data_tmp, ignore_index=True)
        n_repl += 1
        verbose and print("Add the new data to the global dataframe")
    return data, n_repl


################## Data preprocessing

def remove_duplicate_columns(data):
    duplicated_columns_list = []
    list_of_all_columns = list(data.columns)
    for column in list_of_all_columns:
        if list_of_all_columns.count(column) > 1 and not column in duplicated_columns_list:
             duplicated_columns_list.append(column)
    for column in duplicated_columns_list:
        list_of_all_columns[list_of_all_columns.index(column)] = column + '_1' # Rename one of the two columns as _1
    data.columns = list_of_all_columns
    for column in duplicated_columns_list:
         for line in range(data.shape[0]):
             if not(np.isnan(data.at[line, column + '_1'])) and np.isnan(data.at[line, column]):
                 data.at[line, column] = data.at[line, column + '_1']
             elif not(np.isnan(data.at[line, column + '_1'])) and not(np.isnan(data.at[line, column])) and \
                 data.at[line, column + '_1'] != data.at[line, column]:
                 print("!!!WARNING!!! Two different values for the same point in column", column)
         data = data.drop(column + '_1', axis = 1)
    return data

def uniformise_evaluations(data):
    for exp in data["Experiment"].drop_duplicates().values:
        for variant in data["Algorithm"].drop_duplicates().values:
            sub_data = data[(data["Experiment"] == exp) & (data["Algorithm"] == variant)]
            sub_data = sub_data.sort_values(["Replication", "Evaluations"], ignore_index= True)
            replications = sub_data["Replication"].drop_duplicates().values
            replications_evals = []
            evals = []
            need_rewrite = False
            for i, repl in enumerate(replications):
                replications_evals.append(sub_data[sub_data["Replication"] == repl]["Evaluations"].values)
                if len(replications_evals[i]) > len(evals):
                   evals = replications_evals[i]
                elif any([replications_evals[i][j] != evals[j] for j in range(len(replications_evals[i]))]):
                    need_rewrite = True
            assert len(evals) > 0
            print(exp, variant, need_rewrite)
            if not need_rewrite: continue
            for i, repl in enumerate(replications):
                for j in range (len(replications_evals[i])):
                    data.loc[(data["Experiment"] == exp) & (data["Algorithm"] == variant) & \
		             (data["Replication"] == repl) & \
	                     (data["Evaluations"] == replications_evals[i][j]), "Evaluations"] = evals[j]
    return data


################## Data augmentation

def add_qd_score(data, exp, prefixe, min_fit):
    """
    Add the qd score to the data computed from the existing columns.
    """
    if min_fit == None: return data
    if prefixe + "Sum Fitness" not in data.columns: return data
    if prefixe + "Coverage" not in data.columns: return data
    qd_score = data[data.Experiment==exp][prefixe + "Sum Fitness"] \
                    - min_fit * data[data.Experiment==exp][prefixe + "Coverage"]
    data.loc[data.Experiment==exp, prefixe + "QD Score"] = qd_score
    return data

def add_variation(data, stat_period, batch_size):
    """
    Add some variation stats to the data computed from the existing columns.
    """

    if "Nb Evo" not in data.columns or "Improv Eff" not in data.columns or "Nb Improv" in data.columns:
        return data
    data["Nb Improv"] = data["Improv Eff"] * data["Nb Evo"]
    data["Nb Improv"] /= (stat_period / batch_size)
    if "Nb Evo" not in data.columns and "Discovery Eff" not in data.columns or "Nb Discovery" in data.columns:
        return data
    data["Nb Discovery"] = data["Discovery Eff"] * data["Nb Evo"]
    data["Nb Discovery"] /= (stat_period / batch_size)
    
    return data

