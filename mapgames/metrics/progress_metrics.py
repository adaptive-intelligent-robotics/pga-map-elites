import numpy as np

class AllProgressMetrics:
    """
    Class containing all progress metric stats (classic, reeval and robust).
    """
    def __init__(self, save_path, file_name, nb_reeval):
        if nb_reeval == 0:
            self.progress_metrics_dico = {
                "classic": ProgressMetrics("classic", save_path, file_name),
            }
        else:
            self.progress_metrics_dico = {
                "classic": ProgressMetrics("classic", save_path, file_name),
                "reeval": ProgressMetrics("reeval", save_path, file_name),
                "robust": ProgressMetrics("robust", save_path, file_name)
            }
    
    def update(self, label, n_evals, archive):
        self.progress_metrics_dico[label].update(n_evals, archive)

    def write(self, label):
        self.progress_metrics_dico[label].write()

    def reset(self, label):
        self.progress_metrics_dico[label].reset()


class ProgressMetrics:
    """ 
    Define a metric on the progress of the archive.
    """

    def __init__(self, label, save_path, file_name):
        self.label = label # classic, reeval, robust

        prefixe = "" if label == "classic" else (label + "_")
        suffixe = "" if label == "classic" else ("-" + label)

        self.file = open(f"{save_path}/progress{suffixe}_{file_name}.csv", 'w')
        self.file.write(f"n_eval,{prefixe}coverage,{prefixe}min_fitness,{prefixe}max_fitness,{prefixe}sum_fitness,{prefixe}mean_fitness,{prefixe}median_fitness,{prefixe}max_actor_id\n")

        self.reset()


    def update(self, n_evals, archive):
        self.n_evals = n_evals
        fit_list = np.array([x.fitness for x in archive.values()])

        if len(archive) > 0:
            self.max_actor_id = archive[max(archive, key=lambda desc: archive[desc].fitness)].x.id
            self.coverage = len(archive.keys())
            self.min_fitness = fit_list.min()
            self.max_fitness = fit_list.max()
            self.sum_fitness = np.sum(fit_list) # for QD-score
            self.mean_fitness = np.mean(fit_list)
            self.median_fitness = np.median(fit_list)
        else:
            self.max_actor_id = -1
            self.coverage = 0
            self.min_fitness = 0
            self.max_fitness = 0
            self.sum_fitness = 0
            self.mean_fitness = 0
            self.median_fitness = 0

        

    def write(self):
        self.file.write(
            "{},{},{},{},{},{},{},{}\n".format(
                self.n_evals,
                self.coverage,
                self.min_fitness,
                self.max_fitness,
                self.sum_fitness,
                self.mean_fitness,
                self.median_fitness,
                self.max_actor_id)
        )
        print(f"    -> Max fitness: {self.max_fitness} --- Mean fit: {self.mean_fitness}")
        self.file.flush()

    def reset(self):
        self.n_evals = 0
        self.coverage = 0
        self.min_fitness = 0
        self.max_fitness = 0
        self.sum_fitness = 0
        self.mean_fitness = 0
        self.median_fitness = 0
        self.max_actor_id = 0
