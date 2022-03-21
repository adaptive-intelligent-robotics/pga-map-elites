import numpy as np

# Define the grid-level selection (ie cell-selection)
class ArchiveSelector:

    def __init__(self, selec_type = "uniform"):
        self.select_type = selec_type

    def __call__(self, archive, batch_size):
        """
	Return a population of parent controllers
	Input:
	  - archive {dict} - archive to select from
	  - batch-size {int} - number of parent to select
	Output: parent-controllers {list} - list of parent controllers
	"""
        # Get archive keys and size
        keys = list(archive.keys())
        archive_size = len(keys)

        # Select the parent controllers
        parent_controllers = []

        # Uniform selection in the grid
        if self.select_type == "uniform":
            parent_centroids = np.random.randint(archive_size, size = batch_size)
            for centroid in parent_centroids:
                # Use the in-cell selection operator
                parent_controllers += [archive[keys[centroid]].select()] 

        return parent_controllers
