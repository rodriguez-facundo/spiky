# by F. Rodriguez
import spiky

# path to dataset
raw = 'data/raw_data.mat'

# path to parameter configuration file
params = 'parameters/parameters.json'

# create the clustering object
qui = spiky.New()

# load the parameters to perform the clustering
qui.loadParams(pfile=params)

# load the dataset
qui.loadRawFile(rfile=raw)

# run the algorithm
qui.run()

# plot the spikes
qui.plotClusters()

# compute confusion matrix
qui.blur()
