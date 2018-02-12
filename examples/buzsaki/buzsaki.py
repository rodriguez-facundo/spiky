# by F. Rodriguez
import spiky

# path to dataset
raw = 'data/raw_data.dat'

# path to parameter configuration file
params = 'parameters/parameters.json'

# create the clustering object
buz = spiky.New()

# load the parameters to perform the clustering
buz.loadParams(pfile=params)

# load the dataset
buz.loadRawFile(rfile=raw)

# run the algorithm
buz.run()

# plot the spikes
buz.plotClusters()

# compute confusion matrix
buz.blur()
