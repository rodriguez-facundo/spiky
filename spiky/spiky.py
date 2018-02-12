# by F. Rodriguez
import json
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy import signal
from pywt import wavedec as wd

from spiky import toolbox as tb


class New():
    ''' Spike sorting class

    Spike sorting based on Mixture of Gaussians Model, penalized by BIC.

    Parameters
    ----------
    pfile : str
        String containing the path to parameters json file
    rfile : str
        String containing the path to dataset

    Atributes
    ---------
    prms : dict
        Dictionary containing the configuration for preprocessing and clustering
    raw : ndarray
        Dataset
    thres : float
        Threshold level for spike detection
    pks : ndarray
        Array containing the spikes times
    spks : ndarray
        Spikes time series
    wvSpks : ndarray
        Wavelet decomposition of spikes
    extFeat : ndarray
        Array containing extra features such as Amplitud, Energy, Area
    X : ndarray
        Array containing normalized features for clustering
    gmm : Gaussian mixture class object
        The gaussian misxture object
    labels : ndarray
        Array containing the labels for each spike
    lr : ndarray
        L-ratios for each cluster
    '''

    def __init__(self, pfile=None, rfile=None):
        self.pfile = pfile
        self.rfile = rfile
        self.loadParams(pfile=self.pfile)
        self.loadRawFile(rfile=self.rfile)

    def loadParams(self, pfile):
        '''load parameters from json file'''
        params = dict()
        if pfile:
            try:
                with open(pfile) as json_data: self.prms = json.load(json_data)
                if len(self.prms.keys())>1:
                    print('%s file loaded correctly.' %pfile)
                else:
                    print("%s was loaded but it seems empty" %pfile)
            except:
                print("'%s' file was not found" %pfile)
        else:
            self.prms = dict()

    def loadRawArray(self, rarray):
        '''load raw data from np.array'''
        self.raw = rarray

    def loadRawFile(self, rfile):
        ''' Load raw data time serie from '.mat' or '.dat'.
            Recomendation: Use 'int16' types to save space.
        '''

        if rfile:
            try:
                if '.mat' in rfile:
                    name = sio.whosmat(rfile)[0][0]
                    self.raw = sio.loadmat(rfile)['data'][0]
                elif '.dat' in rfile:
                    self.raw = np.fromfile(rfile, dtype='int16')
                else:
                    print('Support only ".mat" or ".dat" files')
            except:
                print("Couldn't load data from '%s' file" % rfile)

        else:
            self.raw = list()

    def filter(self):
        ''' Filter data along one dimension using cascaded second-order sections
            digital IIR filter defined by sos.

            After filtering, data vector x is inverted and filtered one more time
            to minimize phase shifting

        Parameters
        ----------
        y : array_like
        	Input array.
        Q : int, optional
        	Filter's order.
        low : float, optional
        	Filter's low cut frequency.
        high : float, optional.
        	Filter's high cut frequency.
        nyq : int
        	Frequency sampling.
        kind : str, optional
        	Kind of filter to be applied:
        			'band' : band pass filter
        			'high' : high pass filter
        			'low ' : low pass filter

        Return
        ------
        out : ndarray
        	The output of the digital filter.
        '''
        Q = self.prms['filt']['q']
        low = self.prms['filt']['low']
        high = self.prms['filt']['high']
        nyq = self.prms['filt']['hz']

        # Filter constructor
        sos = signal.butter(Q, [1. * low / nyq, 1. * high / nyq],
            btype = kind, output='sos')

        # Apply filter (phase may be shifted)
        x = signal.sosfilt(sos, self.raw)

        # Reverse data
        x = f_eeg[::-1]

        # Apply filter (correct phase shift)
        self.raw = signal.sosfilt(sos, x)

    def run(self):
        # check required data is available
        if len(self.prms.keys())==0 or len(self.raw)==0:
            print("Time/parameters are missing")
            return None

        # start preprocessing
        print("Preprocesing")
        start_time = time.time()

        # calculate threshold level
        self.thres = self.prms['spkD']['thres'] * \
                     np.median(np.abs(self.raw)/0.6745)

        # find peaks in raw data
        self.pks = tb.pk(y=self.raw, thres=self.thres,
                      min_dist=self.prms['spkD']['minD'],
                      way=self.prms['spkD']['way'])

        # extract spike points from each detected peak
        self.spks, self.pks = tb.pts_extraction(raw=self.raw, pks=self.pks,
                                left=self.prms['spkD']['before'],
                                right=self.prms['spkD']['after'],
                                del_m_dist=self.prms['spkE']['minD'],
                                del_tol=self.prms['spkE']['lvl'])

        # interpolate and shift data
        self.spks, self.pks = tb.shift(spikes=self.spks, peaks=self.pks,
                                res=self.prms['spkA']['resol'],
                                way=self.prms['spkD']['way'])

        # wavelet decomposition spikes [[spk 1]...[spk n]]
        self.wvSpks = np.array([ wd(data=spk,
                                    wavelet=self.prms['wv']['func'],
                                    mode=self.prms['wv']['mode'],
                                    level=self.prms['wv']['lvl'])
                                        for spk in self.spks])

        # extra-feature calculation (amplitud, energy, etc)
        self.extFeat = tb.extra_features(spk=self.spks)

        # select features for clustering
        self.X = tb.preprocess(wavelets=self.wvSpks,
                                extra_features=self.extFeat,
                                nn=self.prms['gmm']['ftrs'],
                                cor=self.prms['gmm']['maxCorr'])

        # print preprocess information
        tb.preprocessPrint(thres=self.thres, pks=len(self.pks), t=start_time)

        # start cclustering
        start_time = time.time()
        print('Clustering')

        # BIC-EM-GMM
        self.gmm = tb.gmm_bic(X=self.X, n_clusters=self.prms['gmm']['maxK'],
                            initialization=self.prms['gmm']['inits'])

        # labeling
        self.labels = self.gmm.predict(self.X)

        # compute L-ratio for each cluster
        self.lr = tb.l_ratio(self.X, self.labels)

        # print clustering information
        tb.clusteringPrint(labels=set(self.labels), t=start_time, lratios=self.lr)

    def plotClusters(self):
        ''' plot spikes based on clustering results'''
        plt.style.use('dark_background')
        unique_labels = set(self.labels)
        n = len(unique_labels)
        rows, cols = tb.rowscols(n)
        fig, ax = plt.subplots(rows,cols, sharex=True, sharey=True)
        for i,l in enumerate(unique_labels):
            r = int(i/cols)
            c = i%cols
            n = np.count_nonzero(self.labels == l)
            ax[r][c].plot(np.transpose(np.array(self.spks)[(self.labels==l)]))
            ax[r][c].text(.05,.95,'L: '+str(l)+'\nN: '+ str(n),
                            horizontalalignment='left',
                            verticalalignment='top',
                            transform=ax[r][c].transAxes)

        fig.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes], visible=False)
        plt.show()

    def blur(self):
        ''' perform self bluring between spikes belonging to the same label'''

        print('Bluring')
        # blur spikes
        newSpks = tb.blur(self.spks, self.labels, way=self.prms['spkD']['way'],
                        alpha=self.prms['blur']['alpha'],
                        align=self.prms['spkD']['before'])

        # wavelet decomposition
        wvSpks = np.array([ wd(data=spk, wavelet=self.prms['wv']['func'],
                            mode=self.prms['wv']['mode'],
                            level=self.prms['wv']['lvl'])
                                for spk in newSpks])

        # add extra features
        extFeat = tb.extra_features(spk=newSpks)

        # preprocess features
        X = tb.preprocess(wavelets=wvSpks,
                            extra_features=extFeat,
                            nn=self.prms['gmm']['ftrs'],
                            cor=self.prms['gmm']['maxCorr'])

        # BIC-EM-GMM computation
        gmm = tb.gmm_bic(X=X, n_clusters=self.prms['gmm']['maxK'],
                        initialization=self.prms['gmm']['inits'])

        # new labels
        labels = gmm.predict(X)
        print('  DONE')

        # plot confusion matrix
        tb.confusion_matrix(self.labels, labels)
