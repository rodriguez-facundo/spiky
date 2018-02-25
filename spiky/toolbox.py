# by F. Rodriguez
import json
import time
import itertools
import numpy as np
import progressbar as pb
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import kstest
from scipy.stats.stats import pearsonr
from scipy.interpolate import CubicSpline as cs
from scipy.optimize import linear_sum_assignment

from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture as GMM

def pk(y, thres, min_dist=70, way='peak'):
    """Peak detection routine.

    Finds the numeric index of peaks in *y* by taking its first
    order difference. By using *thres* and *min_dist* parameters, it
    is possible to reduce the number of detected peaks.

    Parameters
    ----------
    y : ndarray
       1D time serie data array.
    thres : int
       Parameter controling the threshold level.
    min_dist : int, optional.
       Minimum distance between detections (peak with
       highest amplitude is preferred).
    way : str (optional)
        If 'peak' computes maximum
        If 'valley' computes minimum

    Returns
    -------
    out : list
        Array containing the numeric indexes of the peaks
    """

    # distance between points must be integer
    min_dist = int(min_dist)

    # flip signal
    if way=='valley': y = np.array([-i for i in y])

    # first order difference
    dy = np.diff(y)

    # propagate left and right values successively
    # to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)

    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    while len(zeros):
        # add pixels 2 by 2 to propagate left and
        # right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                    & (np.hstack([0., dy]) > 0.)
                    & (y > thres))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    peaks = [int(peaks[i]) for i in range(len(peaks))]

    return peaks

def pts_extraction(raw, pks, right=43, left=20, del_m_dist=3, del_tol=0.85):
    ''' Extract spikes from peaks

    Parameter
    ---------
        raw : ndarray
            Raw time serie data.
        pks : list
            Peaks position.
        right : int, optional
            Number of point to the right of the peak.
        left : int, optional
            Number of points to the left of the peak.
        del_m_dist : int
            If more than one peak is within the same time windows,
            the spike will be discarded.
        del_tol : float
            If both peaks detected within the same time windows are
            about the same height, they will be discarded. del_tol
            is the percentage above which the spike will be deleted.

    Return
    ------
        out1 : list
            List containing the spikes
        pks : list
            List containing the time for each spike (excluding deleted spks)

    '''
    # If first index in "pks" is encountere before position "left" in "raw"
    if pks[0] < left:
        # if last index in "pks" is beyond "right" position in "raw"
        if pks[-1] > len(raw_data)-right:
            # skip first and last one
            out = [raw[i-left:i+right+1] for i in pks[1:-1]]
        else:
            # skip only first one
            out = [raw[i-left:i+right+1] for i in pks[1:]]
    else:
        # keep all the peaks
        out = [raw[i-left:i+right+1] for i in pks]

    # delete spk with multiple fires
    multi = simultaneous(out, min_dist=del_m_dist, tol=del_tol)

    print('  Simultaneous spikes deleted: \t%d' % len(multi))
    out = [u for i, u in enumerate(out) if i not in multi]
    pks = [u for i, u in enumerate(pks) if i not in multi]

    return out, pks

def simultaneous(spk, min_dist, tol):
    '''returns indices of spikes with multiple fires'''
    out = list()
    for i, u in enumerate(spk):
        if len(pk(u, thres=tol*max(u), min_dist=min_dist))>1: out.append(i)

    return out

def shift(spikes, peaks, res=100, way='peak'):
    ''' Shift data series in order to align
        maximum/minimum with serie indices.

    Parameters
    ----------
        spikes : np.array(shape(n, dim))
            Time serie.
        peaks : list
            Some pathological spikes shifts could result in non realistic
            spike shapes. Those cases will be deleted from the list of
            available peaks
        res : int
            Resolution points to compute shift.
        way : str()
            if 'valley', the shift will align with global minima
            if 'peak', the shift will align with global maxima

    Returns
    -------
        out : list() (shape(n-deleted spikes, dim))
            List of spikes shifted to be aligned with max/min.
        peaks : list()
            List containing the indices of peaks in raw data
            (pathological cases errased)
    '''
    # lenght of each spike vector
    size = len(spikes[0])

    # steps vector
    t = np.arange(0, size, 1./res)

    # output
    out = list()

    # find minima
    for i, spk in enumerate(spikes):
        # build cubic splines function
        f = cs(range(size), spk, bc_type='natural')

        # compute max/min displacement
        if way=='valley':
            disp = np.argmin(f(t)) /res - np.argmin(spk)
        else:
            disp = np.argmax(f(t)) /res - np.argmax(spk)

        # delete spike if minimum moved further than one position
        if abs(disp)>1:
            out.append([])
        else:
            out.append(f(np.arange(size)+disp))

    # print message with deleted number of spikes
    deleted = len([True for spk in out if len(spk)==0])
    print('  Interpolated spike deleted: \t%d' % deleted)

    # delete pathological cases
    peaks = [pt for spk, pt in zip(out, peaks) if len(spk)!=0]
    out = [spk for spk in out if len(spk)!=0]

    # convert back to integer type
    out = [np.round(spk).astype('int16') for spk in out]

    return out, peaks

def extra_features(spk):
    '''compute set of extra features in data

    Returns
    -------
        energy : float
            Energy computed as sum(x_i)^2/dim
        amplitud : float
            Amplitud of the peak.
        area : float
            Total area covered by the spike.
    '''

    # total energy
    energy = np.sum(np.array(spk)**2, axis=1)

    # maximum amplitud
    amplitud = np.max(spk, axis=1)

    # total area
    area = np.sum(spk, axis=1)

    return np.array([energy, amplitud, area])

def pick_coeff(coeff, n=3, corr=0.35):
    '''takes an array of coefficients and returns the first "n"
        coefficients with correlation below "corr"

        ATENTION:   the list of coefficients is ordered from
                    most to least multimodal coefficient

    Parameters
    ----------
        coeff : np.array (shape(dim, k))
            2D array containing "dim" coefficients with "k" values for each one.
        n : int
            Max. number of coefficient to return.
        corr : float
            Max. correlation allowed between coefficients

    Returns
        out : np.array (shape(n, k))
            Array containing the selected coefficients
    '''

    # variable declaration
    i = 0; j = 1

    out = [coeff[0]]
    stop = len(coeff)

    # repeat until selection of "n" coeff or end of vector
    while len(out)<n and j<stop:
        # go to next one if coeff are highly correlated
        while pearsonr(coeff[i], coeff[j])[0] > abs(corr) and j<stop: j +=1

        # compare correlation with elements already in the final list
        if j<stop:
            if all(pearsonr(u, coeff[j])[0] < abs(corr) for u in out):
                out.append(coeff[j])
                i = j
                j +=1
            else:
                j +=1

    return out

def preprocess(wavelets, extra_features, nn=2, cor=0.3):
    ''' preprocess data to feed into GMM algorithm

    Parameters
    ----------
        wavelets : ndarray
            Wavelet decomposition from pywt module
        extra_features : ndarray
            Extra features to be appended for computation
        nn : int
            Desired final dimension of data points for clustering
        cor : float
            Maximum correlation between features

    Return
    ------
        out : ndarray (shape(nn, x))
            Array with normalized data points to feed GMM algorithm
    '''

    # flat wavelet coefficients
    w_mod = [[inner for outer in pt for inner in outer] for pt in wavelets]

    # arrange wavelets by coefficient [[coeff 1]...[coeff 64]]
    wlts = [[] for i in range(len(w_mod[0]))]
    for w in w_mod:
        for i, k in enumerate(w): wlts[i].append(k)

    # append extra features
    for u in extra_features: wlts.append(u.tolist())

    # find mean and std_dev for each feature
    w_norm = [norm.fit(wavelet) for wavelet in wlts]

    # Lilliefors adaptation for KS test of normality
    lil = [kstest(i, 'norm', args=j) for i, j in zip(wlts, w_norm)]

    # features sorted with respect to lilliefors test
    wlts = [i for j, i in sorted(zip(lil, wlts))][::-1]

    # select features
    out = pick_coeff(wlts, n=nn, corr=cor)

    # arange features as N-Dimensional points
    return StandardScaler().fit_transform(np.transpose(out))

def gmm_bic(X, n_clusters=8, initialization=5):
    '''Gaussian Mixture of Models with BIC score'''
    # widget to print while computing mixture of gaussians
    widgets = [ '  ', pb.Percentage(), ' | ', pb.Timer(), pb.Bar(), pb.ETA(),
                ' | ', pb.DynamicMessage('Neurons')]

    # initialize bic score
    lowest_bic = np.infty
    bic = []

    bar = pb.ProgressBar()
    with pb.ProgressBar(max_value=n_clusters, widgets=widgets) as bar:
        # try different number of clusters
        for k in range(1, n_clusters):
            gmm=GMM(n_components=k, n_init=initialization)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            # keep if the score is the lowest so far
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                bar.update(k, Neurons=k)
            else:
                bar.update(k)

    return best_gmm

def l_ratio(X, labels):
    ''' This is a meassure of how far a cluster is from neighbouring clusters
        computing the mahalanobis distance to the closest point that does not
        belong to the cluster

        ATENTION:   the covariance matrix is estimated with the robust
                    covariance (outliers not taken into account)

    Parameters
    ----------
        X : ndarray
            Data (assumed to be multivariate normal distributed)
        labels : ndarray
            Labels

    Returns
    -------
        lr : list, size(number of clusters)
            L-ratio for each cluster
    '''
    lr = list()

    # unique labels
    unique_l = set(labels).difference([-1])

    # if the set is empty, return 0
    if len(unique_l)==0:
        return -1

    # degrees of freedom
    df = len(X[0])

    # for each cluster
    for label in unique_l:
        # compute points in cluster
        Xi = X[(labels==label)]

        # number of spikes in cluster
        n = len(Xi)

        # compute points out of the cluster
        outliers = X[(labels!=label)]

        # estimate robust covariance
        mcd = MinCovDet().fit(Xi)

        # compute mahalanobis distance for outliers
        Dmcd = mcd.mahalanobis(outliers)

        # compute L-ratio
        lr.append(np.sum(1-chi2.cdf(Dmcd,df))/n)

    return lr

def blur(data, labels, alpha=0.1, align=20, way='peak'):
    '''Blurs spikes with other spikes within the same label

    Parameter
    ---------
        data : ndarray
            Array containing data.
        labels : ndarray
            Array containing labels.
        alpha : float (optional)
            Controls the level of perturbation.
        align : int
            Position of maximum for each spike.
        way : str
            If 'peak' spikes are aligned to maximum.
            If 'valley' spikes are aligned to minimum.

    Returns
    -------
        out : ndarray
            Blurred spikes
    '''
    data = np.array(data)

    # set of unique labels
    unique_labels = set(labels)

    # initialize permutation array
    permute = np.arange(len(labels))

    # output
    out = np.array([spk for spk in data])

    # for each cluster
    for l in unique_labels:
        # create permutation of spike
        permute[(labels==l)] = np.random.permutation(permute[(labels==l)])

        # compute average spikes
        mean = np.mean(data[(labels==l)], axis=0, dtype='int16')

        # blur data
        out[(labels==l)] -=  np.array(alpha * (mean-data[permute[(labels==l)]]),
                                dtype='int16')

    # restore distorted spikes
    if way=='valley':
        keep = [i for i, spk in enumerate(out) if np.argmin(spk)!=align]
    else:
        keep = [i for i, spk in enumerate(out) if np.argmax(spk)!=align]

    for i in keep: out[i] = data[i]

    return out

def confusion_matrix(labels1, labels2):
    '''build confusion matrix after bluring spikes with themselfs

    Parameters
    ----------
        labels1 : ndarray
            Labels from first run
        labels2 : ndarray
            Labels from second run

    Returns
    -------
        Plots confusion matrix

    '''

    unique_l1 = set(labels1)
    unique_l2 = set(labels2)

    n1 = len(labels1)
    n2 = len(labels2)

    # force same amount of spikes
    if n1!=n2: labels2 = np.append(labels2, -np.ones(n1-n2))

    # create confusion matrix
    M = np.zeros((len(unique_l1), len(unique_l2)))

    # vector counting matches
    count = np.zeros(len(labels1), dtype=bool)

    # counting matches
    for i, l1 in enumerate(unique_l1):
        for j, l2 in enumerate(unique_l2):
            M[i,j] = len(count[(labels1==l1) & (labels2==l2)])

    # solve maximization problem (hungarian problem)
    row, col = linear_sum_assignment(np.max(M)-M)

    # rearange confusion matrix
    M = M[:,col][row]

    # ploting
    plot_confusion(M, np.array(list(unique_l2))[col], np.array(list(unique_l1))[row])

def plot_confusion(M, labelx, labely):
    '''plots confusion matrix'''
    plt.imshow(M, interpolation='nearest', cmap=plt.cm.inferno)
    plt.colorbar()
    tick_marks = np.arange(len(M))
    plt.xticks(tick_marks, labelx)
    plt.yticks(tick_marks, labely)
    thresh = M.max() / 2.
    plt.ylabel('True spikes')
    plt.xlabel('Blurred spikes')
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        plt.text(j, i, format(M[i, j], '.0f'),
                 horizontalalignment="center",
                 color="black" if M[i, j] > thresh else "white")
    plt.gca().xaxis.tick_top()
    plt.show()

def rowscols(n):
    ''' deside how many rows and columns will be used to plot subplots'''
    if   n==1: return [1, 1]
    elif n==2: return [2, 1]
    elif n==3: return [3, 1]
    elif n==4: return [2, 2]
    elif n<=6: return [2, 3]
    elif n<=8: return [2, 4]
    elif n<=10: return [2, 5]
    elif n<=12: return [3, 4]
    elif n<=15: return [3, 5]
    else:
        print('Plotting only first 15 classes')
        return [3, 5]

def preprocessPrint(thres, pks, t):
    print('  Threshold: \t\t\t%.2f' % thres)
    print('  Detected peaks:\t\t%d' % pks)
    print('  Extra features:\t\tEnergy, Amplitud, Area')
    text = "  Preprocessing time: \t\t"
    print(text + '%.2f sec.' % (time.time() - t))
    print('  DONE')

def clusteringPrint(labels, t, lratios):
    print("  Clusters found: \t%d" % len(labels))
    text = "  CLustering time: \t\t"
    print(text + '%.2f sec.' % (time.time() - t))
    print('  L-ratios:')
    for i, l in enumerate(labels): print(3*' '+str(l)+': %.2f' % lratios[i])
    print("  DONE")
