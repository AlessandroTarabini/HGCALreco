import uproot3
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import os
import os.path as osp
import awkward0
import math
import torch_geometric
from sklearn.decomposition import PCA
from wpca import WPCA, EMPCA
import torch
import pandas as pd
from torch_geometric.data import Data
import glob

# Function to convert the number of the layer L to the z coordinate
convLtoZ = {1: 322.10272, 2: 323.0473, 3: 325.07275, 4: 326.01727, 5: 328.0428, 6: 328.98727, 7: 331.01276, 8: 331.9572, 9: 333.9828, 10: 334.92725,
            11: 336.95273, 12: 337.89728, 13: 339.9228, 14: 340.86725, 15: 342.89273, 16: 343.8373, 17: 345.86276, 18: 346.80716, 19: 348.83282, 20: 349.77734,
            21: 351.80276, 22: 352.74725, 23: 354.7728, 24: 355.71725, 25: 357.74274, 26: 358.6873, 27: 360.7128, 28: 361.65726, 29: 367.69904, 30: 373.149,
            31: 378.59906, 32: 384.04898, 33: 389.499, 34: 394.94897, 35: 400.39902, 36: 405.84903, 37: 411.27457, 38: 416.7256, 39: 422.17126, 40: 427.62305,
            41: 436.16428, 42: 444.7138, 43: 453.25305, 44: 461.8166, 45: 470.36862, 46: 478.9221, 47: 487.46515, 48: 496.02094, 49: 504.57083, 50: 513.1154}
def conversionLtoZ(layer):
    return convLtoZ[int(layer)]


def angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
#     dot_product = np.dot(unit_vector_1, unit_vector_2)
#     angle = np.arccos(dot_product)
    cross_product = np.cross(unit_vector_1, unit_vector_2)
    cross_product_norm = np.linalg.norm(cross_product)
    angle = np.arcsin(cross_product_norm)
    return np.degrees(angle)

def get_phi(arg):
    if arg>0:
        return math.atan(arg)
    elif arg<0:
        return math.atan(arg)+math.pi
    else:
        return math.pi/2


def cleaning(_feats, pca_algo):

    X = _feats
    X = np.concatenate((X,np.array([conversionLtoZ(i) for i in X[:,2:3]])[:, np.newaxis]), axis=1)
    X[:,[2,4]] = X[:,[4,2]]
    X[:,[3,4]] = X[:,[4,3]] #After this last swap the structure of X is the following [x,y,z,L,energy]
    X = np.concatenate((X,X[:,4][:, np.newaxis]), axis=1)
    X = np.concatenate((X,X[:,4][:, np.newaxis]), axis=1)

    ####------------------------------------------------ fill array with max E LC per layer ------------------------------------------------####
    maxvalarr = []
    for i in range(int(np.max(X[:,3]))+1):
        if (len(X[:,3][X[:,3] == i]) ==0):
            continue
        layi = X[:,4][X[:,3] == i].copy()
        maxidx = np.argmax(layi)
        xmax = X[:,0][X[:,3] == i][maxidx]
        ymax = X[:,1][X[:,3] == i][maxidx]
        lmax = X[:,3][X[:,3] == i][maxidx]
        emax = X[:,4][X[:,3] == i][maxidx]
        maxvalarr.append([xmax,ymax,lmax,emax,conversionLtoZ(lmax)])

    maxvalarr = np.array(maxvalarr)

    #### get maxE LC index
    #Consider only layers<28 to find the maxE LC layer
    tmp = np.array([i for i in X if i[3]<28])
    maxidx = np.argmax(tmp[:,4])
    maxl = tmp[maxidx,3]
    maxe = tmp[maxidx,4]
    maxx = tmp[maxidx,0]
    maxy = tmp[maxidx,1]

    ####------------------------------------------------ fill array with max E LC per layer for +-N layers from the maxE LC layer ------------------------------------------------####
    cleanarr = []
    for i in range(0,16,1):
        if len(X[:,4][X[:,3] == maxl + i])==0 :
            continue

        maxeidx = np.argmax(X[:,4][X[:,3] == maxl + i])
        xclean = X[:,0][X[:,3] == maxl + i][maxeidx]
        yclean = X[:,1][X[:,3] == maxl + i][maxeidx]
        lclean = X[:,3][X[:,3] == maxl + i][maxeidx]
        eclean = X[:,4][X[:,3] == maxl + i][maxeidx]
        cleanarr.append([xclean,yclean,conversionLtoZ(lclean),lclean,eclean,eclean,eclean]) #eclean is added three times to prepare the array for the weighted PCA

    for i in range(-1,-11,-1):
        if len(X[:,4][X[:,3] == maxl + i])==0 :
            continue

        maxeidx = np.argmax(X[:,4][X[:,3] == maxl + i])
        xclean = X[:,0][X[:,3] == maxl + i][maxeidx]
        yclean = X[:,1][X[:,3] == maxl + i][maxeidx]
        lclean = X[:,3][X[:,3] == maxl + i][maxeidx]
        eclean = X[:,4][X[:,3] == maxl + i][maxeidx]
        cleanarr.append([xclean,yclean,conversionLtoZ(lclean),lclean,eclean,eclean,eclean])

    cleanarr = np.array(cleanarr)

    ####------------------------------------------------ PCA with +-N layers from the maxE LC layer ------------------------------------------------####
    cleanarr_pca = np.array([i for i in cleanarr if i[3]<28]) # To compute the PCA axis consider only LC in the EM compartment


    #Energy-weighted PCA
    if pca_algo == 'std':
        pca = PCA(n_components=3)
        pca.fit(cleanarr_pca[:,:3])
    elif pca_algo == 'stdAllLCs':
        pca = PCA(n_components=3)
        pca.fit(X[:,:3])
    elif pca_algo == 'eWeighted':
        pca = WPCA(n_components=3)
        pca.fit(cleanarr_pca[:,:3], weights = cleanarr_pca[:,4:])
    elif pca_algo == 'eWeightedAllLCs':
        pca = WPCA(n_components=3)
        pca.fit(X[:,:3], weights = X[:,4:])

    mincompidx = np.argmax(pca.explained_variance_)

    pca_axis = np.array([*pca.components_[0]])

    origin = [maxx,maxy,maxl]

    ####------------------------------------------------ fill array with LC with least dist to PCA ------------------------------------------------####
    pcaminarr = []
    distance = []
    index = []
    for i in range(int(np.max(X[:,3]))+1): #Loop on layers
        if (len(X[:,3][X[:,3] == i]) ==0):
            continue
        XL = X[X[:,3] == i].copy()
        dpar = []
        for j in range(XL.shape[0]): #Loop over lcs in layer
            dist = np.linalg.norm(np.cross(pca.components_[mincompidx],XL[j,:3] - np.array([maxx,maxy,conversionLtoZ(maxl)])))
            dpar.append(dist)
        dparmin = np.argmin(dpar)
        index.append(i)
        distance.append(dpar[dparmin])

        xpcamin = X[:,0][X[:,3] == i][dparmin]
        ypcamin = X[:,1][X[:,3] == i][dparmin]
        lpcamin = X[:,3][X[:,3] == i][dparmin]
        epcamin = X[:,4][X[:,3] == i][dparmin]
        pcaminarr.append([xpcamin,ypcamin,lpcamin,epcamin,conversionLtoZ(lpcamin)])

    pcaminarr = np.array(pcaminarr)


    ####----------------------------- fill array with LC with least dist to PCA with +-N layers from the maxE LC lay--------------------------------####
    pcaminarr = np.array(pcaminarr)
    pcaminarr = pd.DataFrame(pcaminarr, columns=['x','y','layer','energy','z'])
    cleanpcaarr = np.array(pcaminarr[(pcaminarr['layer']<maxl+15) & (pcaminarr['layer']>maxl-12)])

    return cleanpcaarr, pca_axis, origin




def analyze(filename,idx):

    test = uproot3.open(filename)['ana']['hgc']
    print("opened file name:",str(filename))

    gun_pid = test['gunparticle_id'].array()
    gun_en = test['gunparticle_energy'].array()
    gun_eta = test['gunparticle_eta'].array()
    gun_phi = test['gunparticle_phi'].array()

    clus_x = test['cluster2d_x'].array()
    clus_y = test['cluster2d_y'].array()
    clus_z = test['cluster2d_z'].array()
    clus_l = test['cluster2d_layer'].array()
    clus_en = test['cluster2d_energy'].array()
    clus_rechits = test['cluster2d_rechits'].array()
    clus_rechitseed = test['cluster2d_rechitSeed'].array()

    calo_eta = test['calopart_eta'].array()
    calo_phi =  test['calopart_phi'].array()
    calo_pt =  test['calopart_pt'].array()
    calo_en = test['calopart_energy'].array()
    calo_simen =  test['calopart_simEnergy'].array()
    calo_simclusidx =  test['calopart_simClusterIndex'].array()

    multi_eta = test['multiclus_eta'].array()
    multi_phi = test['multiclus_phi'].array()
    multi_clus2d = test['multiclus_cluster2d'].array()
    multi_cluseed = test['multiclus_cl2dSeed'].array()
    multi_en = test['multiclus_energy'].array()

    clus_rechits = awkward0.fromiter(clus_rechits)
    multi_clus2d = awkward0.fromiter(multi_clus2d)

    gun_en = test['gunparticle_energy'].array()
    gun_eta = test['gunparticle_eta'].array()
    gun_phi = test['gunparticle_phi'].array()

    evskip = 0
    ntrkster = []
    for evt in tqdm(range(multi_eta.size),desc='events processed'):
        for algo in pca_algos:

            ntrkster.append(len(multi_eta[evt]))

            for ngun in range(len(gun_eta[evt])):

                mindr = 999.
                gunidx = 999
                trkidx = 999
                drcut = 0.2
                ntklist = []
                for ntk in range(len(multi_eta[evt])):

                    dr = math.sqrt( (gun_eta[evt][ngun] - multi_eta[evt][ntk])**2
                             + (gun_phi[evt][ngun] - multi_phi[evt][ntk])**2
                             )

                    if (dr<drcut) :
                        ntklist.append(ntk)

                minenf = 999
                for drntk in ntklist:
                    enf = multi_en[evt][drntk]/gun_en[evt][ngun]
                    if (abs(enf -1) < minenf) :
                        minenf = abs(enf-1)
                        trkidx = drntk
                        gunidx = ngun



                if (minenf > 0.2) :
                    evskip += 1
                    continue

                trkgunen = gun_en[evt][gunidx]
                trkguneta = gun_eta[evt][gunidx]
                trkgunphi = gun_phi[evt][gunidx]

                trkcluseta = multi_eta[evt][trkidx]
                trkclusphi = multi_phi[evt][trkidx]
                trkclusen = multi_en[evt][trkidx]


                clusE = clus_en[evt][multi_clus2d[evt][trkidx]].flatten()
                clusX = clus_x[evt][multi_clus2d[evt][trkidx]].flatten()
                clusY = clus_y[evt][multi_clus2d[evt][trkidx]].flatten()
                clusL = clus_l[evt][multi_clus2d[evt][trkidx]].flatten()

                feats = np.stack((clusX,clusY,clusL,clusE)).T



                processed_dir = '/grid_mnt/data__data.polcms/cms/tarabini/GENPHOTESTPU2_noSmearing/step3/step4/'
                import os
                import os.path as osp
                if not os.path.exists(processed_dir):
                    os.makedirs(processed_dir)


                cleanedTrk, axis, maxe = cleaning(feats,algo)

                torch.save(Data(x = torch.tensor(feats, dtype=torch.float32),
                    en = torch.tensor(trkgunen,dtype=torch.float32),
                    geta = torch.tensor(trkguneta,dtype=torch.float32),
                    gphi = torch.tensor(trkgunphi,dtype=torch.float32),
                    cphi = torch.tensor(trkclusphi,dtype=torch.float32),
                    ceta = torch.tensor(trkcluseta,dtype=torch.float32),
                    cen = torch.tensor(trkclusen,dtype=torch.float32),
                    ctrk = torch.tensor(cleanedTrk,dtype=torch.float32),
                    centre = torch.tensor(maxe,dtype=torch.float32),
                    axis = torch.tensor(axis,dtype=torch.float32)),
                    osp.join(processed_dir, 'data_{}_{}_{}_{}.pt'.format(algo,idx,evt,ngun)))




    print("skipped events:",evskip)
    print("average #tracksters:",np.mean(ntrkster))
    return 0


raw_dir='/grid_mnt/data__data.polcms/cms/tarabini/GENPHOTESTPU2_noSmearing/step3/'
fnamelist = [filepath for filepath in glob.glob(raw_dir+'STEP3_*.root')]
# fnamelist = [raw_dir+'STEP3_1.root']

#Loop on PCA algos (four options: std, stdAllLCs, eWeighted, eWeightedAllLCs)
pca_algos = ['eWeighted']

fc = 0
for i in tqdm(fnamelist):
    print("processing file:",i)
    analyze(i,fc)
    fc += 1
