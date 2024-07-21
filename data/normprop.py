import numpy as np
import scipy
import math

def bulid_affinity_matrix(feat, k, mask=None, is_ucf=False):
    T, dim = feat.shape
    max_len = min(k*2, T)

    D = np.zeros((T, max_len))
    I = np.zeros((T, max_len))

    rbf = lambda x, sigma: math.exp((-x)/(2*(math.pow(sigma,2))))
    vfunc = np.vectorize(rbf)

    for i in range(T):
        feat_i = np.expand_dims(feat[i, :], 0)
        if i <= max_len//2:
            iter_idxs = np.arange(0, max_len)
        elif i >= T-max_len//2:
            iter_idxs = np.arange(T-max_len, T)
        else:
            iter_idxs = np.arange(i-max_len//2, i+max_len//2)  #
        feat_i_win = feat[iter_idxs]
        if is_ucf:  # use Euclidean similarity for UCF-Crime
            dm = np.power((np.sum((feat_i-feat_i_win)**2, axis=-1) / (dim)), 1/2)
            sim_i = vfunc(dm, 0.1)
        else:  # use cosine similarity for ShanghaiTech
            sim_i = np.dot(feat_i, feat_i_win.transpose(1,0)) / (np.linalg.norm(feat_i, axis=-1) * np.linalg.norm(feat_i_win, axis=-1))
        sim_i = np.squeeze(sim_i)
        # | t_i - t_j | /N
        wt_i = np.abs(iter_idxs - i) / len(iter_idxs)
        wt_i = np.exp(-wt_i)
        wt_i[i-iter_idxs[0]] = 1
        sim_i[i-iter_idxs[0]] = 0
        sim_i = sim_i * wt_i
        D[i] = sim_i
        I[i] = iter_idxs
    
    row_idx = np.arange(T)
    row_idx_rep = np.tile(row_idx, (max_len, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(T, T))
    if mask is not None:
        W = W.multiply(mask)
    W = W + W.T # symmetry

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D  

    return Wn


def normality_propagation(features, abn_num=7, is_ucf=False):
    T, dim = features.shape
    nor_idxs_pre = np.array([0, 1, T-2, T-1])
    Y_input = np.zeros(T)
    Y_input[nor_idxs_pre] = 1
    alpha = 0.99

    k = min(50, T-1)
    Wn = bulid_affinity_matrix(features, k, is_ucf=is_ucf)

    # direct solve version
    Z = np.zeros(T)
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    Z, _ = scipy.sparse.linalg.cg(A, Y_input, tol=1e-6, maxiter=10)
    Z[Z < 0] = 0
    Z[nor_idxs_pre] = np.max(Z)

    sorted_idxs_Z = np.argsort(Z)
    abn_idxs = sorted_idxs_Z[:abn_num]
    pseudo_labels = np.zeros(T)
    pseudo_labels[abn_idxs] = 1
    
    # obtain score video
    score_v = np.std(Z)

    return Z, pseudo_labels, score_v
