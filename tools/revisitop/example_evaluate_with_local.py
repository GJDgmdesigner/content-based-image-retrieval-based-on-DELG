# EXAMPLE_EVALUATE  Code to evaluate example results on ROxford and RParis datasets.
# Revisited protocol has 3 difficulty setups: Easy (E), Medium (M), and Hard (H), 
# and evaluates the performance using mean average precision (mAP), as well as mean precision @ k (mP@k)
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018
# Written by Yang Min, Shi Baorong, 2020

import os, sys
import os.path as osp
import io
import pickle
import copy
import gc

import numpy as np
import pydegensac

from scipy.io import loadmat
from scipy import spatial
import matplotlib.pyplot as plt
from concurrent import futures

from skimage import feature
from skimage import io as skio

from dataset import configdataset
from compute import compute_map

# data_root =  osp.abspath(osp.dirname(osp.dirname(__file__)))  #/home/phd21_yiming_lin/Project/DELG-Implementation-master/tools/revisitop/
data_root = "/Data_HDD/yiming/datasets/"                              # edit1

# test_dataset = 'roxford5k'
test_dataset = 'rparis6k'  
print('>> {}: Evaluating test dataset...'.format(test_dataset)) 
IMAGE_PATH = os.path.join(data_root, 'datasets', test_dataset, 'jpg')           

# GLOBAL_FEATURE_PATH='roxford5k512gem_delg_res50_3global.mat'
# LOCAL_FEATURE_PATH='roxford5k__s512_localfea.pickle'

GLOBAL_FEATURE_PATH='rparis6k_resnet_rsfm120k_gem.mat'
LOCAL_FEATURE_PATH='roxford5k__s512_localfea.pickle'
ASMK_SCORE_PATH='localfeatures/delg_asmk_6553.pkl'

print("Image_p"+IMAGE_PATH)


NUM_RERANK = 100
MAX_REPROJECTION_ERROR = 20.0
MAX_RANSAC_ITERATIONS = 1000
HOMOGRAPHY_CONFIDENCE = 1.0
MATCHING_THRESHOLD = 1.0
MAX_DISTANCE = 0.99
USE_RATIO_TEST = False
DRAW_MATCHES = False


def global_search(global_feature_path):
    """ rank by global descriptors """ 
    #features = loadmat(os.path.join(data_root, 'features', global_feature_path))             #edit2
    features = loadmat("/Data_HDD/yiming/gjd/features/gnd_rparis6k.mat")
    Q = features['Q']
    X = features['X']

    sim = np.dot(X, Q.T)
    ranks = np.argsort(-sim, axis=0)
    #np.save("ranks_before_gv.npy", ranks)
    return ranks


def compute_putative_matching_keypoints(test_keypoints,
                                        test_descriptors,
                                        train_keypoints,
                                        train_descriptors,
                                        use_ratio_test=USE_RATIO_TEST,
                                        matching_threshold=float(MATCHING_THRESHOLD),
                                        max_distance=float(MAX_DISTANCE)):
    """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""
    train_descriptor_tree = spatial.cKDTree(train_descriptors)

    if use_ratio_test:

        distances,matches=train_descriptor_tree.query(
            test_descriptors,k=2,n_jobs=-1
        )
        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]
        test_matching_keypoints=np.array([
            test_keypoints[i,]
            for i in range(test_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])
        train_matching_keypoints=np.array([
            train_keypoints[matches[i][0],]
            for i in range(train_kp_count)
            if distances[i][0] < matching_threshold*distances[i][1]
        ])

    else:
        _, matches = train_descriptor_tree.query(
              test_descriptors, distance_upper_bound=max_distance)

        test_kp_count = test_keypoints.shape[0]
        train_kp_count = train_keypoints.shape[0]

        test_matching_keypoints = np.array([
              test_keypoints[i,]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
        train_matching_keypoints = np.array([
              train_keypoints[matches[i],]
              for i in range(test_kp_count)
              if matches[i] != train_kp_count
          ])
    return test_matching_keypoints, train_matching_keypoints 


def compute_num_inliers(test_keypoints, 
                        test_descriptors, 
                        train_keypoints,
                        train_descriptors,
                        use_ratio_test=False,
                        draw_matches=True,
                        query_im_array=None,
                        index_im_array=None):
    """Returns the number of RANSAC inliers."""
    test_match_kp, train_match_kp = \
            compute_putative_matching_keypoints(test_keypoints, 
                                                test_descriptors, 
                                                train_keypoints, 
                                                train_descriptors,
                                                use_ratio_test=use_ratio_test)
    if test_match_kp.shape[
          0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`
        return 0, b''

    try:
        _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                            MAX_REPROJECTION_ERROR,
                                            HOMOGRAPHY_CONFIDENCE,
                                            MAX_RANSAC_ITERATIONS)
    except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
        return 0, b''

    inliers = mask if mask is not None else []

    match_viz_bytes = b''
    if isinstance(query_im_array, np.ndarray) and isinstance(index_im_array, np.ndarray) and draw_matches:
        query_im_scale_factors = [1.0, 1.0]
        index_im_scale_factors = [1.0, 1.0]
        inlier_idxs = np.nonzero(inliers)[0]
        _, ax = plt.subplots()
        ax.axis('off')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        feature.plot_matches(
            ax,
            query_im_array,
            index_im_array,
            test_match_kp * query_im_scale_factors,
            train_match_kp * index_im_scale_factors,
            np.column_stack((inlier_idxs, inlier_idxs)),
            only_matches=False)

        match_viz_io = io.BytesIO()
        plt.savefig(match_viz_io, format='jpeg', bbox_inches='tight', pad_inches=0)
        match_viz_bytes = match_viz_io.getvalue()
    
    return int(copy.deepcopy(mask).astype(np.float32).sum()), match_viz_bytes


def rerankGV(cfg, local_feature_path, ranks_before_gv, ranks_after_gv=None):
    gnd = cfg['gnd']
    print('>> reranking ...')
    ranks_after_gv = ranks_before_gv
    train_ids = [x + cfg['ext'] for x in cfg['imlist']]
    test_ids = [x + cfg['qext'] for x in cfg['qimlist']]

    with open(osp.join(data_root, "localfeatures", local_feature_path), "rb") as fin:
        local_features = pickle.load(fin)

    for i in range(len(test_ids)):
        test_img = test_ids[i]
        if i % 7 == 0:
            print(">> start rerank {}: {}.".format(i, test_img))
        test_array = skio.imread(osp.join(IMAGE_PATH, test_img))
        locations=local_features[test_img]["locations"]
        descriptors= local_features[test_img]["descriptors"]

        inliers_numrerank = np.zeros(NUM_RERANK)
        for j in range(NUM_RERANK):
            if ranks_before_gv[j, i] in gnd[i]['junk']:
                continue
            index_img = train_ids[ranks_before_gv[j, i]]
            index_array = skio.imread(osp.join(IMAGE_PATH, index_img))
            tlocations=local_features[index_img]["locations"]
            tdescriptors = local_features[index_img]["descriptors"]
            try:
                num_inliers, match_vis_bytes = compute_num_inliers(locations, descriptors,
                                                     tlocations, tdescriptors,
                                                     use_ratio_test=USE_RATIO_TEST,
                                                     draw_matches=DRAW_MATCHES,
                                                     query_im_array=test_array, 
                                                     index_im_array=index_array)
                # local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE
                if DRAW_MATCHES:
                    with open(osp.join(data_root, "datasets", test_dataset, "savematches", \
                        test_img.split(".")[0] + "," + index_img.split(".")[0] + ".jpg"), "wb") as fout:
                        fout.write(match_vis_bytes)
                inliers_numrerank[j] = num_inliers
            except:
                continue
        ranks_after_gv[:NUM_RERANK, i] = ranks_before_gv[np.argsort(-1 * inliers_numrerank), i]
    return ranks_before_gv, ranks_after_gv


def localRank(cfg, tuple_local_features, train_ids, test_ids, ranks_before_gv):
    gnd = cfg['gnd']
    query_name, part_local_features = tuple_local_features
    print(">> rerank {}".format(query_name))

    i = test_ids.index(query_name)
    test_array = skio.imread(osp.join(IMAGE_PATH,query_name))    
    locations = part_local_features[query_name]["locations"]
    descriptors = part_local_features[query_name]["descriptors"]
    inliers_numrerank = np.zeros(NUM_RERANK)  
    
    for j in range(NUM_RERANK):
        if ranks_before_gv[j, i] in gnd[i]['junk']:
            continue
        index_img = train_ids[ranks_before_gv[j, i]]
        index_array = skio.imread(osp.join(IMAGE_PATH, index_img))
        tlocations=part_local_features[index_img]["locations"]
        tdescriptors = part_local_features[index_img]["descriptors"]
        try:
            num_inliers, match_vis_bytes = compute_num_inliers(locations, descriptors,
                                                 tlocations, tdescriptors,
                                                 use_ratio_test=USE_RATIO_TEST,
                                                 draw_matches=DRAW_MATCHES,
                                                 query_im_array=test_array, 
                                                 index_im_array=index_array)
            # local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE
            if DRAW_MATCHES:
                with open(osp.join(data_root, "datasets", test_dataset, "testcv2", test_img.split(".")[0] + \
                        "," + index_img.split(".")[0] + ".jpg"), "wb") as fout:
                    fout.write(match_vis_bytes)
            inliers_numrerank[j] = num_inliers
        except:
            continue
    return i, inliers_numrerank


def rerankGV_mulprocess(cfg, local_feature_path, ranks_before_gv, ranks_after_gv=None):
    print('>> mulprocess reranking ...')
    ranks_after_gv = ranks_before_gv
    train_ids = [x + cfg['ext'] for x in cfg['imlist']]
    test_ids = [x + cfg['qext'] for x in cfg['qimlist']]

    # with open(osp.join(data_root, "localfeatures", local_feature_path), "rb") as fin:          #edit 4
    with open("/Data_HDD/yiming/gjd/localfea.pickle_1", "rb") as fin:
        local_features = pickle.load(fin)
    print(">> local features loaded ...")
    
    # fix multiprocess memery problem :struct.error: 'i' format requires -2147483648 <= number <= 2147483647
    N_localfeatures = []
    for query_rank in range(len(test_ids)):
        query_idx = test_ids[query_rank]
        fea_dic = {}
        fea_dic[query_idx] = local_features[query_idx]
        for k in range(NUM_RERANK):
            index_rank = ranks_before_gv[k, query_rank]
            index_idx = train_ids[index_rank]
            fea_dic[index_idx] = local_features[index_idx]
        N_localfeatures.append((query_idx, fea_dic))
    
    del local_features
    gc.collect()
        
    with futures.ProcessPoolExecutor(max_workers=24) as executor:
        executor_dict = {executor.submit(localRank, cfg, tuple_fea, train_ids, test_ids, ranks_before_gv): \
                        tuple_fea for tuple_fea in N_localfeatures}
        
    for future in futures.as_completed(executor_dict):
        query_idx, inliers_numrerank = future.result()
        ranks_after_gv[:NUM_RERANK, query_idx] = ranks_before_gv[np.argsort(-1 * inliers_numrerank), query_idx]
    return ranks_before_gv, ranks_after_gv


def reportMAP(test_dataset, cfg, ranks):
    gnd = cfg['gnd']
    # evaluate ranks
    ks = [1, 5, 10]

    # search for easy
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
        gnd_t.append(g)
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

    print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, 
          np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
    print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), 
          np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))


def main():
    cfg = configdataset(test_dataset, data_root)
    ranks = global_search(GLOBAL_FEATURE_PATH) 
    reportMAP(test_dataset, cfg, ranks)
    
    #_, ranks_after_gv = rerankGV(cfg, LOCAL_FEATURE_PATH, ranks)
    _, ranks_after_gv = rerankGV_mulprocess(cfg, LOCAL_FEATURE_PATH, ranks)
    #np.save("ranks_after_gv.npy", ranks_after_gv)
    reportMAP(test_dataset, cfg, ranks_after_gv)

    print("Done!")


def rankASMK():
    cfg = configdataset(test_dataset, data_root)

    ranks = global_search(GLOBAL_FEATURE_PATH) 

    reportMAP(test_dataset, cfg, ranks)
    
    with open(ASMK_SCORE_PATH, "rb") as fin:
        scores = pickle.load(fin)
        print("scores", scores.shape)
    
    ranks = ranks.T
    ranks_after = ranks

    for i in range(ranks.shape[0]):
        asmk_scores = scores[i, ranks[i, :NUM_RERANK]]
        ranks_after[i, :NUM_RERANK] = ranks[i, np.argsort(-1 * asmk_scores)]

    reportMAP(test_dataset, cfg, ranks_after.T)
    print("Done!")

   
if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('tools.py command', file=sys.stderr)
