import pandas as pd
import numpy as np
from sklearn import metrics


def perf_get_far_frr(scores_imp, scores_gen, reverse_sign=False):
    """
    Given a pair of scores, return FAR and FRR

    :param scores_imp:
    :param scores_gen:
    :param reverse_sign: set tot True for distance scores
    :return:
    """
    if reverse_sign:
        scores_ = -1 * np.concatenate((scores_gen, scores_imp), axis=0)
    else:
        scores_ = np.concatenate((scores_gen, scores_imp), axis=0)

    label_ = np.concatenate(
        (np.ones(scores_gen.shape), np.zeros(scores_imp.shape)), axis=0
    )
    fpr, tpr, thresholds = metrics.roc_curve(label_, scores_, pos_label=1)
    far = fpr
    frr = 1 - tpr

    if reverse_sign:
        # we also remove the first item
        thresholds = -thresholds[1:]
        far = far[1:]
        frr = frr[1:]
    return far, frr, thresholds


def perf_get_scores(D: np.ndarray, label_keys: list):
    """
    :param D: A square distance matrix
    :param label_keys: a list of random key representation of the identities. len(label_keys)
    :return: nonmated_scores, mated_scores
    """
    # __import__("ipdb").set_trace()
    label_keys = np.array(label_keys).reshape(
        -1, 1
    )  # we need to rearrange type here for sklearn.pairwise to use

    # compute mask for the genuine scores from the key
    mask = metrics.pairwise.pairwise_distances(label_keys)
    mask_gen = np.asarray(mask < 0.0001) * 1.0
    mask_gen = np.triu(mask_gen) - np.identity(mask_gen.shape[0])

    # plt.imshow(mask_gen)
    # plt.colorbar()
    # plt.show()

    # compute mask for impostor
    mask_imp = np.ones(mask_gen.shape) - mask_gen - np.identity(mask_gen.shape[0])
    mask_imp = np.triu(mask_imp)
    # plt.imshow(mask_imp); plt.colorbar(); plt.show()

    # just checking
    # plt.imshow(mask_imp+mask_gen)

    # scores are symmetrical and so we need only half of them
    # plt.imshow(np.multiply(D, mask_gen));
    # plt.colorbar()

    indice_gen = np.nonzero(mask_gen)
    mated_scores = D[indice_gen[0], indice_gen[1]]

    indice_imp = np.nonzero(mask_imp)
    nonmated_scores = D[indice_imp[0], indice_imp[1]]

    return nonmated_scores, mated_scores


def get_metrics(far, frr, thresholds, far_list=None):
    if far_list is None:
        far_list = [0.1, 0.05, 0.03, 0.01, 0.001, 0.0001]

    __index_eer = np.argmin(abs(far - frr), axis=0)

    res = {
        "eer": (far[__index_eer] + frr[__index_eer]) / 2,
        "eer_thrd": thresholds[__index_eer]
    }

    def compute_frr_at_far(self_frr, self_far, far_list, tag="frr@far={}%"):
        frr_at_far = [
            self_frr[np.argmin(np.abs(self_far - far))] for far in far_list
        ]
        frr_labels = [tag.format(x * 100) for x in far_list]
        return dict(zip(frr_labels, frr_at_far))

    # Compute frr@frr
    res_frr = compute_frr_at_far(
        frr, far, far_list, tag="frr@far={}%"
    )
    res.update(res_frr)
    return res


df = pd.read_csv('MegaFace_1k.csv')
df = df.set_index('labels')

D = metrics.pairwise.pairwise_distances(df)
nonmated_scores, mated_scores = perf_get_scores(D, df.index, )
far, frr, thresholds = perf_get_far_frr(nonmated_scores, mated_scores, reverse_sign=True)
res = get_metrics(far, frr, thresholds)

for k, v in res.items():
    print(f"{k}:{v}")