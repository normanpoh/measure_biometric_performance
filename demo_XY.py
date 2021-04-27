import pandas as pd
import numpy as np
from sklearn import metrics

"""
Use this tutorial to get non-mated and mated scores from a gallery matrix X 
and a probe matrix Y.
"""

def convert_labels2numerical(labels):
    # Convert labels which is alpha numerical to integer values
    label_set = set(labels)  # makes the elements unique
    label_set = list(label_set)
    label_set.sort()  # keeps order
    lookup = dict(zip(label_set, range(len(label_set))))
    numerical_id = [lookup[x] for x in labels]
    return np.asarray(numerical_id), lookup

def near(s, value, machine_precision=np.sqrt(np.finfo(float).eps)):
    # a test for s==value
    return abs(s - value) < machine_precision

def get_df_scores(gallery_labels, probe_labels, scores):
    return pd.DataFrame.from_dict({
        "gallery_labels": gallery_labels,
        "probe_labels": probe_labels,
        "scores": scores})


def perf_get_scores_XY(D, X_labels, Y_labels):
    """
    X_labels are labels (subject_id) of the enrolment samples
    Y_labels are labels (subject_id) of the probe samples
    X_index is a list of indices associated with X_labels
    Y_index is a list of indices associated with Y_labels
    If X_index and Y_index are supplied, the indexes to respective chosen samples will be returned

    """
    def reshape_(x):
        return np.array(x).reshape(-1, 1)

    D_labels = metrics.pairwise_distances(
        reshape_(X_labels), reshape_(Y_labels), metric="l1"
    )
    assert D.shape == D_labels.shape
    mask_gen = near(D_labels, 0) * 1.0

    D = np.asarray(D)
    # checking
    # mask_gen.sum(axis=0)
    # mask_gen.sum(axis=1)

    # D__ = np.multiply( np.array(self.D), np.array( D_labels))
    # D__ = self.D * mask_gen

    D_gallery_labels = metrics.pairwise_distances(
        reshape_(X_labels), reshape_(Y_labels), metric=lambda x,y: x
    )
    D_probe_labels = metrics.pairwise_distances(
        reshape_(X_labels), reshape_(Y_labels), metric=lambda x, y: y
    )

    # genuine scores
    indice_gen = np.nonzero(mask_gen)
    scores_gen = D[indice_gen[0], indice_gen[1]]
    gallery_labels_gen = D_gallery_labels[indice_gen[0], indice_gen[1]]
    probe_labels_gen = D_probe_labels[indice_gen[0], indice_gen[1]]

    df_scores_gen= get_df_scores(gallery_labels_gen, probe_labels_gen, scores_gen)
    df_scores_gen["is_mated"] = True

    # impostor scores
    indice_imp = np.nonzero(1 - mask_gen)
    scores_imp = D[indice_imp[0], indice_imp[1]]
    gallery_labels_imp = D_gallery_labels[indice_imp[0], indice_imp[1]]
    probe_labels_imp = D_probe_labels[indice_imp[0], indice_imp[1]]
    df_scores_imp = get_df_scores(gallery_labels_imp, probe_labels_imp, scores_imp)
    df_scores_imp["is_mated"] = False
    return pd.concat([df_scores_gen, df_scores_imp])


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


# Load gallery features
df_gallery = pd.read_csv('MegaFace_1k.csv')
df_gallery = df_gallery.set_index('labels')

# Load probe features
df_probe = pd.read_csv('MegaFace_1k.csv') # load your own probe file
df_probe = df_probe.set_index('labels')

# Combine the gallery and probe labels and create a common index
_, lookup = convert_labels2numerical(np.hstack([df_gallery.index.values, df_probe.index.values]))
reverse_lookup = dict(zip(lookup.values(), lookup.keys()))

gallery_labels = [lookup[x] for x in df_gallery.index]
probe_labels = [lookup[x] for x in df_probe.index]

D = metrics.pairwise.pairwise_distances(df_gallery, df_probe)

df_scores = perf_get_scores_XY(D, gallery_labels, probe_labels)

df_scores["gallery_labels"] = df_scores["gallery_labels"].astype(int)
df_scores["probe_labels"] = df_scores["probe_labels"].astype(int)

df_scores["gallery_id"]= [reverse_lookup[x] for x in df_scores["gallery_labels"]]
df_scores["probe_id"]= [reverse_lookup[x] for x in df_scores["probe_labels"]]

# Reset the index
df_scores.reset_index(drop=True, inplace=True)

selected_var = ['gallery_id', 'probe_id','scores']
df_scores[selected_var].to_csv("scores.csv", index=False)

# Compute metrics
filter_is_mated = df_scores["is_mated"] == True
far, frr, thresholds = perf_get_far_frr(
    df_scores.loc[~filter_is_mated, "scores"], df_scores.loc[filter_is_mated, "scores"],
    reverse_sign=True
)
res = get_metrics(far, frr, thresholds)

for k, v in res.items():
    print(f"{k}:{v}")
