import matplotlib.pyplot as plt
import itertools
import numpy as np
import os

from natsort import natsorted


def get_label_names():
    label_names = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    return label_names


def plot_confusion_matrix(
    cm,
    target_names=None,
    cmap=None,
    normalize=True,
    labels=True,
    title="Confusion matrix",
    export_name="",
):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(
                    j,
                    i,
                    "{:0.2f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            else:
                plt.text(
                    j,
                    i,
                    "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
    )
    plt.tight_layout()
    if export_name != "":
        if os.path.exists(os.path.dirname(export_name)) is False:
            os.makedirs(os.path.dirname(export_name))
        plt.savefig(export_name, dpi=300)
    else:
        plt.show()


def main():
    root_dir_path = "C:/Users/admin/Downloads/PASCAL VOC 2012 outputs/npy"

    for npy_name in natsorted(os.listdir(root_dir_path)):
        arr = np.load(os.path.join(root_dir_path, npy_name))
        plot_confusion_matrix(
            cm=arr,
            target_names=get_label_names(),
            title=os.path.basename(npy_name),
            export_name=f"tmp/cm_plot/{os.path.splitext(os.path.basename(npy_name))[0]}.png",
        )


if __name__ == "__main__":
    main()
