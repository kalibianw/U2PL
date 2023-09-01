import matplotlib.pyplot as plt
import numpy as np

import os

from natsort import natsorted


def load_files(num_split, dir_name, fname_only=False):
    log_dir_path = f"../experiments/pascal/{num_split}/{dir_name}/log/"
    log_paths = list()
    for fname in natsorted(os.listdir(log_dir_path)):
        if os.path.isfile(os.path.join(log_dir_path, fname)):
            if fname_only is True:
                log_paths.append(fname)
            else:
                log_paths.append(os.path.join(log_dir_path, fname))
    if len(log_paths) == 0:
        raise FileNotFoundError("Log file not found.")

    return log_paths


def read_log(log_path, best_only=False):
    if isinstance(log_path, str):
        return get_score(log_path)

    elif isinstance(log_path, list):
        best_val_results, class_scores = list(), list()
        for each_log_path in log_path:
            best_val_result, class_score = get_score(each_log_path)

            best_val_results.append(best_val_result)
            class_scores.append(class_score)

        if best_only is False:
            return best_val_results, class_scores
        else:
            best_idx = np.argmax(best_val_results)
            return best_val_results[best_idx], class_scores[best_idx]


def read_score(line: str):
    line_split = line.split(sep='.')

    return float(line_split[0][-2:]) + float(line_split[1][:2]) / 100


def get_score(log_path):
    log = np.array(open(log_path, mode='r').readlines())
    best_val_result = 0
    for line in log:
        if "* epoch" in line:
            current_epoch_val_result = read_score(line)
            if current_epoch_val_result > best_val_result:
                best_val_result = current_epoch_val_result

    class_results = None
    for num_line, line in enumerate(log):
        line = line.astype(str)
        if "* epoch" in line:
            current_epoch_val_result = read_score(line)
            if current_epoch_val_result == best_val_result:
                class_results = log[num_line - 21:num_line]
                break
    class_score = list()
    if class_results is None:
        print(f"Can't find best epoch from {log_path}")
        return None, None

    for class_result in class_results:
        class_score.append(read_score(class_result))

    return best_val_result, class_score


def main(best_only=True, write_figure=False):
    num_splits = [
        # 92,
        183,
        # 366,
        # 732,
        # 1464
    ]
    is_semis = [
        # False,
        True
    ]

    for num_split in num_splits:
        mious = list()
        class_scores = list()
        for is_semi in is_semis:
            dir_name = "ours" if is_semi else "suponly"

            log_paths = load_files(num_split=num_split, dir_name=dir_name)
            miou, class_score = read_log(log_paths, best_only=best_only)
            mious.append(miou)
            class_scores.append(class_score)
        if best_only is False:
            mious = np.squeeze(mious)
            class_scores = np.squeeze(class_scores)
        print(mious)
        print(class_scores)

        if write_figure is True:
            for class_score in class_scores:
                plt.plot(list(range(len(class_score))), class_score)
                plt.xticks(ticks=list(range(len(class_score))))
            plt.legend(load_files(num_split, dir_name, fname_only=True))
            plt.title(f"{num_split} - {mious}")

            plt.tight_layout()

            plt.savefig(f"../fig/{num_split}.png")

            plt.clf()


if __name__ == "__main__":
    main(best_only=False, write_figure=True)
