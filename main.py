from task_1 import get_dataset, recall_dog_k, MRR
from task_2 import tfidf
from task_3 import e5baseline
from task_4 import e5learnedbaseline
from task_5 import e5learnednotrandombaseline

from config import Config
from datasets import load_from_disk


if __name__ == '__main__':
    """
    get_dataset(0.2, 42)
    """

    train_ds = load_from_disk('data/train_ds')
    val_ds = load_from_disk('data/test_ds')
    """
    filtered_targets, filtered_predictions = tfidf(train_ds, val_ds)

    with open("results/task_2_results.txt", "w") as f:
        f.write(f"recall_1: {recall_dog_k(filtered_targets, filtered_predictions, 1)}\n")
        f.write(f"recall_3: {recall_dog_k(filtered_targets, filtered_predictions, 3)}\n")
        f.write(f"recall_10: {recall_dog_k(filtered_targets, filtered_predictions, 10)}\n")
        f.write(f"MRR: {MRR(filtered_targets, filtered_predictions)}\n")
    """
    """
    t3_filtered_targets, t3_filtered_predictions = e5baseline(train_ds, val_ds)

    with open("results/task_3_results.txt", "w") as f:
        f.write(f"recall_1: {recall_dog_k(t3_filtered_targets, t3_filtered_predictions, 1)}\n")
        f.write(f"recall_3: {recall_dog_k(t3_filtered_targets, t3_filtered_predictions, 3)}\n")
        f.write(f"recall_10: {recall_dog_k(t3_filtered_targets, t3_filtered_predictions, 10)}\n")
        f.write(f"MRR: {MRR(t3_filtered_targets, t3_filtered_predictions)}\n")
    """
    t4_1_config = Config(10, "Contrast", 0.0001, 32)
    t4_1_filtered_targets, t4_1_filtered_predictions = e5learnedbaseline(train_ds, val_ds, t4_1_config)
    with open("results/task_4_1_results.txt", "w") as f:
        f.write(f"recall_1: {recall_dog_k(t4_1_filtered_targets, t4_1_filtered_predictions, 1)}\n")
        f.write(f"recall_3: {recall_dog_k(t4_1_filtered_targets, t4_1_filtered_predictions, 3)}\n")
        f.write(f"recall_10: {recall_dog_k(t4_1_filtered_targets, t4_1_filtered_predictions, 10)}\n")
        f.write(f"MRR: {MRR(t4_1_filtered_targets, t4_1_filtered_predictions)}\n")

    t4_2_config = Config(10, "Triplet", 0.0001, 32)
    t4_2_filtered_targets, t4_2_filtered_predictions = e5learnedbaseline(train_ds, val_ds, t4_2_config)
    with open("results/task_4_2_results.txt", "w") as f:
        f.write(f"recall_1: {recall_dog_k(t4_2_filtered_targets, t4_2_filtered_predictions, 1)}\n")
        f.write(f"recall_3: {recall_dog_k(t4_2_filtered_targets, t4_2_filtered_predictions, 3)}\n")
        f.write(f"recall_10: {recall_dog_k(t4_2_filtered_targets, t4_2_filtered_predictions, 10)}\n")
        f.write(f"MRR: {MRR(t4_2_filtered_targets, t4_2_filtered_predictions)}\n")

    t5_config = Config(10, "Triplet", 0.0001, 32)
    t5_filtered_targets, t5_filtered_predictions = e5learnednotrandombaseline(train_ds, val_ds, t5_config)
    with open("results/task_5_results.txt", "w") as f:
        f.write(f"recall_1: {recall_dog_k(t5_filtered_targets, t5_filtered_predictions, 1)}\n")
        f.write(f"recall_3: {recall_dog_k(t5_filtered_targets, t5_filtered_predictions, 3)}\n")
        f.write(f"recall_10: {recall_dog_k(t5_filtered_targets, t5_filtered_predictions, 10)}\n")
        f.write(f"MRR: {MRR(t5_filtered_targets, t5_filtered_predictions)}\n")
