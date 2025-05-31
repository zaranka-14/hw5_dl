from datasets import load_dataset


def get_dataset(test_size: float, rand_state: int):
    ds = load_dataset("sentence-transformers/natural-questions")
    ds = ds['train'].train_test_split(test_size=test_size, seed=rand_state)
    ds['train'].save_to_disk('data/train_ds')
    ds['test'].save_to_disk('data/test_ds')


def recall_dog_k(target: list, predict: list[list], k: int) -> float:
    positive = 0

    if len(target) != len(predict):
        raise IndexError

    for index in range(len(target)):
        if target[index] in predict[index][:k]:
            positive += 1

    print(f'Recall {str(k)} proceed')
    return positive * 1.0 / len(target)


def MRR(target: list, predict: list[list]) -> float:

    if len(target) != len(predict):
        raise IndexError

    ranks = [0.0]
    for index in range(len(target)):
        for jindex in range(len(predict[index])):
            if target[index] == predict[index][jindex]:
                ranks.append(1.0 / (jindex + 1))
                break

    print("MRR proceed")
    return sum(ranks) * 1.0 / len(target)
