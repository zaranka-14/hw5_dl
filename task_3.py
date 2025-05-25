import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity


def e5baseline(train_data: Dataset, test_data: Dataset):

    documents = {}
    documents_ids = []
    documents_texts = []

    for index in range(len(train_data)):
        answer = train_data[index]['answer']

        if answer not in documents.values():
            documents[index] = answer
            documents_ids.append(index)
            documents_texts.append('passage: ' + answer)

    test_questions = [('query: ' + data['query']) for data in test_data]
    test_answers = [('passage: ' + data['answer']) for data in test_data]

    n = len(test_questions)

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')

    test_questions.extend(test_answers)

    input_texts = test_questions

    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    queries = batch_dict['input_ids'][:n]
    passages = batch_dict['input_ids'][n:]

    predictions = []

    for query in tqdm(queries, desc="processing task 3"):
        query = query.unsqueeze(0)
        similarities = cosine_similarity(query, passages).flatten()

        sorted_indices = np.argsort(similarities)[::-1]
        sorted_doc_ids = [documents_ids[i] for i in sorted_indices]

        predictions.append(sorted_doc_ids)

    test_targets = []
    for element in test_data:
        for d_id, d_text in documents.items():
            if d_text == element['answer']:
                test_targets.append(d_id)
                break
        else:
            test_targets.append(-1)

    valid_indices = [i for i, target in enumerate(test_targets) if target != -1]
    filtered_targets = [test_targets[i] for i in valid_indices]
    filtered_predictions = [predictions[i] for i in valid_indices]

    return filtered_targets, filtered_predictions
