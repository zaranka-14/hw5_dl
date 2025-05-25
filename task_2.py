from datasets import Dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def tfidf(train_data: Dataset, test_data: Dataset) -> (list, list):
    documents = {}
    documents_ids = []
    documents_texts = []

    for index in range(len(train_data)):
        answer = train_data[index]['answer']

        if answer not in documents.values():
            documents[index] = answer
            documents_ids.append(index)
            documents_texts.append(answer)

    test_questions = [data['query'] for data in test_data]

    vectorizer = TfidfVectorizer()
    answer_vectors = vectorizer.fit_transform(documents_texts)
    test_question_vectors = vectorizer.transform(test_questions)

    predictions = []

    for question in tqdm(test_question_vectors, desc="processing task 2"):
        similarities = cosine_similarity(question, answer_vectors).flatten()

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
