from datasets import Dataset
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def tfidf(train_data: Dataset, test_data: Dataset) -> (list, list):

    test_questions = [data['query'] for data in test_data]
    test_answers = [data['answer'] for data in test_data]

    documents_to_fit = train_data['answer']
    documents_to_fit.extend(train_data['query'])

    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents_to_fit)
    test_question_vectors = vectorizer.transform(test_questions)
    test_answers_vectors = vectorizer.transform(test_answers)

    predictions = []

    for question in tqdm(test_question_vectors, desc="processing task 2"):
        similarities = cosine_similarity(question, test_answers_vectors).flatten()

        sorted_indices = np.argsort(similarities)[::-1]
        sorted_doc_ids = [test_answers[i] for i in sorted_indices]

        predictions.append(sorted_doc_ids)

    return test_answers, predictions
