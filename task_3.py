import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def e5baseline(train_data: Dataset, test_data: Dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base').to(device)

    test_questions = ["query: " + data['query'] for data in test_data]
    test_answers = ["passage: " + data['answer'] for data in test_data]

    batch_size = 32
    test_question_batch = []

    for i in range(0, len(test_questions), batch_size):
        if i + batch_size < len(test_questions) + 1:
            batch_questions = test_questions[i:i + batch_size]
        else:
            batch_questions = test_questions[i:len(test_questions)]

        question_tokens = tokenizer(
            batch_questions,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt').to(device)

        with torch.no_grad():
            model_results = model(**question_tokens)

        batch_predicts = mean_pooling(model_results, question_tokens['attention_mask'])
        batch_predicts = torch.nn.functional.normalize(batch_predicts, p=2, dim=1)
        test_question_batch.append(batch_predicts.cpu())

    test_question_vectors = torch.cat(test_question_batch, dim=0)
    print("Test questions embeddings got\n")

    test_answer_batch = []

    for i in range(0, len(test_answers), batch_size):
        if i + batch_size < len(test_answers) + 1:
            batch_answers = test_answers[i:i + batch_size]
        else:
            batch_answers = test_answers[i:len(test_answers)]

        answer_tokens = tokenizer(
            batch_answers,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt').to(device)

        with torch.no_grad():
            model_results = model(**answer_tokens)

        batch_predicts = mean_pooling(model_results, answer_tokens['attention_mask'])
        batch_predicts = torch.nn.functional.normalize(batch_predicts, p=2, dim=1)
        test_answer_batch.append(batch_predicts.cpu())

    test_answer_vectors = torch.cat(test_answer_batch, dim=0)
    print("Test answers embeddings got\n")

    ids = []

    test_question_vectors = test_question_vectors.to(device)
    test_answer_vectors = test_answer_vectors.to(device)

    for i in tqdm(range(0, len(test_question_vectors), batch_size), desc="Processing batches"):
        if i + batch_size > len(test_question_vectors):
            batch_questions = test_question_vectors[i:len(test_question_vectors)]
        else:
            batch_questions = test_question_vectors[i:i + batch_size]

        batch_similarity = torch.mm(batch_questions, test_answer_vectors.T)
        batch_indices = torch.argsort(batch_similarity, dim=1, descending=True).cpu()
        del batch_similarity
        torch.cuda.empty_cache()
        ids.append(batch_indices)

    sorted_indices = torch.cat(ids, dim=0)

    sorted_indices = sorted_indices.cpu().numpy()

    predictions = []
    for indices in tqdm(sorted_indices, desc="Processing answers"):
        sorted_doc_ids = [test_answers[i] for i in indices]
        predictions.append(sorted_doc_ids)

    print("Cosin similarity proceed\n")
    return test_answers, predictions


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
