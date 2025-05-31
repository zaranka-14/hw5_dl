import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from config import Config
from torch import nn
import torch.nn.functional as F


def e5learnedbaseline(train_data: Dataset, test_data: Dataset, config: Config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base').to(device)

    train_questions = ["query: " + data['query'] for data in train_data]
    train_answers = ["passage: " + data['answer'] for data in train_data]

    test_questions = ["query: " + data['query'] for data in test_data]
    test_answers = ["passage: " + data['answer'] for data in test_data]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if config.loss_fn == "contrastive":
        loss_fn = ContrastiveLoss()
    else:
        loss_fn = TripletLoss()

    for epoch in tqdm(range(config.epochs), desc="Training model"):
        model.train()
        total_loss = 0

        indices = torch.randperm(len(train_questions))
        train_questions = [train_questions[i] for i in indices]
        train_answers = [train_answers[i] for i in indices]

        for i in range(0, len(train_questions), config.batch_size):
            end = i + config.batch_size

            if i + config.batch_size > len(train_questions):
                end = len(train_questions)

            batch_questions = train_questions[i:end]
            batch_answers = train_answers[i:end]

            neg_indices = torch.randperm(len(train_questions))
            batch_negatives = [train_answers[idx] for idx in neg_indices]

            question_tokens = tokenizer(
                batch_questions,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt').to(device)

            answer_tokens = tokenizer(
                batch_answers,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt').to(device)

            negative_tokens = tokenizer(
                batch_negatives,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt').to(device)

            question_embeddings = model(**question_tokens)
            answer_embeddings = model(**answer_tokens)
            negative_embeddings = model(**negative_tokens)

            question_embeddings = mean_pooling(question_embeddings, question_tokens['attention_mask'])
            question_embeddings = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)

            answer_embeddings = mean_pooling(answer_embeddings, answer_tokens['attention_mask'])
            answer_embeddings = torch.nn.functional.normalize(answer_embeddings, p=2, dim=1)

            negative_embeddings = mean_pooling(negative_embeddings, negative_tokens['attention_mask'])
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)

            scores = torch.matmul(question_embeddings, answer_embeddings.T).to(device)
            labels = torch.arange(len(batch_questions), device=device, dtype=torch.float32)

            if config.loss_fn == "contrastive":
                loss = loss_fn(scores, labels.unsqueeze(0))
            else:
                loss = (question_embeddings, answer_embeddings, negative_embeddings)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del loss

        model.eval()

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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, query_emb, pos_emb, neg_emb=None):
        device = query_emb.device
        pos_emb = pos_emb.to(device)

        query_emb = F.normalize(query_emb, p=2, dim=1)
        pos_emb = F.normalize(pos_emb, p=2, dim=1)

        if neg_emb is None:
            sim_matrix = torch.matmul(query_emb, pos_emb.T)

            mask = ~torch.eye(query_emb.size(0), dtype=torch.bool, device=device)
            neg_sim = sim_matrix.masked_select(mask).view(query_emb.size(0), -1)

            pos_sim = torch.diagonal(sim_matrix).unsqueeze(1)

            loss = F.relu(self.margin - pos_sim + neg_sim).mean()
        else:
            neg_emb = neg_emb.to(device)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)

            pos_sim = torch.sum(query_emb * pos_emb, dim=1)
            neg_sim = torch.sum(query_emb * neg_emb, dim=1)

            loss = F.relu(self.margin - pos_sim + neg_sim).mean()

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        device = anchor.device
        positive = positive.to(device)
        negative = negative.to(device)

        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        pos_dist = torch.sum(anchor * positive, dim=1)
        neg_dist = torch.sum(anchor * negative, dim=1)

        loss = F.relu(neg_dist - pos_dist + self.margin).mean()

        return loss
