import numpy as np
import torch
from datasets import Dataset
from torch import Tensor, nn, cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from config import Config
import torch.nn.functional as F


def e5learnedbaseline(train_data: Dataset, test_data: Dataset, config: Config):
    documents = {}
    documents_ids = []
    documents_texts = []

    for index in range(len(train_data)):
        answer = train_data[index]['answer']

        if answer not in documents.values():
            documents[index] = answer
            documents_ids.append(index)
            documents_texts.append('passage: ' + answer)

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

    train_queries = ['query: ' + data['query'] for data in train_data]
    train_answers = ['passage: ' + data['answer'] for data in train_data]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    for epoch in tqdm(range(config.epochs), desc="training model task 4"):
        total_loss = 0
        for i in range(0, len(train_data), config.batch_size):
            batch_end = i + config.batch_size
            if i + config.batch_size > len(train_data) - 1:
                batch_end = len(train_data)
            batch_queries = train_queries[i:batch_end]
            batch_answers = train_answers[i:batch_end]

            query_inputs = tokenizer(batch_queries, padding=True, truncation=True,
                                     max_length=512, return_tensors='pt').to(device)
            answer_inputs = tokenizer(batch_answers, padding=True, truncation=True,
                                      max_length=512, return_tensors='pt').to(device)

            query_emb = average_pool(model(**query_inputs).last_hidden_state,
                                     query_inputs['attention_mask'])
            answer_emb = average_pool(model(**answer_inputs).last_hidden_state,
                                      answer_inputs['attention_mask'])

            query_emb = F.normalize(query_emb, p=2, dim=1)
            answer_emb = F.normalize(answer_emb, p=2, dim=1)

            if config.loss == "Contrastive":
                loss = ContrastiveLoss()(query_emb, answer_emb)
            else:
                indices = torch.randperm(len(train_data))[:batch_end - i].tolist()
                neg_ans = [train_answers[j] for j in indices[batch_end - i - 1:]]
                neg_emb = tokenizer(neg_ans, padding=True, truncation=True,
                                    max_length=512, return_tensors='pt').to(device)
                neg_emb = average_pool(model(**neg_emb).last_hidden_state, neg_emb['attention_mask'])
                loss = TripletLoss()(query_emb, answer_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_data):.4f}')

    model.eval()
    passage_emb = []

    with torch.no_grad():
        for i in range(0, len(documents_texts), config.batch_size):
            batch_end = i + config.batch_size
            if i + config.batch_size > len(train_data) - 1:
                batch_end = len(train_data)
            batch = documents_texts[i:batch_end]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors='pt').to(device)
            emb = average_pool(model(**inputs).last_hidden_state, inputs['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
            passage_emb.append(emb)

    passage_emb = torch.cat(passage_emb)

    test_questions = ['query: ' + data['query'] for data in test_data]
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(test_questions), config.batch_size), desc="evaluating task 4"):
            batch_end = i + config.batch_size
            if i + config.batch_size > len(train_data) - 1:
                batch_end = len(train_data)
            batch = test_questions[i:batch_end]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors='pt').to(device)
            emb = average_pool(model(**inputs).last_hidden_state, inputs['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)

            similarities = cosine_similarity(emb, passage_emb)
            for scores in similarities:
                sorted_ids = np.argsort(scores)[::-1]
                predict = [documents_ids[ind] for ind in sorted_ids]
                predictions.append(predict)

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


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, question, answer):
        batch_sim = torch.mm(question, answer.T)
        pos_sim = batch_sim.diag()
        neg_sim = batch_sim - torch.diag_embed(batch_sim.diag())
        loss = torch.mean(torch.clamp(self.margin - pos_sim.unsqueeze(1) + neg_sim, min=0))
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.cosine_similarity(anchor, positive)
        neg_dist = F.cosine_similarity(anchor, negative)
        losses = torch.relu(neg_dist - pos_dist + self.margin)
        return torch.mean(losses)
