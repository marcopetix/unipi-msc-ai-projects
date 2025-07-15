import scipy as sp
import torch

def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score

def pearson_correlation_score(predictions, labels):
    mean_pred = torch.mean(predictions)
    mean_labels = torch.mean(labels)

    covariance = torch.sum((predictions - mean_pred) * (labels - mean_labels))
    std_pred = torch.sqrt(torch.sum((predictions - mean_pred) ** 2))
    std_labels = torch.sqrt(torch.sum((labels - mean_labels) ** 2))

    pearson_correlation = covariance / (std_pred * std_labels)

    return pearson_correlation

def pearson_correlation_loss(predictions, labels):
    return 1 - pearson_correlation_score(predictions, labels)


