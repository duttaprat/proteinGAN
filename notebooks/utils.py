import torch


def batched_index_select(data, dim, indicies):
    """

    Args:
        data: tensor
        dim: from which dimention to select
        indicies: to select

    Returns:
        Selects elements of given indices in given dimention
    """
    views = [data.shape[0]] + [1 if i != dim else -1 for i in range(1, len(data.shape))]
    expanse = list(data.shape)
    expanse[0] = -1
    expanse[dim] = -1
    indicies = indicies.view(views).expand(expanse)
    return torch.gather(data, dim, indicies)

def bert_loss(logits, label_ids, vocab_size):
    """

    Args:
        logits: logits return from model (includes log_softmax
        label_ids:  correct label ids
        vocab_size: size of vocabulary

    Returns:
        Returns loss per each guess
    """
    log_probs = logits
    original_shape = label_ids.shape
    label_ids = torch.reshape(label_ids, [-1])
    one_hot_labels = torch.zeros(len(label_ids), vocab_size).cuda()
    one_hot_labels = one_hot_labels.scatter_(1, label_ids.unsqueeze(1), 1.)
    one_hot_labels = torch.reshape(one_hot_labels, [*original_shape, vocab_size])
    per_example_loss = -torch.sum(log_probs * one_hot_labels, dim=-1)
    return per_example_loss


def append(x, y):
    """

    Args:
        x: first array
        y: second array

    Returns:
        Concatenates two arrays. If first is None, just assigns to value of second one
    """
    if x is None:
        x = y
    else:
        x = torch.cat((x, y))
    return x