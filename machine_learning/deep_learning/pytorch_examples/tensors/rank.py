import torch

bins = torch.tensor([1, 4, 5, 8])  # определили границы (-∞, 1), [4, 5), [5, 8) и [8, ∞).
values = torch.tensor([0, 1, 2, 3, 5, 6, 8, 10, 15, 20, 25])
ranked_values = torch.bucketize(input=values, boundaries=bins, right=True)
print(ranked_values)  # получим группы, к которым принадлежат значения из values.
