from gcommand_loader import GCommandLoader
import torch

dataset = GCommandLoader('./data/train')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=20, pin_memory=True, sampler=None)

for k, (input,label) in enumerate(dataset):
    print(input.size(), label)
