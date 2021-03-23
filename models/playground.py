import torch
import torch.nn as nn
import torch.nn.functional as F

transformer = nn.Transformer(d_model=60, nhead=5)
dummy_src = torch.zeros(41, 16, 60)
dummy_tgt = torch.zeros(82, 16, 60)

out = transformer.forward(src=dummy_src, tgt=dummy_tgt)

print(out.shape)