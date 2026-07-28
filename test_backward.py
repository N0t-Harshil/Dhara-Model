from src.methos_v3.model import MethosV3Model, MethosV3Config
import torch
c = MethosV3Config(vocab_size=100, hidden_size=64, d_state=32, d_hidden=64)
m = MethosV3Model(config=c)
x = torch.randint(0, 100, (2, 16))
out = m(x, labels=x, task_ids=torch.zeros(2, dtype=torch.long), language_ids=torch.zeros(2, 16, dtype=torch.long))
print(f'Loss: {out.loss.item():.4f}')
out.loss.backward()
print('Backward OK')
