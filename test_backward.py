from src.dhara.model import DharaModel, DharaConfig
import torch
c = DharaConfig(vocab_size=100, hidden_size=64, d_state=32, d_hidden=64)
m = DharaModel(config=c)
x = torch.randint(0, 100, (2, 16))
out = m(x, labels=x, task_ids=torch.zeros(2, dtype=torch.long), language_ids=torch.zeros(2, 16, dtype=torch.long))
print(f'Loss: {out.loss.item():.4f}')
out.loss.backward()
print('Backward OK')
