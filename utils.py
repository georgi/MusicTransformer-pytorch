import os
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm


def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def sample(model, sample_length, prime_sequence, device, temperature=1):
    model.eval()
    input_sequence = prime_sequence.copy()
    input_tensor = torch.LongTensor(input_sequence).unsqueeze(0).to(device)

    for i in tqdm(range(sample_length)):
        out = model(input_tensor)[0, -1, :]
        probs = F.softmax(out / temperature, dim=0)
        top = torch.topk(probs, 5)
        for i in range(len(probs)):
            if i not in top.indices:
                probs[i] = 0
        c = torch.multinomial(probs, 1)
        input_tensor = torch.cat([input_tensor[:, 1:], c[None]], dim=1)
        input_sequence.append(c.item())

    return input_sequence
