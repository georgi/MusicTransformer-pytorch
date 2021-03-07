import os
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


def sample(model, sample_length, prime_sequence, device, top_p, top_k=10):
    import torch
    import torch.nn.functional as F
    model.eval()
    input_sequence = prime_sequence.copy()
    input_tensor = torch.LongTensor(input_sequence).unsqueeze(0).to(device)

    for i in tqdm(range(sample_length)):
        logits = model(input_tensor).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        to_remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        to_remove[0] = False  # always incude top result
        to_remove[top_k:] = True  # at most include top K results
        indices_to_remove = sorted_indices[to_remove]
        probs.scatter_(0, indices_to_remove, 0.0)
        c = torch.multinomial(probs, 1)
        input_tensor = torch.cat([input_tensor[:, 1:], c[None]], dim=1)
        input_sequence.append(c.item())

    return input_sequence
