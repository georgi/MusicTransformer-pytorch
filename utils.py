import os
import torch
import torch.nn.functional as F
# from custom.config import config


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
        input_tensor = torch.cat([input_tensor[:,1:], c[None]], dim=1)
        input_sequence.append(c.item())

    return input_sequence


def attention_image_summary(name, attn, step=0, writer=None):
    """Compute color image summary.
    Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
    """
    num_heads = attn.size(1)
    # [batch, query_length, memory_length, num_heads]
    image = attn.permute(0, 2, 3, 1)
    image = torch.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = F.pad(image, [0,  -num_heads % 3, 0, 0, 0, 0, 0, 0,])
    image = split_last_dimension(image, 3)
    image = image.max(dim=4).values
    grid_image = torchvision.utils.make_grid(image.permute(0, 3, 1, 2))
    writer.add_image(name, grid_image, global_step=step, dataformats='CHW')


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    x_shape = x.size()
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return torch.reshape(x, x_shape[:-1] + (n, m // n))


def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

