import torch

PATCH_SIZE = 16  # must match wrapper


def pad_mels(batch):
    """
    Pads mel spectrograms so:
      1. All samples in the batch have same time length
      2. Time dim (T) is divisible by PATCH_SIZE (e.g., 16)
    """

    mels = [item["input_values"] for item in batch]  # (T, F)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    # Step 1: pad each sample to max T in the batch
    max_len = max(mel.shape[0] for mel in mels)

    # After padding to max_len, enforce T divisible by patch size
    # Example: if max_len = 1915, next divisible by 16 is 1920
    if max_len % PATCH_SIZE != 0:
        max_len = ((max_len + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

    padded = []
    for mel in mels:
        T, F = mel.shape
        pad_len = max_len - T

        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, F)
            mel = torch.cat([mel, pad_tensor], dim=0)

        padded.append(mel)

    # Stack into a batch
    padded = torch.stack(padded, dim=0)  # (B, T, F)

    return {
        "input_values": padded,
        "labels": labels,
    }