import torch

def pad_mels(batch):
    """
    Batch is a list of samples from SANDDataset:
    { 'input_values': (T,F), 'labels': int, ... }
    We pad input_values along T to the max length in the batch.
    """
    # unpack
    mels = [item["input_values"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    # find max length in batch
    max_len = max(mel.shape[0] for mel in mels)

    # pad all sequences to max_len
    padded = []
    for mel in mels:
        T, F = mel.shape
        pad_len = max_len - T
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, F)
            mel = torch.cat([mel, pad_tensor], dim=0)
        padded.append(mel)

    padded = torch.stack(padded, dim=0)  # (B, T, F)

    return {
        "input_values": padded,
        "labels": labels,
    }