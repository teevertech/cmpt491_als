import torch

PATCH_SIZE = 16
FIXED_T = 1024    # <<< IMPORTANT: must match training length!


def pad_mels(batch):
    """
    Pads all mel spectrograms to FIXED_T (1024) frames,
    ensuring the time dimension is ALWAYS identical across
    training, validation, and testing.
    
    This prevents ElasticAST positional embedding mismatches.
    """

    mels = [item["input_values"] for item in batch]  # (T, F)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    padded = []

    for mel in mels:
        T, F = mel.shape

        # --- CROP if too long ---
        if T > FIXED_T:
            mel = mel[:FIXED_T, :]

        # --- PAD if too short ---
        elif T < FIXED_T:
            pad_len = FIXED_T - T
            pad_tensor = torch.zeros(pad_len, F)
            mel = torch.cat([mel, pad_tensor], dim=0)

        padded.append(mel)

    # Stack into (B, FIXED_T, F)
    padded = torch.stack(padded, dim=0)

    return {
        "input_values": padded,
        "labels": labels,
    }