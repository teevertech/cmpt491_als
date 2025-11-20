import torch

PATCH_SIZE = 16            # must match ElasticAST patch size
MAX_WIDTH = 1024           # ElasticAST maximum supported time dimension


def pad_mels(batch):
    """
    Pads mel spectrograms so:
      1. All samples in the batch have the same time length (T)
      2. T is capped at MAX_WIDTH (ElasticAST cannot handle very large values)
      3. T is divisible by PATCH_SIZE (e.g., 16)
    """

    # Unpack spectrograms and labels
    mels = [item["input_values"] for item in batch]  # each is (T, F)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    # ------------------------------------------------------------------
    # STEP 1 — Compute batch max length, capped by ElasticAST MAX_WIDTH
    # ------------------------------------------------------------------

    raw_max_len = max(mel.shape[0] for mel in mels)
    max_len = min(raw_max_len, MAX_WIDTH)    # cap to 1024

    # Align downward to nearest multiple of patch size
    # (ElasticAST requires patch-compatible widths)
    max_len = (max_len // PATCH_SIZE) * PATCH_SIZE
    if max_len < PATCH_SIZE:
        max_len = PATCH_SIZE

    # ------------------------------------------------------------------
    # STEP 2 — Crop or pad each sample to max_len
    # ------------------------------------------------------------------
    padded = []

    for mel in mels:
        T, F = mel.shape

        # --- CROP if too long ---
        if T > max_len:
            mel = mel[:max_len, :]

        # --- PAD if too short ---
        elif T < max_len:
            pad_len = max_len - T
            pad_tensor = torch.zeros(pad_len, F)
            mel = torch.cat([mel, pad_tensor], dim=0)

        padded.append(mel)

    # Stack into final batch tensor: (B, T, F)
    padded = torch.stack(padded, dim=0)

    return {
        "input_values": padded,
        "labels": labels,
    }