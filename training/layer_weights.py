import torch
import torch.nn.functional as F

# ======== PATHS ========
LAYER_WEIGHTS_PATH = "/sd1/jhansi/interns/somesh/whisper-asl/whistress/training/training_results/layer_weights.pt"

# Must match model.py
LAYER_TEMPERATURE = 0.05

# =======================

def entropy(p):
    return -(p * torch.log(p + 1e-9)).sum()


def main():
    print(f"Loading layer weights from:\n{LAYER_WEIGHTS_PATH}\n")

    weights_dict = torch.load(LAYER_WEIGHTS_PATH, map_location="cpu")

    encoder_raw = weights_dict["encoder_weights"]
    decoder_raw = weights_dict["decoder_weights"]

    print("========== RAW WEIGHTS ==========")
    print("Encoder raw:", encoder_raw)
    print("Decoder raw:", decoder_raw)
    print()

    # Softmax normalization with temperature
    encoder_soft = F.softmax(encoder_raw / LAYER_TEMPERATURE, dim=0)
    decoder_soft = F.softmax(decoder_raw / LAYER_TEMPERATURE, dim=0)

    print("========== NORMALIZED (Softmax) ==========\n")

    print("---- Encoder Layer Weights ----")
    for i, w in enumerate(encoder_soft):
        print(f"Encoder Layer {i:2d}: {w.item():.6f}")

    print("\n---- Decoder Layer Weights ----")
    for i, w in enumerate(decoder_soft):
        print(f"Decoder Layer {i:2d}: {w.item():.6f}")

    print("\n========== DISTRIBUTION STATS ==========")
    print(f"Encoder entropy: {entropy(encoder_soft).item():.6f}")
    print(f"Decoder entropy: {entropy(decoder_soft).item():.6f}")


if __name__ == "__main__":
    main()