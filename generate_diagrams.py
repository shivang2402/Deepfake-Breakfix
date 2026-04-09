"""Generate all architecture and pipeline diagrams for the report."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def draw_pipeline_overview(save_path):
    """Full project pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # title
    ax.text(8, 9.5, "Project Pipeline Overview", ha="center", fontsize=16, fontweight="bold")

    # boxes
    boxes = [
        # (x, y, w, h, label, color)
        (0.5, 7.5, 3, 1.2, "Dataset\n140k Real & Fake Faces\n(50k/50k train, 10k/10k test)", "#E3F2FD"),
        (4.5, 7.5, 3, 1.2, "Preprocessing\nResize 224x224\nNormalize (ImageNet)\nAugmentation (train)", "#E8F5E9"),
        (8.5, 7.5, 3, 1.2, "Baseline Training\nResNet-18\nEfficientNet-B0\nAdam + Cosine LR", "#FFF3E0"),
        (12.5, 7.5, 3, 1.2, "Baseline Eval\nAccuracy, Precision\nRecall, F1\nConfusion Matrix", "#F3E5F5"),

        # attack row
        (0.5, 5, 3.5, 1.2, "Image Degradations\nJPEG, Noise, Blur\nDownscale/Upscale", "#FFEBEE"),
        (4.5, 5, 3.5, 1.2, "Adversarial Attacks\nFGSM, PGD, C&W\nEOT-PGD", "#FFEBEE"),
        (9, 5, 3, 1.2, "Transferability\nAttack ResNet\nTest on EfficientNet\n& vice versa", "#FFEBEE"),
        (12.5, 5, 3, 1.2, "Frequency Analysis\nFFT Spectrum\nReal vs Fake", "#FFEBEE"),

        # defense row
        (0.5, 2.5, 3, 1.2, "Adversarial Training\nMix clean + PGD\n50/50 ratio", "#E8F5E9"),
        (4.5, 2.5, 3, 1.2, "AFSL Defense\nFeature Similarity\nLearning", "#E8F5E9"),
        (8.5, 2.5, 3, 1.2, "Ensemble Defense\nResNet + EfficientNet\nAvg Softmax", "#E8F5E9"),
        (12.5, 2.5, 3, 1.2, "Re-evaluate\nAll attacks on\nall defenses", "#F3E5F5"),

        # results
        (4, 0.5, 8, 1.2, "Results: Degradation Curves | Comparison Plots | Grad-CAM Heatmaps | Frequency Plots", "#FFFDE7"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=8, fontweight="bold")

    # row labels
    ax.text(0.1, 8.1, "DATA & MODEL", fontsize=9, fontweight="bold", color="#1565C0", rotation=90, va="center")
    ax.text(0.1, 5.6, "ATTACKS", fontsize=9, fontweight="bold", color="#C62828", rotation=90, va="center")
    ax.text(0.1, 3.1, "DEFENSES", fontsize=9, fontweight="bold", color="#2E7D32", rotation=90, va="center")
    ax.text(0.1, 1.1, "OUTPUT", fontsize=9, fontweight="bold", color="#F57F17", rotation=90, va="center")

    # arrows between rows
    for x in [2, 6, 10, 14]:
        ax.annotate("", xy=(x, 7.5), xytext=(x, 6.3),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    for x in [2, 6, 10]:
        ax.annotate("", xy=(x, 5), xytext=(x, 3.8),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    ax.annotate("", xy=(8, 2.5), xytext=(8, 1.8),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def draw_resnet_arch(save_path):
    """ResNet-18 architecture for deepfake detection."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(7, 4.5, "ResNet-18 for Deepfake Detection", ha="center", fontsize=14, fontweight="bold")

    layers = [
        (0.5, 1.5, 1.5, 2, "Input\n3x224x224", "#E3F2FD"),
        (2.5, 1.5, 1.5, 2, "Conv1\n7x7, 64\nBN + ReLU\nMaxPool", "#BBDEFB"),
        (4.5, 1.5, 1.5, 2, "Layer1\n2 blocks\n64 filters\nResidual", "#90CAF9"),
        (6.5, 1.5, 1.5, 2, "Layer2\n2 blocks\n128 filters\nResidual", "#64B5F6"),
        (8.5, 1.5, 1.5, 2, "Layer3\n2 blocks\n256 filters\nResidual", "#42A5F5"),
        (10.5, 1.5, 1.5, 2, "Layer4\n2 blocks\n512 filters\nResidual", "#2196F3"),
        (12.5, 1.5, 1.5, 2, "AvgPool\nFC 512→2\nFake/Real", "#FFF3E0"),
    ]

    for x, y, w, h, label, color in layers:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=7.5, fontweight="bold")

    # arrows
    for i in range(len(layers) - 1):
        x1 = layers[i][0] + layers[i][2]
        x2 = layers[i+1][0]
        y = layers[i][1] + layers[i][3] / 2
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.text(7, 0.8, "Pretrained on ImageNet → Fine-tuned on 140k Real & Fake Faces",
            ha="center", fontsize=10, style="italic", color="#666")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def draw_efficientnet_arch(save_path):
    """EfficientNet-B0 architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(7, 4.5, "EfficientNet-B0 for Deepfake Detection", ha="center", fontsize=14, fontweight="bold")

    layers = [
        (0.3, 1.5, 1.5, 2, "Input\n3x224x224", "#E8F5E9"),
        (2.1, 1.5, 1.5, 2, "Stem\n3x3 Conv\n32 filters\nBN+Swish", "#C8E6C9"),
        (3.9, 1.5, 1.5, 2, "MBConv1\n3x3, 16\nSE block\nx1", "#A5D6A7"),
        (5.7, 1.5, 1.5, 2, "MBConv6\n3x3, 24\nSE block\nx2", "#81C784"),
        (7.5, 1.5, 1.5, 2, "MBConv6\n5x5, 40-112\nSE block\nx6", "#66BB6A"),
        (9.3, 1.5, 1.5, 2, "MBConv6\n5x5, 192-320\nSE block\nx5", "#4CAF50"),
        (11.1, 1.5, 1.5, 2, "Head\n1x1 Conv\nAvgPool\nFC→2", "#FFF3E0"),
    ]

    for x, y, w, h, label, color in layers:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=7.5, fontweight="bold")

    for i in range(len(layers) - 1):
        x1 = layers[i][0] + layers[i][2]
        x2 = layers[i+1][0]
        y = layers[i][1] + layers[i][3] / 2
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.text(7, 0.8, "Compound scaling (depth, width, resolution) — Pretrained on ImageNet",
            ha="center", fontsize=10, style="italic", color="#666")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def draw_attack_pipeline(save_path):
    """Attack evaluation pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7, 7.5, "Attack Evaluation Pipeline", ha="center", fontsize=14, fontweight="bold")

    # input
    ax.add_patch(mpatches.FancyBboxPatch((5.5, 6.2), 3, 0.9, boxstyle="round,pad=0.1",
                                          facecolor="#E3F2FD", edgecolor="#333", linewidth=1.5))
    ax.text(7, 6.65, "Test Set (20k images)", ha="center", va="center", fontsize=9, fontweight="bold")

    # degradation branch
    deg_attacks = ["JPEG (Q=10-90)", "Noise (σ=0.01-0.2)", "Blur (k=3-11)", "Downscale (2x-8x)"]
    ax.add_patch(mpatches.FancyBboxPatch((0.5, 3.5), 4, 2.2, boxstyle="round,pad=0.1",
                                          facecolor="#FFEBEE", edgecolor="#C62828", linewidth=1.5))
    ax.text(2.5, 5.4, "Image Degradations", ha="center", fontsize=10, fontweight="bold", color="#C62828")
    for i, atk in enumerate(deg_attacks):
        ax.text(2.5, 5.0 - i * 0.4, f"• {atk}", ha="center", fontsize=8)

    # adversarial branch
    adv_attacks = ["FGSM (ε=0.001-0.08)", "PGD (ε=0.001-0.08)", "C&W L2 (c=0.1-10)", "EOT-PGD (ε=0.01-0.08)"]
    ax.add_patch(mpatches.FancyBboxPatch((5.5, 3.5), 4, 2.2, boxstyle="round,pad=0.1",
                                          facecolor="#FFEBEE", edgecolor="#C62828", linewidth=1.5))
    ax.text(7.5, 5.4, "Adversarial Attacks", ha="center", fontsize=10, fontweight="bold", color="#C62828")
    for i, atk in enumerate(adv_attacks):
        ax.text(7.5, 5.0 - i * 0.4, f"• {atk}", ha="center", fontsize=8)

    # transferability
    ax.add_patch(mpatches.FancyBboxPatch((10.5, 3.5), 3, 2.2, boxstyle="round,pad=0.1",
                                          facecolor="#FFEBEE", edgecolor="#C62828", linewidth=1.5))
    ax.text(12, 5.4, "Transferability", ha="center", fontsize=10, fontweight="bold", color="#C62828")
    ax.text(12, 4.8, "• ResNet→EfficientNet", ha="center", fontsize=8)
    ax.text(12, 4.4, "• EfficientNet→ResNet", ha="center", fontsize=8)
    ax.text(12, 4.0, "• PGD + FGSM", ha="center", fontsize=8)

    # arrows from test set
    ax.annotate("", xy=(2.5, 5.7), xytext=(5.5, 6.2),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    ax.annotate("", xy=(7, 5.7), xytext=(7, 6.2),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    ax.annotate("", xy=(12, 5.7), xytext=(8.5, 6.2),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # model box
    ax.add_patch(mpatches.FancyBboxPatch((3, 1.5), 4, 1.2, boxstyle="round,pad=0.1",
                                          facecolor="#FFF3E0", edgecolor="#333", linewidth=1.5))
    ax.text(5, 2.1, "Model Under Test\nBaseline | Robust (AT) | AFSL | Ensemble",
            ha="center", va="center", fontsize=8, fontweight="bold")

    # metrics box
    ax.add_patch(mpatches.FancyBboxPatch((8, 1.5), 4.5, 1.2, boxstyle="round,pad=0.1",
                                          facecolor="#E8F5E9", edgecolor="#333", linewidth=1.5))
    ax.text(10.25, 2.1, "Metrics\nAccuracy, Precision, Recall, F1\nConfusion Matrix, Degradation Curves",
            ha="center", va="center", fontsize=8, fontweight="bold")

    # arrows
    ax.annotate("", xy=(5, 2.7), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
    ax.annotate("", xy=(8, 2.1), xytext=(7, 2.1),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    # output
    ax.add_patch(mpatches.FancyBboxPatch((3, 0.2), 9.5, 0.8, boxstyle="round,pad=0.1",
                                          facecolor="#FFFDE7", edgecolor="#333", linewidth=1.5))
    ax.text(7.75, 0.6, "Output: JSON results + Degradation plots + Comparison charts + Grad-CAM heatmaps",
            ha="center", va="center", fontsize=8, fontweight="bold")

    ax.annotate("", xy=(7.75, 1.0), xytext=(7.75, 1.5),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def draw_defense_comparison(save_path):
    """Defense strategies comparison diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7, 5.5, "Defense Strategies", ha="center", fontsize=14, fontweight="bold")

    defenses = [
        (0.5, 1, 3, 3.5, "Adversarial Training\n(AT)", "#E8F5E9",
         ["Mix 50% clean + 50% PGD", "during training", "",
          "Clean acc: ~98.9%", "PGD ε=0.01: ~46.6%", "",
          "Simple but effective", "Some clean acc drop"]),
        (4, 1, 3, 3.5, "AFSL\n(Feature Similarity)", "#E8F5E9",
         ["Train features to be", "similar for clean &", "adversarial inputs",
          "Expected clean: ~96-97%", "Expected PGD: ~70%+", "",
          "Stronger than AT", "More compute needed"]),
        (7.5, 1, 3, 3.5, "Ensemble\n(ResNet + EfficientNet)", "#E8F5E9",
         ["Average softmax from", "two architectures", "",
          "Harder to fool both", "at same time", "",
          "No retraining needed", "2x inference cost"]),
        (11, 1, 2.5, 3.5, "No Defense\n(Baseline)", "#FFEBEE",
         ["Clean acc: 99.59%", "FGSM ε=0.005: 53%", "PGD ε=0.01: 8.5%", "",
          "Completely broken", "by adversarial", "attacks"]),
    ]

    for x, y, w, h, title, color, lines in defenses:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.3, title, ha="center", va="center", fontsize=9, fontweight="bold")
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h - 0.7 - i * 0.35, line, ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    os.makedirs("results/diagrams", exist_ok=True)

    draw_pipeline_overview("results/diagrams/pipeline_overview.png")
    draw_resnet_arch("results/diagrams/resnet18_architecture.png")
    draw_efficientnet_arch("results/diagrams/efficientnet_b0_architecture.png")
    draw_attack_pipeline("results/diagrams/attack_pipeline.png")
    draw_defense_comparison("results/diagrams/defense_comparison.png")

    print("\nAll diagrams generated in results/diagrams/")


if __name__ == "__main__":
    main()
