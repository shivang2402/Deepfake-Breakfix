"""Generate architecture and pipeline diagrams."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def make_box(ax, x, y, w, h, text, color="#E3F2FD", fontsize=9, border="#333"):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=color, edgecolor=border, linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", linespacing=1.4)


def h_arrow(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))


def v_arrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))


def l_arrow_down_right(ax, x1, y1, x2, y2):
    """L-shaped: go down then right."""
    ax.plot([x1, x1], [y1, y2], color="#444", lw=1.5)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y2),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))


def l_arrow_down_left(ax, x1, y1, x2, y2):
    """L-shaped: go down then left."""
    ax.plot([x1, x1], [y1, y2], color="#444", lw=1.5)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y2),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))


def row_separator(ax, y, x1, x2):
    """Draw a wide horizontal arrow spanning the row."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#999", lw=2,
                                linestyle="--"))


def draw_pipeline_overview(path):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(8, 11.5, "Project Pipeline", ha="center", fontsize=16, fontweight="bold")

    # 4 columns: x = 1.5, 5.0, 8.5, 12.0  width = 3.0
    cx = [1.5, 5.0, 8.5, 12.0]
    bw = 3.0

    # --- ROW 1: DATA & MODEL ---
    r1y = 9.5
    rh = 1.3
    ax.text(0.15, r1y + rh / 2, "DATA &\nMODEL", fontsize=9, fontweight="bold",
            color="#1565C0", va="center", ha="center")

    make_box(ax, cx[0], r1y, bw, rh, "Dataset\n140k Real & Fake\n50k/50k, 10k/10k", "#E3F2FD", 9, "#1565C0")
    make_box(ax, cx[1], r1y, bw, rh, "Preprocessing\nResize 224x224\nNormalize, Augment", "#E3F2FD", 9, "#1565C0")
    make_box(ax, cx[2], r1y, bw, rh, "Train Models\nResNet-18\nEfficientNet-B0", "#E3F2FD", 9, "#1565C0")
    make_box(ax, cx[3], r1y, bw, rh, "Evaluate\nAcc, Precision\nRecall, F1", "#E3F2FD", 9, "#1565C0")

    for i in range(3):
        h_arrow(ax, cx[i] + bw, cx[i + 1], r1y + rh / 2)

    # row connector: wide dashed arrow spanning all 4 columns
    conn_y = r1y - 0.5
    ax.plot([1.5, 15], [conn_y, conn_y], color="#999", lw=1, linestyle="--")
    ax.plot([8, 8], [conn_y, conn_y - 0.3], color="#444", lw=1.5)
    ax.annotate("", xy=(8, conn_y - 0.6), xytext=(8, conn_y - 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))

    # --- ROW 2: ATTACKS ---
    r2y = 7.0
    ax.text(0.15, r2y + rh / 2, "ATTACKS", fontsize=9, fontweight="bold",
            color="#C62828", va="center", ha="center")

    make_box(ax, cx[0], r2y, bw, rh, "Degradations\nJPEG | Noise\nBlur | Downscale", "#FFEBEE", 9, "#C62828")
    make_box(ax, cx[1], r2y, bw, rh, "Adversarial\nFGSM | PGD\nC&W | EOT-PGD", "#FFEBEE", 9, "#C62828")
    make_box(ax, cx[2], r2y, bw, rh, "Transferability\nAttack model A\nTest on model B", "#FFEBEE", 9, "#C62828")
    make_box(ax, cx[3], r2y, bw, rh, "Frequency\nFFT spectrum\nReal vs Fake", "#FFEBEE", 9, "#C62828")

    # row connector
    conn_y2 = r2y - 0.5
    ax.plot([1.5, 15], [conn_y2, conn_y2], color="#999", lw=1, linestyle="--")
    ax.plot([8, 8], [conn_y2, conn_y2 - 0.3], color="#444", lw=1.5)
    ax.annotate("", xy=(8, conn_y2 - 0.6), xytext=(8, conn_y2 - 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))

    # --- ROW 3: DEFENSES ---
    r3y = 4.5
    ax.text(0.15, r3y + rh / 2, "DEFENSES", fontsize=9, fontweight="bold",
            color="#2E7D32", va="center", ha="center")

    make_box(ax, cx[0], r3y, bw, rh, "Adversarial Training\n50% clean + 50% PGD\nduring training", "#E8F5E9", 9, "#2E7D32")
    make_box(ax, cx[1], r3y, bw, rh, "AFSL Defense\nFeature similarity\nclean = adversarial", "#E8F5E9", 9, "#2E7D32")
    make_box(ax, cx[2], r3y, bw, rh, "Ensemble\nResNet + EfficientNet\navg predictions", "#E8F5E9", 9, "#2E7D32")
    make_box(ax, cx[3], r3y, bw, rh, "Re-evaluate\nAll attacks on\nall defended models", "#F3E5F5", 9, "#7B1FA2")

    for i in range(3):
        h_arrow(ax, cx[i] + bw, cx[i + 1], r3y + rh / 2)

    # row connector
    conn_y3 = r3y - 0.5
    ax.plot([1.5, 15], [conn_y3, conn_y3], color="#999", lw=1, linestyle="--")
    ax.plot([8, 8], [conn_y3, conn_y3 - 0.3], color="#444", lw=1.5)
    ax.annotate("", xy=(8, conn_y3 - 0.6), xytext=(8, conn_y3 - 0.3),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))

    # --- ROW 4: OUTPUT ---
    r4y = 2.0
    ax.text(0.15, r4y + rh / 2, "OUTPUT", fontsize=9, fontweight="bold",
            color="#E65100", va="center", ha="center")

    make_box(ax, cx[0], r4y, bw, rh, "Degradation Curves\nAccuracy vs\nattack intensity", "#FFFDE7", 9, "#F57F17")
    make_box(ax, cx[1], r4y, bw, rh, "Comparison Charts\nBaseline vs Robust\nvs AFSL vs Ensemble", "#FFFDE7", 9, "#F57F17")
    make_box(ax, cx[2], r4y, bw, rh, "Grad-CAM\nHeatmaps before\nand after attacks", "#FFFDE7", 9, "#F57F17")
    make_box(ax, cx[3], r4y, bw, rh, "Frequency Plots\nSpectral difference\nreal vs fake", "#FFFDE7", 9, "#F57F17")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def draw_resnet_arch(path):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7.5, 3.6, "ResNet-18 Architecture", ha="center", fontsize=14, fontweight="bold")

    bw, bh, by, gap = 1.7, 2.2, 0.7, 0.3
    colors = ["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5", "#FFF3E0"]
    texts = [
        "Input\n3 x 224 x 224",
        "Conv1\n7x7, 64\nMaxPool",
        "Layer 1\n2 blocks\n64 ch, 56x56",
        "Layer 2\n2 blocks\n128 ch, 28x28",
        "Layer 3\n2 blocks\n256 ch, 14x14",
        "Layer 4\n2 blocks\n512 ch, 7x7",
        "Head\nAvgPool\nFC 512 -> 2",
    ]

    for i in range(7):
        x = 0.5 + i * (bw + gap)
        make_box(ax, x, by, bw, bh, texts[i], colors[i], 8.5)
        if i == 5:
            ax.texts[-1].set_color("white")
        if i > 0:
            h_arrow(ax, 0.5 + (i - 1) * (bw + gap) + bw, x, by + bh / 2)

    x6 = 0.5 + 5 * (bw + gap)
    ax.annotate("Grad-CAM\ntargets here",
                xy=(x6 + bw / 2, by + bh),
                xytext=(x6 + bw + 0.8, by + bh + 0.6),
                fontsize=8, ha="center", color="#C62828", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#C62828", lw=1.5))

    ax.text(7.5, 0.25, "ImageNet pretrained  ->  FC layer swapped  ->  fine-tuned on 140k faces",
            ha="center", fontsize=9, color="#666")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def draw_efficientnet_arch(path):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7.5, 3.6, "EfficientNet-B0 Architecture", ha="center", fontsize=14, fontweight="bold")

    bw, bh, by, gap = 1.7, 2.2, 0.7, 0.3
    colors = ["#E8F5E9", "#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A", "#4CAF50", "#FFF3E0"]
    texts = [
        "Input\n3 x 224 x 224",
        "Stem\n3x3 Conv, 32\nBN + Swish",
        "MBConv1\n3x3, 16 ch\nSE, x1",
        "MBConv6\n3x3, 24 ch\nSE, x2",
        "MBConv6\n5x5, 40-112\nSE, x6",
        "MBConv6\n5x5, 192-320\nSE, x5",
        "Head\n1x1 Conv\nFC -> 2",
    ]

    for i in range(7):
        x = 0.5 + i * (bw + gap)
        make_box(ax, x, by, bw, bh, texts[i], colors[i], 8.5)
        if i == 5:
            ax.texts[-1].set_color("white")
        if i > 0:
            h_arrow(ax, 0.5 + (i - 1) * (bw + gap) + bw, x, by + bh / 2)

    ax.text(7.5, 0.25, "Compound scaling (depth x width x resolution)  ->  classifier swapped to 2 classes",
            ha="center", fontsize=9, color="#666")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def draw_attack_pipeline(path):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7.5, 9.5, "Attack Evaluation Pipeline", ha="center", fontsize=15, fontweight="bold")

    # step 1: test set (centered)
    s1x, s1w, s1y, s1h = 4.5, 6, 8.0, 1
    make_box(ax, s1x, s1y, s1w, s1h, "Step 1: Test Set\n20,000 images (10k fake + 10k real)", "#E3F2FD", 10, "#1565C0")

    # step 2: three attack boxes
    s2y = 5.5
    s2h = 1.8
    s2w = 4.2
    # box centers (x midpoints)
    b1_cx = 0.5 + s2w / 2     # 2.6
    b2_cx = 5.4 + s2w / 2     # 7.5
    b3_cx = 10.3 + s2w / 2    # 12.4

    # T-junction: vertical from step1 center, horizontal bar, verticals into boxes
    junction_y = 7.6
    ax.plot([7.5, 7.5], [s1y, junction_y], color="#444", lw=1.5)  # down from step1
    ax.plot([b1_cx, b3_cx], [junction_y, junction_y], color="#444", lw=1.5)  # horizontal bar
    # vertical lines down to box tops (no arrowhead artifacts — use plain lines + triangle)
    for bx in [b1_cx, b2_cx, b3_cx]:
        ax.plot([bx, bx], [junction_y, s2y + s2h + 0.05], color="#444", lw=1.5)
        # small triangle arrowhead
        ax.plot(bx, s2y + s2h + 0.05, marker="v", color="#444", markersize=6)

    make_box(ax, 0.5, s2y, s2w, s2h,
             "Step 2a: Degradations\n\nJPEG (Q = 10 to 90)\nGaussian noise (s = 0.01 to 0.2)\nGaussian blur (k = 3 to 11)\nDownscale (2x, 4x, 8x)",
             "#FFEBEE", 8.5, "#C62828")

    make_box(ax, 5.4, s2y, s2w, s2h,
             "Step 2b: Adversarial\n\nFGSM (e = 0.001 to 0.08)\nPGD 10-step (e = 0.001 to 0.08)\nC&W L2 (c = 0.1 to 10)\nEOT-PGD (e = 0.01 to 0.08)",
             "#FFEBEE", 8.5, "#C62828")

    make_box(ax, 10.3, s2y, s2w, s2h,
             "Step 2c: Transferability\n\nAttack ResNet-18,\ntest on EfficientNet-B0\n\nAttack EfficientNet-B0,\ntest on ResNet-18",
             "#FFEBEE", 8.5, "#C62828")

    # merge: lines from box bottom centers, horizontal bar, single arrow down
    merge_y = 5.0
    for bx in [b1_cx, b2_cx, b3_cx]:
        ax.plot([bx, bx], [s2y, merge_y], color="#444", lw=1.5)
    ax.plot([b1_cx, b3_cx], [merge_y, merge_y], color="#444", lw=1.5)
    ax.plot([7.5, 7.5], [merge_y, 4.3], color="#444", lw=1.5)
    ax.plot(7.5, 4.3, marker="v", color="#444", markersize=6)

    # step 3: model
    make_box(ax, 2.5, 3.0, 10, 1.2,
             "Step 3: Run Through Model\nBaseline (ResNet-18)  |  Adversarial Training  |  AFSL  |  Ensemble",
             "#FFF3E0", 10, "#E65100")

    # T-junction from step3 to step4a and step4b
    s4_jy = 2.5
    ax.plot([7.5, 7.5], [3.0, s4_jy], color="#444", lw=1.5)
    ax.plot([4.25, 10.75], [s4_jy, s4_jy], color="#444", lw=1.5)
    # arrows down to 4a and 4b
    ax.plot([4.25, 4.25], [s4_jy, 2.05], color="#444", lw=1.5)
    ax.plot(4.25, 2.05, marker="v", color="#444", markersize=6)
    ax.plot([10.75, 10.75], [s4_jy, 2.05], color="#444", lw=1.5)
    ax.plot(10.75, 2.05, marker="v", color="#444", markersize=6)

    # step 4: two boxes side by side
    make_box(ax, 1.5, 0.7, 5.5, 1.3,
             "Step 4a: Metrics\nAccuracy, Precision, Recall, F1\nConfusion matrix per setting",
             "#E8F5E9", 9, "#2E7D32")
    make_box(ax, 8, 0.7, 5.5, 1.3,
             "Step 4b: Visualize\nDegradation curves, Comparison plots\nGrad-CAM heatmaps",
             "#E8F5E9", 9, "#2E7D32")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def draw_defense_comparison(path):
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(8, 6.5, "Defense Strategies Comparison", ha="center", fontsize=15, fontweight="bold")

    cw = 3.5
    ch = 5.2
    cy = 0.8
    gap = 0.4

    cols = [
        ("No Defense\n(Baseline)", "#FFCDD2", "#C62828",
         ["Clean: 99.59%", "", "FGSM e=0.005: 53%", "PGD e=0.01: 8.5%", "PGD e=0.02: 0.8%",
          "", "Completely broken"]),
        ("Adversarial\nTraining", "#C8E6C9", "#2E7D32",
         ["Clean: 98.92% (-0.7%)", "", "PGD e=0.01: 46.6%", "FGSM e=0.08: 39.8%",
          "", "Big improvement", "Simple to implement"]),
        ("AFSL\n(Feature Similarity)", "#C8E6C9", "#2E7D32",
         ["Expected clean: ~97%", "", "Expected PGD: ~70%+", "(Goswami et al. 2024)",
          "", "Strongest defense", "2x compute per batch"]),
        ("Ensemble\n(ResNet + EffNet)", "#C8E6C9", "#2E7D32",
         ["Expected clean: ~99%", "", "Harder to fool both", "architectures at once",
          "", "No retraining needed", "2x inference cost"]),
    ]

    for i, (title, bg, border, lines) in enumerate(cols):
        x = 0.4 + i * (cw + gap)

        # main box
        rect = patches.FancyBboxPatch(
            (x, cy), cw, ch, boxstyle="round,pad=0.15",
            facecolor=bg, edgecolor=border, linewidth=2
        )
        ax.add_patch(rect)

        # title area
        ax.text(x + cw / 2, cy + ch - 0.55, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=border)

        # divider line
        div_y = cy + ch - 1.1
        ax.plot([x + 0.2, x + cw - 0.2], [div_y, div_y], color=border, lw=1, alpha=0.5)

        # content lines
        for j, line in enumerate(lines):
            ax.text(x + 0.3, div_y - 0.35 - j * 0.45, line, fontsize=9)

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def draw_adversarial_training_flow(path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(7, 4.6, "Adversarial Training Flow", ha="center", fontsize=14, fontweight="bold")

    # input batch
    make_box(ax, 0.3, 1.7, 2, 1.3, "Training Batch\n(32 images)", "#E3F2FD", 10, "#1565C0")

    # split: horizontal line right, then T-junction up and down
    split_x = 3.0
    ax.plot([2.3, split_x], [2.35, 2.35], color="#444", lw=1.5)
    ax.plot([split_x, split_x], [1.35, 3.35], color="#444", lw=1.5)

    # top path: clean
    ax.plot([split_x, 3.8], [3.35, 3.35], color="#444", lw=1.5)
    ax.annotate("", xy=(3.8, 3.35), xytext=(3.5, 3.35),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))
    make_box(ax, 3.8, 2.7, 2.2, 1.2, "50% Clean\n(unchanged)", "#E8F5E9", 10, "#2E7D32")

    # bottom path: adversarial
    ax.plot([split_x, 3.8], [1.35, 1.35], color="#444", lw=1.5)
    ax.annotate("", xy=(3.8, 1.35), xytext=(3.5, 1.35),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))
    make_box(ax, 3.8, 0.7, 2.2, 1.2, "50% PGD\nAttacked\n(e = 0.02)", "#FFEBEE", 10, "#C62828")

    # merge: both paths go right to a junction, then into combined
    merge_x = 6.8
    ax.plot([6.0, merge_x], [3.3, 3.3], color="#444", lw=1.5)
    ax.plot([6.0, merge_x], [1.3, 1.3], color="#444", lw=1.5)
    ax.plot([merge_x, merge_x], [1.3, 3.3], color="#444", lw=1.5)
    ax.plot([merge_x, 7.5], [2.3, 2.3], color="#444", lw=1.5)
    ax.annotate("", xy=(7.5, 2.3), xytext=(7.2, 2.3),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))

    make_box(ax, 7.5, 1.7, 2, 1.3, "Combined\nBatch\n(mixed)", "#FFF3E0", 10, "#E65100")

    # train
    h_arrow(ax, 9.5, 10.2, 2.35)
    make_box(ax, 10.2, 1.7, 1.8, 1.3, "Train\n(backprop)", "#E3F2FD", 10, "#1565C0")

    # output
    h_arrow(ax, 12.0, 12.5, 2.35)
    make_box(ax, 12.5, 1.7, 1.3, 1.3, "Robust\nModel", "#C8E6C9", 10, "#2E7D32")

    # labels
    ax.text(3.0, 3.7, "split", fontsize=8, ha="center", color="#666")
    ax.text(6.8, 3.6, "merge", fontsize=8, ha="center", color="#666")

    ax.text(7, 0.15, "Model sees both clean and attacked images during training -> learns harder-to-fool features",
            ha="center", fontsize=9, color="#555")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs("results/diagrams", exist_ok=True)
    draw_pipeline_overview("results/diagrams/pipeline_overview.png")
    draw_resnet_arch("results/diagrams/resnet18_architecture.png")
    draw_efficientnet_arch("results/diagrams/efficientnet_b0_architecture.png")
    draw_attack_pipeline("results/diagrams/attack_pipeline.png")
    draw_defense_comparison("results/diagrams/defense_comparison.png")
    draw_adversarial_training_flow("results/diagrams/adversarial_training_flow.png")
    print("\nAll diagrams generated in results/diagrams/")


if __name__ == "__main__":
    main()
