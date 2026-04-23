# Shivang Patel
# CS 5330, Final Project: Deepfake Breakfix
# Generate all degradation and comparison plots from JSON results

# standard library
import os
import sys

# local
from src.visualization.plots import plot_all_degradation_curves, plot_all_comparisons


# main function
def main(argv):
    os.makedirs("results/plots", exist_ok=True)

    baseline_path = "results/tables/attack_results.json"
    robust_path = "results/tables/attack_results_robust.json"

    # plot baseline degradation curves
    if os.path.exists(baseline_path):
        print("Plotting baseline degradation curves...")
        plot_all_degradation_curves(baseline_path, "results/plots")
    else:
        print("Missing %s" % baseline_path)

    # plot comparisons if robust results exist
    if os.path.exists(baseline_path) and os.path.exists(robust_path):
        print("Plotting baseline vs robust comparisons...")
        plot_all_comparisons(baseline_path, robust_path, "results/plots")
    else:
        print("Skipping comparison plots, need both baseline and robust results")

    print("Done!")
    return


if __name__ == "__main__":
    main(sys.argv)
