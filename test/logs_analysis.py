import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogAnalyzer:
    def __init__(self, log_dir="logs", smoothing_alpha=0.85):
        self.log_dir = log_dir
        self.smoothing_alpha = smoothing_alpha
        self.configs = {
            "M1_Baseline_DINO": {"embedding": "DINOv2", "param_shift": "baseline"},
            "M2_Baseline_CLIP_ViT": {"embedding": "CLIP_ViT", "param_shift": "baseline"},
            "M3_Baseline_CLIP_RN": {"embedding": "CLIP_RN", "param_shift": "baseline"},
            "M4_ParamShift_DINO": {"embedding": "DINOv2", "param_shift": "div_search"},
            "M5_ParamShift_CLIP_ViT": {"embedding": "CLIP_ViT", "param_shift": "div_search"}
        }
        self.data = {}

    def load_logs(self):
        for name in self.configs:
            path = os.path.join(self.log_dir, name, "run_log.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["member"] = name
                self.data[name] = df
            else:
                print(f"Warning: Log file not found for {name}. Skipping.")

    def _exponential_smooth(self, series, alpha):
        smoothed = [series.iloc[0]]
        for val in series.iloc[1:]:
            smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
        return smoothed

    def compute_decoupling_metrics(self):
        if len(self.data) < 4:
            raise ValueError("Insufficient logs loaded. Need at least M1, M2, M4, M5 for decoupling.")

        # Aggregate final episode metrics (last 20% of episodes for stability)
        final_window = 0.2
        summary_stats = {}
        for name, df in self.data.items():
            cutoff = int(len(df) * (1 - final_window))
            tail = df.iloc[cutoff:]
            summary_stats[name] = {
                "mean_total_reward": tail["total_reward"].mean(),
                "mean_steps": tail["steps_taken"].mean(),
                "mean_length": tail["summary_length"].mean(),
                "reward_std": tail["total_reward"].std(),
                "smoothed_reward": self._exponential_smooth(df["total_reward"], self.smoothing_alpha)[-1]
            }

        stats_df = pd.DataFrame(summary_stats).T
        stats_df.index.name = "experiment"
        
        # Decoupling calculations
        embedding_effect_vit = stats_df.loc["M2_Baseline_CLIP_ViT", "smoothed_reward"] - stats_df.loc["M1_Baseline_DINO", "smoothed_reward"]
        param_effect_dino = stats_df.loc["M4_ParamShift_DINO", "smoothed_reward"] - stats_df.loc["M1_Baseline_DINO", "smoothed_reward"]
        interaction_effect = (stats_df.loc["M5_ParamShift_CLIP_ViT", "smoothed_reward"] - stats_df.loc["M2_Baseline_CLIP_ViT", "smoothed_reward"]) - param_effect_dino

        print("\n--- DECOUPLING ANALYSIS REPORT ---")
        print("1. Embedding Effect (CLIP_ViT vs DINO, fixed params): {:.4f}".format(embedding_effect_vit))
        print("2. Parameter Effect (div_search vs baseline, fixed DINO): {:.4f}".format(param_effect_dino))
        print("3. Interaction Effect: {:.4f}".format(interaction_effect))
        print("   -> Near 0: Parameters and embeddings act independently.")
        print("   -> Significant deviation: Reward shaping sensitivity is embedding-specific.")
        
        return stats_df

    def plot_convergence(self, output_path="convergence_plots.png"):
        plt.figure(figsize=(12, 6))
        for name, df in self.data.items():
            smoothed = self._exponential_smooth(df["total_reward"], self.smoothing_alpha)
            plt.plot(smoothed, label=name, linewidth=2)
        plt.title("Exponentially Smoothed Reward Convergence")
        plt.xlabel("Episode Index")
        plt.ylabel("Smoothed Total Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Convergence plot saved to {output_path}")

if __name__ == "__main__":
    analyzer = LogAnalyzer()
    analyzer.load_logs()
    stats = analyzer.compute_decoupling_metrics()
    print("\nFinal 20% Episode Averages:")
    print(stats[["smoothed_reward", "reward_std", "mean_steps", "mean_length"]].round(4))
    analyzer.plot_convergence()