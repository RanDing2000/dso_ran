import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the data
data = {
    "Occlusion Level": ["[0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", 
                        "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)"] * 2,
    "Successful Rate": [
        # TARGO-Net (Noise 0.02)
        0.8697863682604272, 0.8109756097560976, 0.8372093023255814, 
        0.8064516129032258, 0.8185483870967742, 0.8062563067608476, 
        0.8076923076923077, 0.796576032225579, 0.7977755308392316,
        # TARGO-Net (Noise 0.05)
        0.8520, 0.7970, 0.8240, 0.7960, 0.8290, 
        0.8110, 0.7950, 0.7940, 0.7930
    ],
    "Model": ["TARGO-Net (Noise 0.02)"]*9 + ["TARGO-Net (Noise 0.05)"]*9
}

df = pd.DataFrame(data)

# Map occlusion bins into new ranges
def map_to_class(level):
    if level in ["[0,0.1)", "[0.1,0.2)", "[0.2,0.3)"]:
        return "[0,0.3)"
    elif level in ["[0.3,0.4)", "[0.4,0.5)", "[0.5,0.6)"]:
        return "[0.3,0.6)"
    else:
        return "[0.6,0.9)"

df["Class"] = df["Occlusion Level"].apply(map_to_class)

# Group by range and model, take average
df_grouped = df.groupby(["Class", "Model"], as_index=False)["Successful Rate"].mean()

# 保证区间顺序
class_order = ["[0,0.3)", "[0.3,0.6)", "[0.6,0.9)"]
df_grouped["Class"] = pd.Categorical(df_grouped["Class"], 
                                     categories=class_order, 
                                     ordered=True)

# Set the plot size
plt.figure(figsize=(10, 7))

# Define markers/colors/line styles
markers = {
    "TARGO-Net (Noise 0.02)": "s", 
    "TARGO-Net (Noise 0.05)": "D"
}
palette = {
    "TARGO-Net (Noise 0.02)": "#4865A9",  
    "TARGO-Net (Noise 0.05)": "#B2B2B2"   
}
dashes = {
    "TARGO-Net (Noise 0.02)": (2, 2),     
    "TARGO-Net (Noise 0.05)": (1, 1)      
}

# 如果没有 Times New Roman，则 fallback 到 serif
plt.rcParams["font.family"] = ["Times New Roman", "serif"]

# Plot
sns.lineplot(
    data=df_grouped, 
    x="Class", 
    y="Successful Rate", 
    hue="Model", 
    style="Model", 
    markers=markers, 
    palette=palette, 
    dashes=dashes
)

# Labels
plt.xlabel("Occlusion Difficulty", fontsize=32)
plt.ylabel("Successful Rate", fontsize=32)

plt.xticks(fontsize=28)
plt.yticks(fontsize=28)

plt.legend(fontsize=22, frameon=False)
plt.grid(False)

plt.savefig("noise_targo_classes.pdf", bbox_inches="tight")
# plt.show()
