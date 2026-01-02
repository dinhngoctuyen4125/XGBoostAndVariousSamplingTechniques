import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = './creditcard.csv'
OUTPUT_DIR = "visualization"

def feature_distribution(df):
    df_legit = df[df['Class'] == 0]
    df_fraud = df[df['Class'] == 1]

    cols = [c for c in df.columns if c != "Class"]
    plots_per_row = 5
    n_rows = math.ceil(len(cols) / plots_per_row)

    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(20, 5 * n_rows))

    if n_rows == 1:
        axes = [axes]

    for idx, col in enumerate(cols):
        r = idx // plots_per_row
        c = idx % plots_per_row
        ax = axes[r][c]
        
        sns.kdeplot(df_legit[col], ax=ax, fill=True, alpha=0.3, 
                    label="Hợp lệ", color="skyblue")
        sns.kdeplot(df_fraud[col], ax=ax, fill=True, alpha=0.3, 
                    label="Gian lận", color="salmon")
        
        ax.set_title(col, fontsize=14)
        # ax.set_xlabel(col)
        ax.set_ylabel("Mật độ")
        
        if idx == 0:
            ax.legend()
        else:
            ax.legend().remove()

    for empty_idx in range(len(cols), n_rows * plots_per_row):
        r = empty_idx // plots_per_row
        c = empty_idx % plots_per_row
        axes[r][c].axis("off")

    plt.tight_layout()
    # plt.show()

    save_path = os.path.join(OUTPUT_DIR, "feature_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def correlation_matrix(df):
    plt.figure(figsize=(8, 8))

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=False,
        linewidths=0.5,
        square=True,
        cbar=True
    )

    plt.title("Correlation Heatmap of Dataset Features")
    plt.tight_layout()
    # plt.show()

    save_path = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def transaction_amount(df):
    plt.figure(figsize=(5, 5))

    # Lấy dữ liệu theo từng class
    class0 = df[df["Class"] == 0]["Amount"]
    class1 = df[df["Class"] == 1]["Amount"]

    # Vẽ scatter thẳng hàng
    plt.scatter([0]*len(class0), class0, alpha=0.5, s=20, color='gray')
    plt.scatter([1]*len(class1), class1, alpha=0.5, s=20, color='gray')

    plt.title("Lượng giao dịch theo các lớp")
    plt.xlabel("Class (0: Hợp lệ, 1: Gian lận)")
    plt.ylabel("Lượng giao dịch")

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks([0, 1], ["0", "1"], fontsize=10)
    plt.xlim(-0.5, 1.5)

    # plt.show()
    save_path = os.path.join(OUTPUT_DIR, "transaction_amount.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def class_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df['Class'])

    plt.title("Phân phối các lớp")
    plt.xlabel("Class (0: Hợp lệ, 1: Gian lận)")
    plt.ylabel("Số mẫu")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

def main():
    df = pd.read_csv(FILE_PATH)

    # vẽ phân phối các lớp
    class_distribution(df)
    # vẽ lượng giao dịch theo các lớp
    transaction_amount(df)
    # vẽ ma trận tương quan
    correlation_matrix(df)
    # vẽ phân phối đặc trưng
    feature_distribution(df)

if __name__ == "__main__":
    main()
