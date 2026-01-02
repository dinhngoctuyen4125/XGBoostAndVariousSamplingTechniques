import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "ketqua"

def plot_learning_curve(model, X, y, scoring="f1"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Training score")
    plt.plot(train_sizes, val_mean, "o-", label="Validation score")

    plt.xlabel("Số lượng mẫu huấn luyện")
    plt.ylabel(scoring.upper())
    plt.title("Learning Curve of XGBoost")
    plt.legend()
    plt.grid(alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "learning_curve_xgboost.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# vẽ confusion matrix
def conf_matrix_visualization(conf_matrix, name_file):
    class_names = [f'Class {i}' for i in range(2)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor='black'
    )
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Nhãn Thực Tế', fontsize=14)
    plt.xlabel('Nhãn Dự Đoán', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"{name_file}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()