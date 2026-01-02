import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from preprocessing import preprocess_data, undersampling_data, oversampling_data, ctgan_data, tvae_data
from learning_visualization import plot_learning_curve, conf_matrix_visualization

FILE_PATH = './creditcard.csv'
OUTPUT_DIR = "ketqua"
THRESHOLD = 0.65

# cấu hình XGBoost
def get_model():
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric=['aucpr', 'logloss'],
        random_state=42,
        learning_rate=0.4,
        n_estimators=1000,
        tree_method='hist',
        n_jobs=-1,
    )
    
    return model

def main():
    # Tạo đường dẫn đến file ghi kết quả
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(OUTPUT_DIR, "results.txt")

    # đọc dữ liệu và tiền xử lý
    df = pd.read_csv(FILE_PATH)
    X_train, y_train, X_test, y_test = preprocess_data(df)

    # các phương pháp sampling
    sampling_types = ['undersampling', 'oversampling', 'ctgan', 'tvae']

    # vẽ learning curve cho XGBoost trên tập dữ liệu gốc
    base_model = get_model()
    plot_learning_curve(
        model=base_model,
        X=X_train,
        y=y_train,
        scoring="f1"
    )

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ THỰC NGHIỆM XGBOOST + SAMPLING\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ngưỡng phân lớp (Threshold): {THRESHOLD}\n\n")

        for sampling_type in sampling_types:
            model = get_model()

            if sampling_type == 'undersampling':
                f.write('--- Undersampling ---\n')
                X_sample, y_sample = undersampling_data(X_train, y_train)
            elif sampling_type == 'oversampling':
                f.write('--- Oversampling ---\n')
                X_sample, y_sample = oversampling_data(X_train, y_train)
            elif sampling_type == 'ctgan':
                f.write('--- CTGAN ---\n')
                X_sample, y_sample = ctgan_data(X_train, y_train)
            elif sampling_type == 'tvae':
                f.write('--- TVAE ---\n')
                X_sample, y_sample = tvae_data(X_train, y_train)

            model.fit(X_sample, y_sample)

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= THRESHOLD).astype(int)

            # đo kết quả
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            f.write(f'Accuracy:  {accuracy*100:.2f} %\n')
            f.write(f'Precision: {precision*100:.2f} %\n')
            f.write(f'Recall:    {recall*100:.2f} %\n')
            f.write(f'F1 Score:  {f1*100:.2f} %\n\n')
            f.write('-' * 40 + '\n\n')

            # vẽ ra conf matrix
            conf_matrix_visualization(conf_matrix, sampling_type)

if __name__ == "__main__":
    main()