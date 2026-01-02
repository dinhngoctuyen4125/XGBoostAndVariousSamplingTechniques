## Dự án Phát hiện Gian lận Thẻ tín dụng (Credit Card Fraud Detection)

Dự án này tập trung vào việc xây dựng mô hình phân loại gian lận giao dịch thẻ tín dụng bằng thuật toán **XGBoost**, kết hợp với nhiều kỹ thuật **Data Resampling** khác nhau để giải quyết vấn đề mất cân bằng dữ liệu cực kỳ nghiêm trọng (Imbalanced Data).

### Tổng quan Dataset

* **Nguồn:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Đặc điểm:** Tập dữ liệu chứa các giao dịch được thực hiện bởi thẻ tín dụng trong hai ngày, trong đó có 492 vụ gian lận trên tổng số 284,807 giao dịch.
* **Lưu ý:** Do giới hạn kích thước file, tập dữ liệu `creditcard.csv` không được đính kèm trong repository này. Bạn cần tải về và đặt ở thư mục gốc.

---

### Cấu trúc thư mục

```text
.
├── ketqua/                     # Chứa file kết quả kết quả định lượng và confused matrix
├── visualization/              # Chứa các biểu đồ phân tích đặc trưng dữ liệu
├── creditcard.csv              # Dataset (Tải riêng)
├── preprocessing.py            # Tiền xử lý, Feature Engineering và các hàm Sampling
├── data_visualization.py       # File tạo các biểu đồ phân tích dữ liệu
├── learning_visualization.py   # File tạo các biểu đồ phân tích trong quá trình học
├── run.py                      # File huấn luyện và đánh giá mô hình
├── README.md                   # Tài liệu hướng dẫn
└── .gitignore                  # Các file không đẩy lên Git (.csv)

```

---

### Các kỹ thuật Sampling sử dụng

Dự án thử nghiệm và so sánh 4 phương pháp cân bằng dữ liệu:

1. **Undersampling:** Sử dụng `RandomUnderSampler` để giảm bớt số lượng mẫu của lớp đa số.
2. **Oversampling:** Sử dụng `BorderlineSMOTE` để tạo thêm các mẫu giả lập dựa trên ranh giới giữa hai lớp.
3. **CTGAN (Conditional GAN):** Sử dụng mạng GAN để sinh dữ liệu giả lập cho lớp thiểu số (Gian lận).
4. **TVAE (Tabular Variational Autoencoder):** Sử dụng mô hình Autoencoder để học phân phối của dữ liệu gian lận và sinh mẫu mới.

---

### Hướng dẫn thực hiện

#### 1. Cài đặt thư viện

Yêu cầu Python 3.8+. Cài đặt các thư viện cần thiết:

```bash
pip install pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn sdv

```

#### 2. Phân tích dữ liệu

Chạy script sau để tạo các biểu đồ phân phối và tương quan trong thư mục `visualization/`:

```bash
python data_visualization.py

```

#### 3. Huấn luyện và Đánh giá

Chạy script chính để huấn luyện mô hình XGBoost với lần lượt 4 phương pháp sampling:

```bash
python run.py

```

Sau khi chạy xong:

* File `ketqua/results.txt` sẽ lưu các chỉ số: Accuracy, Precision, Recall, F1-Score.
* Các ảnh ma trận nhầm lẫn (`confusion_matrix`) sẽ được lưu vào thư mục `ketqua/`.

---

### Quy trình xử lý dữ liệu (Preprocessing)

1. **Feature Engineering:** Trích xuất đặc trưng thời gian (Giờ trong ngày) và phân loại thành các buổi: Sáng, Chiều, Tối, Đêm.
2. **Scaling:** Sử dụng `MinMaxScaler` để đưa các giá trị về khoảng .
3. **Threshold:** Sử dụng ngưỡng phân loại (Threshold) là **0.65** thay vì 0.5 mặc định để tối ưu hóa khả năng nhận diện gian lận.

---

### Kết quả mong đợi

Mô hình sẽ được đánh giá dựa trên các chỉ số ưu tiên cho dữ liệu mất cân bằng:

* **Accuracy:** đo tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra.
* **Precision:** đo tỷ lệ giao dịch thực sự gian lận trong số những giao dịch mà mô hình dự đoán là gian lận.
* **Recall:** đo tỷ lệ giao dịch gian lận được mô hình phát hiện đúng.
* **F1-score:** là trung bình điều hòa giữa Precision và Recall.