import pandas as pd
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.utils import shuffle

RANDOM_STATE = 42

def tvae_data(X_train, y_train):
    fraud_data = X_train[y_train == 1].copy().reset_index(drop=True)
    fraud_data['Class'] = 1
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(fraud_data)

    tvae = TVAESynthesizer(
        metadata,
        epochs=100,
        batch_size=500
    )

    # chỉ học trên dữ liệu fraud
    tvae.fit(fraud_data)

    n_fraud_needed = (y_train == 0).sum() - len(fraud_data)
    synthetic_fraud = tvae.sample(num_rows=n_fraud_needed)

    X_train_scaled = pd.concat([X_train, synthetic_fraud.drop('Class', axis=1)])
    y_train_scaled = pd.concat([y_train, synthetic_fraud['Class']])

    X_train_scaled, y_train_scaled = shuffle(X_train_scaled, y_train_scaled, random_state=RANDOM_STATE)

    return X_train_scaled, y_train_scaled

def ctgan_data(X_train, y_train):
    fraud_data = X_train[y_train == 1].copy().reset_index(drop=True)
    fraud_data['Class'] = 1
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(fraud_data)

    ctgan = CTGANSynthesizer(
        metadata,
        epochs=100,
        batch_size=min(100, len(fraud_data)),
        generator_dim=(128, 128),
        discriminator_steps=2,
        discriminator_dim=(128, 128),
        verbose=True
    )

    # chỉ học trên dữ liệu fraud
    ctgan.fit(fraud_data)

    n_fraud_needed = (y_train == 0).sum() - len(fraud_data)
    synthetic_fraud = ctgan.sample(num_rows=n_fraud_needed)

    X_train_scaled = pd.concat([X_train, synthetic_fraud.drop('Class', axis=1)])
    y_train_scaled = pd.concat([y_train, synthetic_fraud['Class']])

    X_train_scaled, y_train_scaled = shuffle(X_train_scaled, y_train_scaled, random_state=RANDOM_STATE)

    return X_train_scaled, y_train_scaled

def oversampling_data(X_train, y_train):
    scaler = BorderlineSMOTE()
    X_train_scaled, y_train_scaled = scaler.fit_resample(X_train, y_train)

    return X_train_scaled, y_train_scaled

def undersampling_data(X_train, y_train):
    scaler = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_scaled, y_train_scaled = scaler.fit_resample(X_train, y_train)

    return X_train_scaled, y_train_scaled

def assign_day_segment(hour):
    if 6 <= hour < 12:
        return 'Morning'      # 6h-12h: Buổi sáng
    elif 12 <= hour < 18:
        return 'Afternoon'    # 12h-18h: Buổi chiều
    elif 18 <= hour < 24:
        return 'Evening'      # 18h-24h: Buổi tối
    else:
        return 'Night'        # 0h-6h: Buổi đêm

def compute_time_col(data):
    data['Hour'] = (data['Time'] // 3600) % 24
    data['Day_Segment'] = data['Hour'].apply(assign_day_segment)
    data = pd.get_dummies(data, columns=['Day_Segment'], drop_first=True)
    data['Day_Segment_Evening'] = data['Day_Segment_Evening'].astype(int)
    data['Day_Segment_Morning'] = data['Day_Segment_Morning'].astype(int)
    data['Day_Segment_Night'] = data['Day_Segment_Night'].astype(int)

    return data

def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_clean = compute_time_col(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )

    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, y_train, X_test_scaled, y_test