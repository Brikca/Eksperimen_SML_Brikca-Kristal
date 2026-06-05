import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):

    print("=" * 50)
    print("MEMULAI PREPROCESSING DATASET TELCO CHURN")
    print("=" * 50)

    # ==================================================
    # LOAD DATA
    # ==================================================

    df = pd.read_csv(input_path)

    print("\nDataset berhasil dimuat")
    print(f"Ukuran dataset awal: {df.shape}")

    # ==================================================
    # DROP CUSTOMER ID
    # ==================================================

    df.drop(
        columns=["customerID"],
        inplace=True
    )

    print("\nKolom customerID berhasil dihapus")

    # ==================================================
    # CEK DUPLIKASI
    # ==================================================

    duplicate_count = df.duplicated().sum()

    print(
        f"Jumlah data duplikat setelah customerID dihapus: {duplicate_count}"
    )

    print(
        "Data duplikat tidak dihapus karena kemungkinan "
        "merepresentasikan pelanggan berbeda dengan karakteristik yang sama."
    )

    # ==================================================
    # KONVERSI TOTALCHARGES
    # ==================================================

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print("\nKonversi TotalCharges berhasil dilakukan")

    # ==================================================
    # CEK MISSING VALUE
    # ==================================================

    print("\nJumlah missing value sebelum penanganan:")

    print(df.isnull().sum())

    # ==================================================
    # HANDLE MISSING VALUE
    # ==================================================

    df.dropna(
        subset=["TotalCharges"],
        inplace=True
    )

    print("\nJumlah missing value setelah penanganan:")

    print(df.isnull().sum())

    print(f"\nUkuran dataset setelah drop missing value: {df.shape}")

    # ==================================================
    # PENYEDERHANAAN KATEGORI
    # ==================================================

    replace_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    for col in replace_cols:

        df[col] = df[col].replace({
            "No internet service": "No",
            "No phone service": "No"
        })

    print("\nPenyederhanaan kategori berhasil dilakukan")

    # ==================================================
    # ENCODING TARGET
    # ==================================================

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    print("\nEncoding target Churn berhasil dilakukan")

    # ==================================================
    # ONE HOT ENCODING
    # ==================================================

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns.tolist()

    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    print("\nOne Hot Encoding berhasil dilakukan")

    # ==================================================
    # DETEKSI OUTLIER (IQR)
    # ==================================================

    numerical_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    print("\nDeteksi Outlier Menggunakan IQR")

    for col in numerical_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        outlier_count = (
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ).sum()

        print(f"{col}: {outlier_count} outlier")

    print(
        "\nOutlier tidak ditangani karena tidak ditemukan "
        "nilai ekstrem yang signifikan."
    )

    # ==================================================
    # SCALING
    # ==================================================

    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    print("\nStandardisasi berhasil dilakukan")

    # ==================================================
    # INFORMASI DATA AKHIR
    # ==================================================

    print("\nInformasi Dataset Setelah Preprocessing")

    print(df.info())

    print("\n5 Data Teratas")

    print(df.head())

    print(f"\nShape Akhir Dataset: {df.shape}")

    # ==================================================
    # SAVE DATASET
    # ==================================================

    df.to_csv(
        output_path,
        index=False
    )

    print("\nDataset berhasil disimpan")

    print(f"Lokasi file: {output_path}")

    print("\nPREPROCESSING SELESAI")


# ==================================================
# MAIN PROGRAM
# ==================================================

if __name__ == "__main__":

    INPUT_PATH = "telco_raw/telco_raw.csv"
    OUTPUT_PATH = "preprocessing/telco_preprocessed.csv"

    preprocess_data(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH
    )
