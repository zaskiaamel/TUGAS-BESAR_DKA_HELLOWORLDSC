
# ✈️ Analisis Tren Harga Tiket Pesawat pada Rute Domestik✈️
### Menggunakan Fuzzy Inference System untuk Rekomendasi Waktu Pembelian

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-F59E0B?style=for-the-badge)]()
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

<br/>

> **AeroForecast: Smart Booking Assistant** — Sistem rekomendasi berbasis *Fuzzy Inference System* yang menganalisis pola dan tren harga tiket pesawat untuk membantu penumpang menentukan waktu pembelian tiket yang paling optimal.

<br/>

---

</div>

## 👥 Tim Pengembang

<div align="center">

| Nama | NIM | Email | Kelas |
|:----:|:---:|:-----:|:-----:|
| Ghaisan Hanifah Siregar | 103012400322 | ghaisanhanifah@student.telkomuniversity.ac.id | IF-48-01 |
| Zaskia Amelia Nurudin | 103012400290 | zaskiaamelianurudin@student.telkomuniversity.ac.id | IF-48-01 |
| Rifqy Anugerah | 103012480039 | rifqyanugerahra@student.telkomuniversity.ac.id | IF-48-01 |
| Nailah Dhiya Marzuqoh | 103012400421 | nailahdhiyamarzuqoh@student.telkomuniversity.ac.id | IF-48-01 |


**S1 Informatika · Fakultas Informatika · Telkom University, Bandung**
**Mata Kuliah:** CAK2HAB3 – Dasar Kecerdasan Artifisial · Semester Genap 2025/2026

</div>

<br/>

---

## 📋 Daftar Isi

- [Latar Belakang](#-latar-belakang)
- [Tujuan Proyek](#-tujuan-proyek)
- [Dataset](#-dataset)
- [Metodologi](#-metodologi)
- [Struktur Proyek](#-struktur-proyek)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil dan Analisis](#-hasil-dan-analisis)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Referensi](#-referensi)

<br/>

---

## 🌐 Latar Belakang

Industri penerbangan domestik Indonesia mengalami pertumbuhan signifikan dalam satu dekade terakhir. Namun, salah satu tantangan terbesar yang dihadapi konsumen adalah **fluktuasi harga tiket pesawat** yang sulit diprediksi dan sering kali tidak transparan.

Maskapai penerbangan menggunakan strategi ***dynamic pricing*** — penyesuaian tarif secara real-time berdasarkan:

- 🪑 Tingkat hunian kursi (seat availability)
- 📅 Waktu pemesanan relatif terhadap keberangkatan
- 🌤️ Musim dan hari-hari besar nasional
- 🏆 Kompetisi harga antar maskapai
- 🎫 Kelas layanan (Economy vs Business)

Kondisi ini menciptakan **ketidakpastian** bagi penumpang — harga tiket dapat berfluktuasi **hingga 200–300%** tergantung kapan tiket dibeli. Proyek ini hadir sebagai solusi cerdas berbasis **Fuzzy Inference System (FIS)** yang mampu menangani ambiguitas tersebut dan memberikan rekomendasi yang intuitif.

<br/>

---

## 🎯 Tujuan Proyek

```
┌────────────────────────────────────────────────────────────┐
│                    PIPELINE AEROFORECAST                   │
│                                                            │
│  Dataset  ──►  EDA & Stats  ──►  Time Series  ──►  FIS     │
│                                                    │       │
│                                            Mamdani + Sugeno│
│                                                    │       │
│                                      Evaluasi & Rekomendasi
└────────────────────────────────────────────────────────────┘
```

1. **Analisis Eksplorasi Data (EDA)** — mengungkap pola dan distribusi harga tiket pesawat secara komprehensif
2. **Statistik Deskriptif** — merangkum karakteristik dataset secara kuantitatif
3. **Time Series Analysis** — mengidentifikasi tren harga berdasarkan waktu pemesanan dan sweet spot pembelian
4. **Implementasi FIS Mamdani** — klasifikasi harga menggunakan output himpunan fuzzy (defuzzifikasi centroid)
5. **Implementasi FIS Sugeno** — klasifikasi harga menggunakan output fungsi konstan (defuzzifikasi weighted average)
6. **Evaluasi Komparatif** — membandingkan performa kedua metode menggunakan metrik standar

<br/>

---

## 📊 Dataset

**Sumber:** [Flight Price Prediction — Kaggle (Shubham Bathwal)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

Dataset berisi **300.000+ entri** data harga tiket penerbangan domestik dengan fitur-fitur berikut:

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| `airline` | Kategorik | Nama maskapai penerbangan (6 maskapai) |
| `flight` | Kategorik | Kode penerbangan unik |
| `source_city` | Kategorik | Kota asal (6 kota besar) |
| `departure_time` | Kategorik | Slot waktu keberangkatan (6 kategori) |
| `stops` | Kategorik | Jumlah transit: `zero`, `one`, `two_or_more` |
| `arrival_time` | Kategorik | Slot waktu kedatangan (6 kategori) |
| `destination_city` | Kategorik | Kota tujuan (6 kota besar) |
| `class` | Kategorik | Kelas penerbangan: `Economy` / `Business` |
| `duration` | Numerik | Durasi penerbangan dalam jam (0.83 – 50 jam) |
| `days_left` | Numerik | Hari tersisa sebelum keberangkatan (1 – 49 hari) |
| `price` ⭐ | Numerik | **Harga tiket (variabel target)** |

<br/>

---

## 🔬 Metodologi

### 1. Prapemrosesan Data

```
Raw Dataset
    │
    ├── Inspeksi Awal (dimensi, tipe data, nilai unik)
    ├── Penanganan Missing Value (imputasi median/modus)
    ├── Deteksi & Eliminasi Outlier (metode IQR)
    ├── Pemilihan Fitur (korelasi + domain knowledge)
    ├── Encoding Kategorik (Label & One-Hot Encoding)
    ├── Normalisasi Numerik (Min-Max Scaling → [0, 1])
    ├── Labeling Kategori Harga (persentil ke-33 & ke-67)
    └── Train-Test Split (80:20, stratified)
```

### 2. Exploratory Data Analysis (EDA)

- 📊 Distribusi harga (histogram, boxplot, violin plot)
- 🏢 Perbandingan harga per maskapai
- 🗺️ Heatmap harga per rute (kota asal × kota tujuan)
- ⏰ Analisis harga berdasarkan waktu keberangkatan dan jumlah transit
- 📈 Matriks korelasi antarvariabel

### 3. Time Series Analysis

- Tren harga rata-rata berdasarkan `days_left` (rolling mean)
- Analisis volatilitas harga per interval waktu pemesanan
- Identifikasi **sweet spot** pembelian tiket
- Segmentasi pola temporal Economy vs Business

### 4. Rancangan Fuzzy Inference System

**Variabel Input:**

| Variabel | Rentang | Nilai Linguistik |
|----------|---------|-----------------|
| `days_left` | 0 – 49 hari | Mendekati · Sedang · Jauh |
| `duration` | 0.83 – 50 jam | Singkat · Menengah · Panjang |
| `stops` | 0 – 2 | Langsung · Satu Transit · Banyak Transit |

**Variabel Output:**

| Variabel | Nilai Linguistik |
|----------|-----------------|
| `price_category` | 🟢 Murah · 🟡 Sedang · 🔴 Mahal |

**Contoh Fuzzy Rules:**
```
IF days_left = Mendekati AND stops = Langsung       THEN price = Mahal
IF days_left = Jauh      AND stops = Banyak_Transit THEN price = Murah
IF days_left = Sedang    AND duration = Singkat     THEN price = Sedang
IF days_left = Mendekati AND duration = Panjang     THEN price = Mahal
IF days_left = Jauh      AND duration = Menengah    THEN price = Sedang
```

**Perbandingan Mamdani vs Sugeno:**

| Aspek | Mamdani | Sugeno |
|-------|---------|--------|
| Output | Himpunan fuzzy | Fungsi konstan/linear |
| Defuzzifikasi | Centroid | Weighted Average |
| Interpretabilitas | ✅ Tinggi | 🔶 Sedang |
| Efisiensi Komputasi | 🔶 Sedang | ✅ Tinggi |
| Cocok Untuk | Sistem rekomendasi | Sistem adaptif/real-time |

### 5. Evaluasi

| Metrik | Formula |
|--------|---------|
| Akurasi | (TP + TN) / (TP + TN + FP + FN) |
| Precision (Macro) | (1/N) × Σ TPᵢ / (TPᵢ + FPᵢ) |
| Recall (Macro) | (1/N) × Σ TPᵢ / (TPᵢ + FNᵢ) |
| F1-Score (Macro) | (1/N) × Σ 2·Prec·Rec / (Prec + Rec) |

Evaluasi juga mencakup: confusion matrix, analisis per kelas, dan perbandingan waktu inferensi.

<br/>

---

## 📁 Struktur Proyek

```
TUGAS-BESAR_DKA_HELLOWORLDSC/
│
├── 📂 data/
│   ├── raw/
│   │   └── flight_price.csv              # Dataset mentah dari Kaggle
│   └── processed/
│       ├── flight_price_clean.csv        # Setelah prapemrosesan
│       └── flight_price_labeled.csv      # Dengan label kategori harga
│
├── 📂 notebooks/
│   ├── 01_EDA_dan_Statistik.ipynb        # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb            # Prapemrosesan data
│   ├── 03_TimeSeries_Analysis.ipynb      # Analisis tren temporal
│   ├── 04_FIS_Mamdani.ipynb             # Implementasi FIS Mamdani
│   ├── 05_FIS_Sugeno.ipynb              # Implementasi FIS Sugeno
│   └── 06_Evaluasi_Komparatif.ipynb     # Perbandingan & evaluasi akhir
│
├── 📂 src/
│   ├── fuzzy/
│   │   ├── __init__.py
│   │   ├── membership.py                 # Fungsi keanggotaan (trimf, trapmf)
│   │   ├── rules.py                      # Basis aturan fuzzy
│   │   ├── mamdani.py                    # Engine FIS Mamdani
│   │   └── sugeno.py                     # Engine FIS Sugeno
│   └── utils/
│       ├── preprocessing.py              # Fungsi prapemrosesan
│       └── evaluation.py                 # Fungsi evaluasi model
│
├── 📂 results/
│   ├── figures/                          # Grafik dan visualisasi EDA
│   └── metrics/                          # Output metrik evaluasi
│
├── 📂 docs/
│   └── Proposal_AeroForecast.pdf         # Proposal tugas besar
│
├── 📄 requirements.txt                   # Dependensi Python
├── 📄 LICENSE
└── 📄 README.md
```

<br/>

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/zaskiaamel/TUGAS-BESAR_DKA_HELLOWORLDSC.git
cd TUGAS-BESAR_DKA_HELLOWORLDSC
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Unduh Dataset

Unduh dataset dari Kaggle:
```bash
# Menggunakan Kaggle API
kaggle datasets download -d shubhambathwal/flight-price-prediction
unzip flight-price-prediction.zip -d data/raw/
```
Atau unduh secara manual dari: [https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

### 4. Jalankan Notebook

Jalankan notebook secara berurutan sesuai nomor prefiks:

```bash
jupyter notebook
```

Urutan eksekusi yang disarankan:
```
01 → 02 → 03 → 04 → 05 → 06
EDA    Prep   TS   Mam  Sug   Eval
```

<br/>

---

## 📈 Hasil dan Analisis

> 🚧 *Bagian ini akan diperbarui setelah implementasi selesai.*

### Ringkasan Performa Model

| Model | Akurasi | Precision | Recall | F1-Score |
|-------|:-------:|:---------:|:------:|:--------:|
| FIS Mamdani | — | — | — | — |
| FIS Sugeno | — | — | — | — |
| Decision Tree *(baseline)* | — | — | — | — |
| Logistic Regression *(baseline)* | — | — | — | — |

### Temuan Utama

- 📌 **Sweet Spot Pembelian:** akan diperbarui setelah analisis time series
- 📌 **Maskapai dengan Harga Paling Stabil:** akan diperbarui setelah EDA
- 📌 **Perbandingan Mamdani vs Sugeno:** akan diperbarui setelah evaluasi

<br/>

---

## 🛠️ Teknologi yang Digunakan

```
Python 3.10+
│
├── numpy               # Komputasi numerik & implementasi FIS from scratch
├── pandas              # Manipulasi dan analisis data
├── matplotlib          # Visualisasi dasar
├── seaborn             # Visualisasi statistik
├── scikit-learn        # Preprocessing & model baseline
└── jupyter             # Lingkungan pengembangan interaktif
```

> ⭐ **Catatan:** Implementasi Fuzzy Inference System (Mamdani & Sugeno) dibuat **dari nol (from scratch)** tanpa library fuzzy eksternal seperti `scikit-fuzzy`, sebagai nilai tambah proyek ini.

<br/>

---

## 📚 Referensi

1. P. Belobaba, A. Odoni, and C. Barnhart, *The Global Airline Industry*, 2nd ed. Chichester: John Wiley & Sons, 2015.
2. K. Talluri and G. van Ryzin, *The Theory and Practice of Revenue Management*. New York: Springer, 2004.
3. E. H. Mamdani, "Application of fuzzy algorithms for control of simple dynamic plant," *Proc. Institution of Electrical Engineers*, vol. 121, no. 12, pp. 1585–1588, 1974.
4. M. Sugeno and G. Kang, "Fuzzy modeling and control of multivariable systems," *Proc. IEEE Int. Symp. Fuzzy Information Processing*, 1985, pp. 262–268.
5. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021. [Online]: https://otexts.com/fpp3
6. J. W. Tukey, *Exploratory Data Analysis*. Reading, MA: Addison-Wesley, 1977.
7. G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," *Proc. NeurIPS*, 2017, pp. 3146–3154.
8. S. Bathwal, "Flight Price Prediction Dataset," *Kaggle*, 2022. [Online]: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

<br/>

---

<div align="center">

**AeroForecast: Smart Booking Assistant**

*Tugas Besar CAK2HAB3 – Dasar Kecerdasan Artifisial*
*Semester Genap 2025/2026 · Telkom University*

<br/>

Made with ☕ by **HelloWorldSC** · IF-48-01

</div>
