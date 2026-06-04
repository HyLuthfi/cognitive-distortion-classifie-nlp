<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:8b5cf6&height=120&section=header" width="100%">

<div align="center">
  <img src="https://via.placeholder.com/1000x500/F8FAFC/2563EB?text=Cognitive+Distortion+AI+Dashboard" alt="AI Dashboard" width="100%" style="border-radius: 16px; box-shadow: 0px 10px 20px rgba(0,0,0,0.1);">
  
  <br />
  <br />

  # 🧠 NEURA-COG AI
  <a href="https://github.com/HyLuthfi"><img src="https://readme-typing-svg.demolab.com?font=Plus+Jakarta+Sans&weight=800&size=24&pause=1000&color=2563EB&center=true&vCenter=true&width=800&lines=Enterprise-Grade+AI+Text+Classifier;Real-Time+Cognitive+Distortion+Analysis;RoBERTa+%2B+LoRA+Micro-Adapter;Premium+Glassmorphism+Aesthetics" alt="Typing SVG" /></a>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  </p>

  <p align="center" style="font-size: 1.1rem; color: #475569;">
    Sebuah sistem cerdas berbasis <b>Natural Language Processing (NLP)</b> untuk mendeteksi dan mengkategorikan <i>Cognitive Distortions</i> secara real-time. Dibangun dengan arsitektur <b>RoBERTa + PEFT LoRA</b> dan antarmuka <b>Premium Light Glassmorphism</b> untuk analisis psikologis forensik berskala besar.
  </p>
</div>

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 🚀 Fitur & Inovasi Utama

<table align="center" width="100%">
  <tr>
    <td width="50%" valign="top">
      <b>⚡ Dual-Stage Inference Pipeline</b><br/>
      Analisis berjenjang yang mensimulasikan logika penyaringan psikologis sejati: Deteksi Biner (Normal vs Distorsi) di Tahap 1, dilanjutkan dengan klasifikasi 11 Distorsi Spesifik di Tahap 2 secara instan.
    </td>
    <td width="50%" valign="top">
      <b>☁️ Cloud-Native Architecture</b><br/>
      Penarikan *weights* Model Level 1 secara otomatis dari peladen HuggingFace untuk efisiensi repositori dan isolasi beban memori di lingkungan *cloud* berbasis Streamlit.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <b>🔬 LoRA Micro-Adapter</b><br/>
      Teknologi injeksi *parameter-efficient fine-tuning* (PEFT). Model Level 2 menggunakan bobot tambahan super ringan (~4.7 MB) untuk pengenalan 11 kelas tanpa merusak bobot dasar RoBERTa (>500 MB).
    </td>
    <td width="50%" valign="top">
      <b>🎨 Premium Glassmorphism UI</b><br/>
      Antarmuka *Clean Light Theme* bernuansa biru modern (*slate white*) dengan *radial background*, tab melayang (*floating tabs*), dan animasi mikroskopis interaktif.
    </td>
  </tr>
</table>

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 🧬 Metodologi Pembelajaran Mesin

Sistem ini mengeksekusi data melalui alur forensik yang ketat. Proses ini merupakan representasi otentik dari dua eksperimen Jupyter Notebook (`binary-classification-roberta-sentiment.ipynb` dan `multiclass-classification-roberta-lora.ipynb`).

### Tahap 0: Pra-Pemrosesan (*Data Cleansing*)
*   Penghapusan token *noise* menggunakan regex (Simbol `$`, URL `http`, *Mentions* `@`).
*   Batas maksimal *sequence length* dioptimalkan pada **256 Token**.
*   Tokenisasi menggunakan `w11wo/indonesian-roberta-base-sentiment-classifier`.

### Level 1: Binary Classification Gate (RoBERTa)
Gerbang pemisah antara pikiran sehat (Normal) dan terdistorsi. Menggunakan metode **Full Fine-Tuning**.
*   **Learning Rate:** `2e-5` (Cosine Scheduler, Warmup 0.1)
*   **Epochs:** `20`
*   **Batch Size:** `16`
*   **Weight Decay:** `0.05`
*   **Akurasi Akhir:** **96.50%**

### Level 2: Multiclass Extractor (LoRA Adapter)
Pengekstrak akar distorsi (11 jenis *Cognitive Distortion*). Menggunakan metode adaptasi matriks rendah **Low-Rank Adaptation**.
*   **Target Modules:** `["query", "value"]`
*   **Rank (r) / Alpha / Dropout:** `16` / `32` / `0.1`
*   **Learning Rate:** `3e-4`
*   **Epochs:** `15`
*   **Akurasi Akhir:** **81.82%**

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 📊 Kinerja Kuantitatif Model (Level 2)

Evaluasi terhadap kelas minoritas terdistribusi untuk mendiagnosis distorsi secara spesifik:

| Kategori Distorsi | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **All-or-nothing** | 0.62 | 0.67 | 0.64 | 24 |
| **Discounting the positives** | 0.89 | 0.89 | 0.89 | 36 |
| **Emotional Reasoning** | 1.00 | 0.33 | 0.50 | 6 |
| **Fortune-telling** | 0.33 | 0.11 | 0.17 | 9 |
| **Labeling** | 0.79 | 0.86 | 0.83 | 58 |
| **Magnification/Minimization** | 1.00 | 0.46 | 0.63 | 13 |
| **Mental filter** | 0.81 | 0.89 | 0.85 | 28 |
| **Mind Reading** | 0.78 | 0.86 | 0.82 | 58 |
| **Overgeneralization** | 0.66 | 0.72 | 0.69 | 32 |
| **Personalization/Blame** | 0.93 | 0.91 | 0.92 | 43 |
| **Should statement** | 0.96 | 0.95 | 0.95 | 56 |
| ***Macro Average*** | *0.80* | *0.70* | *0.72* | *363* |

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 🛠️ Panduan Eksekusi Lokal

Sistem ini didesain agar sangat portabel. Ikuti perintah berikut untuk meniru (*replicate*) lingkungan awan (cloud) ke mesin lokal Anda:

1. **Kloning Repositori**
   ```bash
   git clone https://github.com/HyLuthfi/Cognitive-Distortion-App.git
   cd Cognitive-Distortion-App
   ```

2. **Inisialisasi Lingkungan & Pustaka**
   Sangat disarankan menggunakan *Virtual Environment* dengan Python 3.9 - 3.11.
   ```bash
   pip install -r requirements.txt
   ```

3. **Injeksi Tensor & Nyalakan Dashboard**
   ```bash
   streamlit run app.py
   ```
   Web-App akan terbuka secara otonom di `http://localhost:8501`. Selama inisialisasi pertama, sistem akan mengunduh bobot Level 1 (~500MB) langsung dari Hub HuggingFace `Luthfi22/indo-cognitive-distortion-binary`.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:8b5cf6&height=120&section=footer" width="100%"/>
</p>
