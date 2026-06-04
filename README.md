<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:8b5cf6&height=120&section=header" width="100%">

<div align="center">
  <!-- Nanti Anda bisa ganti src gambar di bawah ini dengan screenshot Streamlit Anda -->
  <img src="https://via.placeholder.com/800x400/0f172a/3b82f6?text=Cognitive+Distortion+Dashboard+Screenshot" alt="AI Dashboard" width="100%" style="border-radius: 12px;">
  
  <br />
  <br />

  # 🧠 NEURA-COG AI
  <a href="https://github.com/HyLuthfi"><img src="https://readme-typing-svg.demolab.com?font=Outfit&weight=800&size=22&pause=1000&color=3B82F6&center=true&vCenter=true&width=800&lines=Enterprise-Grade+AI+Text+Classifier;Real-Time+Cognitive+Distortion+Analysis;RoBERTa+%2B+LoRA+Micro-Adapter;HuggingFace+Cloud+Integration" alt="Typing SVG" /></a>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  </p>

  <p align="center">
    Sebuah sistem cerdas berbasis <b>Natural Language Processing (NLP)</b> untuk mendeteksi dan mengkategorikan <i>Cognitive Distortions</i> secara real-time. Dibangun dengan arsitektur <b>RoBERTa + PEFT LoRA</b> dan antarmuka <b>Premium Dark Theme</b> untuk analisis psikologis forensik.
  </p>
</div>

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 🚀 Fitur Utama

<table align="center" width="100%">
  <tr>
    <td width="50%" valign="top">
      <b>Dual-Stage Inference Pipeline</b><br/>
      Analisis berjenjang: Deteksi biner (Normal/Distorsi) dilanjutkan dengan klasifikasi 11 kelas spesifik secara instan.
    </td>
    <td width="50%" valign="top">
      <b>Cloud-Native Architecture</b><br/>
      Penarikan model Level 1 secara otomatis dari peladen HuggingFace untuk efisiensi penyimpanan dan menghindari OOM (<i>Out of Memory</i>).
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <b>LoRA Micro-Adapter</b><br/>
      Teknologi injeksi *weights* berukuran super ringan (4.7 MB) untuk inferensi 11-Class tanpa mengorbankan performa *backbone*.
    </td>
    <td width="50%" valign="top">
      <b>Forensic Analytics Dashboard</b><br/>
      Visualisasi probabilitas model dan metrik sentimen interaktif dengan estetika mode gelap profesional.
    </td>
  </tr>
</table>

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 📸 Galeri Sistem

### 1. Binary Detection Analytics (Level 1)
Deteksi akurat apakah sebuah kalimat mengandung unsur distorsi kognitif.
*(Screenshot akan ditambahkan)*

### 2. Multiclass Distortion Matrix (Level 2)
Pemetaan probabilistik dari 11 kategori distorsi psikologis.
*(Screenshot akan ditambahkan)*

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 🧬 Arsitektur AI & Teknologi

### Machine Learning Core
- **Base Model:** `w11wo/indonesian-roberta-base-sentiment-classifier`
- **Level 1 (Binary):** Full Fine-Tuning 
- **Level 2 (Multiclass):** Parameter-Efficient Fine-Tuning (LoRA)
- **Rank (r):** 16
- **Target Modules:** `query`, `value`

### Repositori & Deployment
- **HuggingFace Hub:** Tempat bersarangnya *weights* model utama.
- **GitHub:** Penyimpanan kode sumber dan *adapter* mikro.
- **Streamlit Cloud:** Infrastruktur hosting web-app.

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 🌐 Koneksi Cloud API

Aplikasi ini dirancang untuk berkomunikasi dengan *cloud repository* secara asinkron saat *cold start*:

### 1. Level 1 Pipeline
Menggunakan arsitektur `AutoModelForSequenceClassification`:
**Endpoint:** `https://huggingface.co/Luthfi22/indo-cognitive-distortion-binary`

### 2. Level 2 Pipeline
Menggunakan arsitektur `PeftModel` yang menginjeksi adapter lokal ke *base model* HuggingFace:
**Base:** `w11wo/indonesian-roberta-base-sentiment-classifier` + **Adapter:** `./lora_adapter_level2`

<p align="center"><img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%"></p>

## 🛠️ Panduan Instalasi (Lokal)

Ikuti langkah-langkah berikut untuk menjalankan dasbor AI ini di *local machine*.

1. **Clone Repository**
   ```bash
   git clone https://github.com/HyLuthfi/Cognitive-Distortion-App.git
   cd Cognitive-Distortion-App
   ```

2. **Setup Environment**
   Dianjurkan menggunakan Python 3.9 - 3.11.
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Streamlit Engine**
   ```bash
   streamlit run app.py
   ```
   Akses `http://localhost:8501` di *browser* Anda.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:8b5cf6&height=120&section=footer" width="100%"/>
</p>
