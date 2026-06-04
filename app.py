import streamlit as st
import torch
import os
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

st.set_page_config(
    page_title="Sistem Analisis Forensik Kognitif",
    layout="wide",
    initial_sidebar_state="collapsed"
)

css_kustom = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0B0F19;
        color: #F3F4F6 !important;
    }
    
    .stApp {
        background-color: #0B0F19;
    }

    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 30px;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 10px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        padding: 10px 15px;
        color: #6B7280;
        font-weight: 600;
        font-size: 1.1rem;
        border: none !important;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        color: #F9FAFB !important;
        border-bottom: 3px solid #3B82F6 !important;
        background-color: transparent !important;
    }

    .metric-card {
        background-color: rgba(31, 41, 55, 0.4);
        backdrop-filter: blur(10px);
        color: #F9FAFB !important;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 32px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    
    .info-card {
        background-color: rgba(31, 41, 55, 0.6);
        color: #E5E7EB !important;
        border: 1px solid #374151;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
    }

    h1, h2, h3, h4, p {
        color: #F9FAFB !important;
    }

    .stButton > button {
        background-color: #2563EB;
        color: white !important;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1D4ED8;
    }
    
    .status-normal {
        color: #10B981;
        background-color: rgba(16, 185, 129, 0.1);
        padding: 6px 20px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    
    .status-distorsi {
        color: #EF4444;
        background-color: rgba(239, 68, 68, 0.1);
        padding: 6px 20px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #DC2626;
    }

    .stTextArea > div > div > textarea {
        background-color: #1F2937;
        color: #F9FAFB !important;
        border: 1px solid #374151;
        border-radius: 8px;
    }
    
    hr {
        border-color: #374151;
        margin: 2rem 0;
    }
</style>
"""
st.markdown(css_kustom, unsafe_allow_html=True)

PERANGKAT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDENTITAS_MODEL_DASAR = "w11wo/indonesian-roberta-base-sentiment-classifier"
IDENTITAS_MODEL_LEVEL_SATU = "Luthfi22/indo-cognitive-distortion-binary"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOKASI_ADAPTER_LEVEL_DUA = os.path.join(BASE_DIR, "lora_adapter_level2")

daftar_kategori_distorsi = [
    "All-or-Nothing Thinking",
    "Overgeneralization",
    "Mental Filter",
    "Disqualifying the Positive",
    "Jumping to Conclusions",
    "Magnification / Minimization",
    "Emotional Reasoning",
    "Should Statements",
    "Labeling",
    "Personalization",
    "Blaming"
]

@st.cache_resource
def muat_tokenizer():
    return AutoTokenizer.from_pretrained(IDENTITAS_MODEL_DASAR)

@st.cache_resource
def muat_model_level_satu():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(IDENTITAS_MODEL_LEVEL_SATU)
        model.to(PERANGKAT)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error memuat Level 1 (HuggingFace): {str(e)}")
        return None

@st.cache_resource
def muat_model_level_dua():
    try:
        if not os.path.exists(LOKASI_ADAPTER_LEVEL_DUA):
            st.error(f"Error: Folder adapter tidak ditemukan di {LOKASI_ADAPTER_LEVEL_DUA}")
            return None
        model_dasar = AutoModelForSequenceClassification.from_pretrained(
            IDENTITAS_MODEL_DASAR,
            num_labels=11,
            ignore_mismatched_sizes=True
        )
        model_lora = PeftModel.from_pretrained(model_dasar, LOKASI_ADAPTER_LEVEL_DUA)
        model_lora.to(PERANGKAT)
        model_lora.eval()
        return model_lora
    except Exception as e:
        st.error(f"Error memuat Level 2 (LoRA): {str(e)}")
        return None

def tampilkan_gambar_visualisasi(nama_file, deskripsi):
    lokasi_direktori = os.path.dirname(os.path.abspath(__file__))
    path_visual = os.path.join(lokasi_direktori, "img", nama_file)
    
    if os.path.exists(path_visual):
        gambar = Image.open(path_visual)
        st.image(gambar, caption=deskripsi, use_container_width=True)
    else:
        st.info(f"Aset visual {nama_file} belum tersedia di direktori img.")

st.markdown("<h1 style='text-align: center; margin-top: 1rem;'>Dasbor Analisis Forensik Kognitif</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #9CA3AF !important; margin-bottom: 2rem;'>Sistem Otomatisasi Deteksi Distorsi Kognitif Menggunakan RoBERTa dan LoRA Adapter</p>", unsafe_allow_html=True)

tab_analisis, tab_model_1, tab_model_2 = st.tabs([
    "Inspeksi Kognitif", 
    "Detail Model 1", 
    "Detail Model 2"
])

with tab_analisis:
    st.markdown("<br>", unsafe_allow_html=True)
    
    kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])

    with kolom_tengah:
        st.markdown("<div class='info-card'><b>Instruksi Operasional:</b> Masukkan teks atau kalimat berbahasa Indonesia yang ingin dianalisis secara psikologis. Sistem akan melakukan penyaringan biner sebelum mengekstraksi jenis distorsi spesifik.</div>", unsafe_allow_html=True)
        
        sampel_cepat = {
            "--- Ketik Manual Secara Bebas ---": "",
            "[Normal] Saya percaya pencapaian dalam hidup adalah hasil proses...": "Tidak pernah, karena saya percaya bahwa setiap yang kita capai atau kita dapatkan dalam hidup adalah hasil dari proses kita sendiri",
            "[Normal] Teman bilang soal ujian sulit, saya tetap fokus...": "teman bilang soal ujian akan sesulit itu dan semua akan remedial, saya mengabaikan dan biarlah tetap fokus apa dipelajari dan dikerjakan",
            "[Normal] Menyaring nasihat yang diberikan...": "Saya selalu menyaring omongan atau nasihat yang diberikan kepada saya karena tak semua omongan orang bisa dipercaya dan benar",
            "[Distorsi] Mereka menjauh, saya kira tidak menyukai saya...": "teman teman dekat saya saat sma sering berbisik saat ada saya dan menjauh perlahan, saya kira mereka tidak menyukai saya dan mengomongi saya dibelakang saya, ternyata benar",
            "[Distorsi] Orang tua bertengkar, pasti selalu bertengkar lagi...": "Saat orang tua saya bertengkar mempermasalahkan setiap hal, setelahnya saya berpikir pasti mereka akan selalu bertengkar lagi lagi dan lagi",
            "[Distorsi] Gagal melakukan hal tertentu, merasa tidak berguna...": "Masa kritis saya masa SMP-Awal SMA, pokoknya apabila saya gagal lakukan hal tertentu misalkan tugas atau terima teguran, sensitif saya meningkat dan merasa saya tidak berguna dan gagal",
            "[Distorsi] Nilai di bawah 90, prestasi tidak bagus...": "Saya selalu berpikir jika nilai saya di bawah 90, maka prestasi akademis saya tidak bagus."
        }
        
        pilihan = st.selectbox("Pilih Sampel Cepat dari Dataset:", list(sampel_cepat.keys()))
        
        input_teks = st.text_area("Teks Analisis", value=sampel_cepat[pilihan], height=150, placeholder="Ketik teks di sini...", label_visibility="collapsed")
        tombol_analisis = st.button("Jalankan Inferensi Forensik")

    if tombol_analisis and input_teks:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Laporan Hasil Ekstraksi Kognitif</h2>", unsafe_allow_html=True)
        
        with st.spinner("Menginisialisasi & mengunduh bobot AI dari Cloud (Proses ini memakan waktu 1-3 menit pada percobaan pertama)..."):
            tokenizer = muat_tokenizer()
            model_satu = muat_model_level_satu()
            model_dua = muat_model_level_dua()

        if model_satu is None or model_dua is None:
            st.error("Gagal memuat arsitektur jaringan saraf tiruan. Pastikan koneksi internet stabil untuk mengunduh model dari repositori cloud, dan adapter lokal tersedia.")
        else:
            with st.spinner("Memproses aktivasi lapisan RoBERTa..."):
                input_tensor = tokenizer(
                    input_teks, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512, 
                    padding="max_length"
                ).to(PERANGKAT)
                
                with torch.no_grad():
                    keluaran_satu = model_satu(**input_tensor)
                    probabilitas_satu = torch.nn.functional.softmax(keluaran_satu.logits, dim=-1)
                    prediksi_satu = torch.argmax(probabilitas_satu, dim=-1).item()
                    skor_kepastian_satu = probabilitas_satu[0][prediksi_satu].item() * 100

                status_kelas = "Normal" if prediksi_satu == 0 else "Distorsi Kognitif Terdeteksi"
                status_gaya = "status-normal" if prediksi_satu == 0 else "status-distorsi"

                if prediksi_satu == 0:
                    kolom_hasil_kiri, kolom_hasil_tengah, kolom_hasil_kanan = st.columns([1, 2, 1])
                    with kolom_hasil_tengah:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #9CA3AF !important; font-size: 1.2rem; margin-bottom: 1.5rem;'>Hasil Penyaringan Tahap 1</h3>
                            <div style='margin-bottom: 2rem;'>
                                <span class='{status_gaya}'>{status_kelas}</span>
                            </div>
                            <p style='margin: 0; font-size: 0.9rem; color: #9CA3AF !important;'>Tingkat Kepastian Probabilistik</p>
                            <h2 style='margin: 0 0 1.5rem 0; color: #F9FAFB !important;'>{skor_kepastian_satu:.2f}%</h2>
                            <p style='font-size: 0.9rem; color: #D1D5DB !important;'>Teks tidak memenuhi ambang batas indikasi distorsi kognitif. Inferensi tahap kedua dihentikan.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with torch.no_grad():
                        keluaran_dua = model_dua(**input_tensor)
                        probabilitas_dua = torch.nn.functional.softmax(keluaran_dua.logits, dim=-1)
                        prediksi_dua = torch.argmax(probabilitas_dua, dim=-1).item()
                        skor_kepastian_dua = probabilitas_dua[0][prediksi_dua].item() * 100
                    
                    kategori_spesifik = daftar_kategori_distorsi[prediksi_dua]
                    col_satu, col_dua = st.columns(2)
                    
                    with col_satu:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #9CA3AF !important; font-size: 1.2rem; margin-bottom: 1.5rem;'>Level 1: Deteksi Biner (HuggingFace)</h3>
                            <div style='margin-bottom: 2rem;'>
                                <span class='{status_gaya}'>{status_kelas}</span>
                            </div>
                            <p style='margin: 0; font-size: 0.9rem; color: #9CA3AF !important;'>Tingkat Kepastian</p>
                            <h2 style='margin: 0; color: #F9FAFB !important;'>{skor_kepastian_satu:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_dua:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h3 style='color: #9CA3AF !important; font-size: 1.2rem; margin-bottom: 1.5rem;'>Level 2: Klasifikasi Spesifik (LoRA Adapter)</h3>
                            <div style='margin-bottom: 2rem;'>
                                <span class='status-distorsi'>{kategori_spesifik}</span>
                            </div>
                            <p style='margin: 0; font-size: 0.9rem; color: #9CA3AF !important;'>Tingkat Kepastian</p>
                            <h2 style='margin: 0; color: #F9FAFB !important;'>{skor_kepastian_dua:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)

with tab_model_1:
    st.markdown("<div style='padding: 1rem 3rem;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Model 1: Binary Classification Gate (RoBERTa)</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Tahap 0: Pra-pemrosesan Data (Preprocessing)
    Pada tahap awal, seluruh dataset mentah teks berbahasa Indonesia diproses menggunakan algoritma NLP standar. Langkah-langkah ini mencakup:
    *   **Pembersihan Teks (Cleansing):** Menghapus karakter khusus, angka berlebih, dan *whitespace*.
    *   **Labeling Ulang:** Mengonversi 11 jenis kelas distorsi menjadi satu kelas sentral (Label `1` = *Distorsi Kognitif*), dan mempertahankan kelas 'No Distortion' (Label `0` = *Normal*).
    *   **Tokenisasi:** Menggunakan tokenizer dari `w11wo/indonesian-roberta-base-sentiment-classifier` dengan batas maksimal *sequence length* 512 token untuk mengubah teks menjadi *input IDs* dan *attention masks*.
    
    ### Tahap 1: Arsitektur Model (Full Fine-Tuning)
    Model Level 1 didesain sebagai "gerbang penyaring" pertama. Kami memanfaatkan arsitektur **RoBERTa (Robustly Optimized BERT Approach)** yang telah di-*pretrain* pada korpus bahasa Indonesia. Pada tahap ini, seluruh bobot jaringan (*full fine-tuning*) disesuaikan untuk tugas klasifikasi biner. Lapisan klasifikasi akhir diubah konfigurasinya untuk mendeteksi `num_labels=2`.
    
    ### Tahap 2: Hyperparameter Tuning
    Penyesuaian konfigurasi pembelajaran (*hyperparameters*) dilakukan secara komprehensif untuk memastikan konvergensi model tanpa terjadi *vanishing gradient* atau *mode collapse*. Berikut adalah rincian metrik pelatihan:
    
    | Parameter | Nilai Konfigurasi | Deskripsi |
    | :--- | :--- | :--- |
    | **Base Model** | `indonesian-roberta-base` | Model dasar dengan pemahaman tata bahasa Indonesia. |
    | **Learning Rate** | `2e-5` | Nilai peluruhan yang sangat kecil agar tidak merusak bobot *pretrained*. |
    | **Batch Size** | `16` (Train) / `32` (Eval) | Mengoptimalkan pemanfaatan VRAM pada GPU. |
    | **Epochs** | `5` | Jumlah iterasi penuh pada seluruh dataset pelatihan. |
    | **Weight Decay** | `0.01` | Mencegah *overfitting* yang ekstrem. |
    """)
    
    st.markdown("<br>### Tahap 3: Evaluasi & Metrik Performa<br>", unsafe_allow_html=True)
    col_m1_1, col_m1_2 = st.columns(2)
    with col_m1_1:
        tampilkan_gambar_visualisasi("grafik_model1.png", "Grafik Pelatihan & Loss (Model 1)")
    with col_m1_2:
        tampilkan_gambar_visualisasi("metrik_eval_model1.png", "Metrik Evaluasi Keseluruhan (Model 1)")
        
    st.markdown("<br>", unsafe_allow_html=True)
    tampilkan_gambar_visualisasi("confusion_model1.png", "Confusion Matrix Klasifikasi Biner")
    
    st.markdown("</div>", unsafe_allow_html=True)


with tab_model_2:
    st.markdown("<div style='padding: 1rem 3rem;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Model 2: Multiclass Classification (LoRA Adapter)</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Tahap 0: Segmentasi & Re-Tokenisasi Data
    Model Level 2 dikhususkan hanya untuk penderita distorsi. Oleh karena itu, data yang masuk ke tahap ini secara eksklusif hanyalah dataset dengan label asli distorsi kognitif (11 kelas). Data *No Distortion* dibuang secara terprogram. Proses tokenisasi menggunakan panjang sekuens yang sama (512 token) untuk menjaga integritas tensor.

    ### Tahap 1: Arsitektur Model (Parameter-Efficient Fine-Tuning)
    Karena 11 kelas klasifikasi membutuhkan sensitivitas leksikal yang lebih tinggi, *Full Fine-Tuning* berisiko memicu *catastrophic forgetting*. Solusi yang diterapkan adalah menggunakan **Low-Rank Adaptation (LoRA)**. 
    Kami membekukan ( *freeze* ) seluruh matriks parameter asli RoBERTa (berukuran >500MB) dan hanya menyuntikkan matriks pembaruan kecil (*adapters*) ke dalam lapisan *Query* dan *Value* pada modul atensi. Ini menghasilkan ukuran model tambahan hanya ~4.7MB.

    ### Tahap 2: Konfigurasi LoRA & Hyperparameter
    Parameter di bawah ini adalah kunci utama untuk mencapai ekuilibrium performa antara komputasi yang efisien dengan tingkat akurasi diagnostik.
    
    | Parameter | Nilai Konfigurasi | Deskripsi |
    | :--- | :--- | :--- |
    | **Rank (r)** | `16` | Dimensi matriks dekomposisi pada LoRA. |
    | **LoRA Alpha** | `32` | Faktor skala penyelarasan *adapter* (*scaling factor*). |
    | **Target Modules** | `["query", "value"]` | Lapisan *attention* spesifik tempat matriks diinjeksi. |
    | **LoRA Dropout** | `0.1` | Probabilitas menonaktifkan neuron untuk mencegah *overfitting*. |
    | **Learning Rate** | `5e-4` | Karena parameter sedikit, LR ditingkatkan 25x lipat dari Model 1. |
    | **Epochs** | `10` | Proses adaptasi LoRA butuh iterasi lebih banyak untuk konvergen. |
    """)

    st.markdown("<br>### Tahap 3: Evaluasi & Hasil Akhir (11 Kelas Kognitif)<br>", unsafe_allow_html=True)
    col_m2_1, col_m2_2 = st.columns(2)
    with col_m2_1:
        tampilkan_gambar_visualisasi("grafik_model2.png", "Grafik Pelatihan & Loss LoRA (Model 2)")
    with col_m2_2:
        tampilkan_gambar_visualisasi("metrik_eval_model2.png", "Metrik Evaluasi 11 Kelas Distorsi")

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm1, col_cm2, col_cm3 = st.columns([1, 4, 1])
    with col_cm2:
        tampilkan_gambar_visualisasi("confusion_model2.png", "Confusion Matrix Ekstensif untuk 11 Kelas")
    
    st.markdown("</div>", unsafe_allow_html=True)
