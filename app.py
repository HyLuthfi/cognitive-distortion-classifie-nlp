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
    path_visual = os.path.join(lokasi_direktori, "assets", nama_file)
    
    if os.path.exists(path_visual):
        gambar = Image.open(path_visual)
        st.image(gambar, caption=deskripsi, use_container_width=True)
    else:
        st.info(f"Aset visual {nama_file} belum tersedia di direktori assets.")

st.markdown("<h1 style='text-align: center; margin-top: 1rem;'>Dasbor Analisis Forensik Kognitif</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #9CA3AF !important; margin-bottom: 2rem;'>Sistem Otomatisasi Deteksi Distorsi Kognitif Menggunakan RoBERTa dan LoRA Adapter</p>", unsafe_allow_html=True)

tab_analisis, tab_metodologi, tab_visualisasi = st.tabs([
    "Inspeksi Kognitif", 
    "Metodologi Sistem", 
    "Analisis Matriks"
])

with tab_analisis:
    st.markdown("<br>", unsafe_allow_html=True)
    
    kolom_kiri, kolom_tengah, kolom_kanan = st.columns([1, 2, 1])

    with kolom_tengah:
        st.markdown("<div class='info-card'><b>Instruksi Operasional:</b> Masukkan teks atau kalimat berbahasa Indonesia yang ingin dianalisis secara psikologis. Sistem akan melakukan penyaringan biner sebelum mengekstraksi jenis distorsi spesifik.</div>", unsafe_allow_html=True)
        
        input_teks = st.text_area("Teks Analisis", height=150, placeholder="Ketik teks di sini...", label_visibility="collapsed")
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
                    
                    try:
                        kategori_spesifik = model_dua.config.id2label.get(prediksi_dua, daftar_kategori_distorsi[prediksi_dua])
                    except Exception:
                        kategori_spesifik = f"Kategori {prediksi_dua}"

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

with tab_metodologi:
    st.markdown("<div style='padding: 1rem 3rem;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Metodologi & Arsitektur Cloud</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col_met1, col_met2 = st.columns(2)
    with col_met1:
        st.markdown("### Arsitektur Berjenjang")
        st.markdown("""
        Sistem ini menggunakan pendekatan *Two-Stage Inference Pipeline* untuk mengoptimalkan komputasi dan meminimalisir kesalahan klasifikasi (False Positives):
        
        1. **Level 1 (Binary Gate)** - Teks disaring oleh model *Full Fine-Tuned* RoBERTa. Model ini dilatih secara khusus hanya untuk membedakan antara bahasa yang normal dengan bahasa yang mengandung distorsi psikologis.
        2. **Level 2 (Multiclass Routing)** - Apabila Gerbang Level 1 mendeteksi adanya distorsi, data akan diteruskan ke model sekunder yang telah diinjeksi dengan *Low-Rank Adaptation* (LoRA). Model ini memetakan teks ke dalam 11 kelas psikologi spesifik.
        """)
    
    with col_met2:
        st.markdown("### Optimalisasi Memori dengan PEFT")
        st.markdown("""
        Penerapan *Parameter-Efficient Fine-Tuning* (PEFT) memungkinkan arsitektur ini berjalan pada lingkungan server yang sangat terbatas:
        
        *   **HuggingFace Cloud**: Pembobotan utama untuk Level 1 sebesar 500 MB tidak disimpan di *repository* lokal, melainkan dipanggil secara asinkron dari *cloud*.
        *   **LoRA Micro-Adapter**: Untuk pengenalan 11 kelas yang kompleks, sistem tidak memerlukan 500 MB tambahan. Sistem hanya menggunakan injeksi matriks *query* dan *value* berukuran 4.7 MB, yang menghemat ruang hingga 99% tanpa penurunan metrik performa.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_visualisasi:
    st.markdown("<div style='padding: 1rem 3rem;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Analisis Matriks Validasi</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    Evaluasi model dilakukan secara independen terhadap himpunan data uji. Visualisasi *Confusion Matrix* di bawah ini merepresentasikan kemampuan model dalam mendiagnosis distribusi kelas minoritas.
    """)
    st.markdown("<br>", unsafe_allow_html=True)
    
    tampilkan_gambar_visualisasi("Confusion_Matrix_Level2.png", "Matriks Kebingungan Klasifikasi 11 Kelas Kognitif")
    
    st.markdown("</div>", unsafe_allow_html=True)
