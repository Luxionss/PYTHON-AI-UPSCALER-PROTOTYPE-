import sys
import os
import subprocess
import shutil
import cv2
import numpy as np
import torch
from datetime import datetime

try:
    import torchvision.transforms.functional as tf
    sys.modules['torchvision.transforms.functional_tensor'] = tf
except ImportError:
    pass

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    HAS_REAL_ESRGAN = True
    IMPORT_ERROR_MSG = ""
except ImportError as e:
    HAS_REAL_ESRGAN = False
    IMPORT_ERROR_MSG = str(e)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QProgressBar, QLabel, QLineEdit, 
                             QFileDialog, QTextEdit, QHBoxLayout, QComboBox, QSlider, QSplitter)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

import urllib.request

# --- DİL DESTEĞİ SÖZLÜĞÜ ---
TRANSLATIONS = {
    'TR': {
        'title': "AI UPSCALER PROTOTYPE V2 (BY LUXİONS)",
        'path_ph': "Dosya Yolu (Video veya Görsel)",
        'out_ph': "Kaydedilecek Yer",
        'btn_sel': "📂 KAYNAK SEÇ",
        'btn_out_sel': "💾 HEDEF SEÇ",
        'scale_lbl': "Çözünürlük Çarpanı:",
        'sharpen_lbl': "Ekstra Keskinlik",
        'contrast_lbl': "Kontrast",
        'denoise_lbl': "Yumuşatma (Denoise)",
        'btn_start': "GOJO İLE TAKAS OLMAZ ABi",
        'btn_gpu': "GPU Durumunu Kontrol Et",
        'terminal': "> TERMİNAL ÇIKTISI",
        'log_start': "--- İŞLEM BAŞLADI ---",
        'log_model_missing': "Model bulunamadı, indiriliyor...",
        'btn_clear': "TEMİZLE",
        'log_model_missing': "Model dosyası bulunamadı, indiriliyor...",
        'log_gpu_active': "CUDA AKTİF",
        'log_cpu_mode': "CPU MODU (Yavaş)",
        'log_success': "BAŞARILI! Şuraya kaydedildi:",
        'gpu_info_title': "--- GPU DURUMU KONTROL EDİLİYOR ---",
        'gpu_active_yes': "CUDA AKTİF: Evet",
        'gpu_active_no': "CUDA AKTİF: Hayır",
        'gpu_used': "Kullanılan GPU",
        'gpu_ready': "BAŞARILI: GPU kullanıma hazır.",
        'gpu_err': "HATA: GPU bulunamadı veya CUDA yüklü değil.",
        'gpu_diag': "TEŞHİS: Yüklü PyTorch sürümü sadece CPU desteği içeriyor.",
        'gpu_fix': "CUDA sürücülerini ve kurulumu kontrol edin.",
        'log_ffmpeg_err': "HATA: FFmpeg bulunamadı!",
        'log_engine_ready': "Gelişmiş AI Motoru Hazır"
    },
    'EN': {
        'title': "AI UPSCALER PROTOTYPE V2 (BY LUXIONS)",
        'path_ph': "File Path (Video or Image)",
        'out_ph': "Save Location",
        'btn_sel': "📂 SELECT SOURCE",
        'btn_out_sel': "💾 SELECT TARGET",
        'scale_lbl': "Resolution Multiplier:",
        'sharpen_lbl': "Extra Sharpening",
        'contrast_lbl': "Contrast",
        'denoise_lbl': "Smoothing (Denoise)",
        'btn_start': "START UPSCALING",
        'btn_gpu': "Check GPU Status",
        'terminal': "> TERMINAL OUTPUT",
        'log_start': "--- PROCESS STARTED ---",
        'btn_clear': "CLEAR",
        'log_model_missing': "Model not found, downloading...",
        'log_gpu_active': "CUDA ACTIVE",
        'log_cpu_mode': "CPU MODE (Slow)",
        'log_success': "SUCCESS! Saved to:",
        'gpu_info_title': "--- CHECKING GPU STATUS ---",
        'gpu_active_yes': "CUDA ACTIVE: Yes",
        'gpu_active_no': "CUDA ACTIVE: No",
        'gpu_used': "Used GPU",
        'gpu_ready': "SUCCESS: GPU is ready for use.",
        'gpu_err': "ERROR: GPU not found or CUDA not installed.",
        'gpu_diag': "DIAGNOSIS: Installed PyTorch version only supports CPU.",
        'gpu_fix': "Check CUDA drivers and installation.",
        'log_ffmpeg_err': "ERROR: FFmpeg not found!",
        'log_engine_ready': "Advanced AI Engine Ready"
    }
}


class UpscaleWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)

    def __init__(self, input_path, output_path, use_gpu, is_image_processing, target_scale, sharpen_val, contrast_val, denoise_val, lang='TR'):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.is_image_processing = is_image_processing
        self.target_scale = target_scale
        self.denoise_val = denoise_val 
        self.sharpen_val = sharpen_val
        self.contrast_val = contrast_val
        self.lang = lang
        self.t = TRANSLATIONS[lang]

    def apply_post_processing(self, img):
        if self.contrast_val > 1.0:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clip_limit = min(2.0, self.contrast_val)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        if self.sharpen_val > 0.0:
            gaussian = cv2.GaussianBlur(img, (0, 0), 3)
            weight = self.sharpen_val * 0.5 
            img = cv2.addWeighted(img, 1.0 + weight, gaussian, -weight, 0)
            
        return cv2.convertScaleAbs(img, alpha=self.contrast_val, beta=0)

    def download_model_if_needed(self, model_path):
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        if not os.path.exists(model_path):
            self.log.emit("Real-ESRGAN Modeli bulunamadı. İnternetten indiriliyor (yaklaşık 67MB)...")
            try:
                urllib.request.urlretrieve(url, model_path)
                self.log.emit("Model başarıyla indirildi!")
            except Exception as e:
                self.log.emit(f"Model indirme hatası: {str(e)}")
                return False
        return True

    def run(self):
        try:
            self.log.emit(self.t['log_start'])
            working_dir = os.path.dirname(os.path.abspath(__file__))

            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                local_ffmpeg = os.path.join(working_dir, "ffmpeg.exe")
                if os.path.exists(local_ffmpeg):
                    ffmpeg_bin = local_ffmpeg
                elif not self.is_image_processing:
                    self.log.emit(self.t['log_ffmpeg_err'])
                    self.finished.emit(False)
                    return

            temp_file = os.path.join(working_dir, "temp_process.mp4")
            model_path = os.path.join(working_dir, "RealESRGAN_x4plus.pth")
            
            if self.use_gpu and torch.cuda.is_available():
                torch_ver = torch.__version__
                device = torch.device('cuda')
                device_status = f"CUDA AKTİF ({torch.cuda.get_device_name(0)}) [Sürüm: {torch_ver}]"
                half_precision = True 
            elif self.use_gpu:
                torch_ver = torch.__version__
                device = torch.device('cpu')
                device_status = f"HATA: GPU Seçildi ama CUDA bulunamadı! (Yüklü Torch: {torch_ver})"
                if "+cpu" in torch_ver:
                    self.log.emit("TEŞHİS: Bilgisayarında şu an '+cpu' sürümü yüklü. Bu sürüm GPU kullanamaz.")
                
                self.log.emit("BU DURUMU ÇÖZMEK İÇİN ŞUNLARI YAP:")
                self.log.emit("1. pip uninstall torch torchvision torchaudio -y")
                self.log.emit("2. pip cache purge  <-- (BU ÖNEMLİ, eski dosyaları siler)")
                self.log.emit("3. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
                half_precision = False
            else:
                device = torch.device('cpu')
                device_status = "CPU/İŞLEMCİ (İşlem yavaş olabilir)"
                half_precision = False

            self.log.emit(f"Kullanılan Mod: {device_status}")

            upsampler = None
            if HAS_REAL_ESRGAN:
                if self.download_model_if_needed(model_path):
                    self.log.emit("Real-ESRGAN ağı belleğe yükleniyor...")
                    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                    upsampler = RealESRGANer(
                        scale=4,
                        model_path=model_path,
                        model=model,
                        tile=128, 
                        tile_pad=10,
                        pre_pad=0,
                        half=half_precision,
                        device=device
                    )
                    self.log.emit(self.t['log_engine_ready'])
            else:
                self.log.emit("--- AI MOTORU EKSİK (KURULUM HATASI) ---")
                if IMPORT_ERROR_MSG:
                    self.log.emit(f"Sistem Hatası: {IMPORT_ERROR_MSG}")
                self.log.emit("AI EKSİK KİNGO SIRAYLA YAZCAN ŞUNLARI:")
                self.log.emit("1. pip uninstall basicsr realesrgan -y")
                self.log.emit("2. pip install setuptools wheel")
                self.log.emit("3. pip install basicsr --no-deps")
                self.log.emit("4. pip install realesrgan")
                self.log.emit("Eğer hala hata varsa 3. adımı '--no-build-isolation' ile dene.")
                
            scale_factor = self.target_scale

            if self.is_image_processing:
                img = cv2.imread(self.input_path)
                if img is None:
                    self.log.emit("HATA: Görsel açılamadı!")
                    self.finished.emit(False)
                    return
                
                if self.denoise_val > 0:
                    img = cv2.fastNlMeansDenoisingColored(img, None, self.denoise_val, self.denoise_val, 7, 15)

                if self.denoise_val > 5:
                    k_size = 3 if self.denoise_val < 10 else 5
                    img = cv2.medianBlur(img, k_size)

                orig_h, orig_w = img.shape[:2]
                self.log.emit(f"Görsel yeniden inşa ediliyor: {orig_w}x{orig_h} -> {orig_w*scale_factor}x{orig_h*scale_factor}")

                if upsampler:
                    try:
                        upscaled_img, _ = upsampler.enhance(img, outscale=scale_factor)
                    except Exception as e:
                        self.log.emit(f"AI İşleme Hatası (VRAM yetersiz olabilir): {str(e)}")
                        self.finished.emit(False)
                        return
                else:
                    w = (orig_w * scale_factor) // 2 * 2
                    h = (orig_h * scale_factor) // 2 * 2
                    upscaled_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
                
                upscaled_img = self.apply_post_processing(upscaled_img)
                cv2.imwrite(self.output_path, upscaled_img)
                
                self.log.emit(f"{self.t['log_success']}\n{self.output_path}")
                self.progress.emit(100) 
                self.finished.emit(True)

            else:
                cap = cv2.VideoCapture(self.input_path)
                if not cap.isOpened():
                    self.finished.emit(False)
                    return
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = (orig_w * scale_factor) // 2 * 2
                h = (orig_h * scale_factor) // 2 * 2
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(temp_file, fourcc, fps, (w, h))

                self.log.emit(f"Kareler tek tek AI ile çiziliyor (Toplam: {total_frames})...")
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    if self.denoise_val > 0:
                        frame = cv2.fastNlMeansDenoisingColored(frame, None, self.denoise_val, self.denoise_val, 7, 15)

                    if upsampler:
                        frame, _ = upsampler.enhance(frame, outscale=scale_factor)
                        if frame.shape[1] != w or frame.shape[0] != h:
                            frame = cv2.resize(frame, (w, h))
                    else:
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)

                    frame = self.apply_post_processing(frame)
                    writer.write(frame)
                    
                    frame_count += 1
                    self.progress.emit(int((frame_count / total_frames) * 100))
                    if frame_count % 10 == 0:
                        self.log.emit(f"İşleniyor: {frame_count}/{total_frames}")
                
                writer.release()
                cap.release()

                self.log.emit("Ses birleştiriliyor (FFmpeg)...")
                try:
                    cmd = [ffmpeg_bin, '-y', '-i', temp_file, '-i', self.input_path, 
                           '-map', '0:v:0', '-map', '1:a?', '-c:v', 'libx264', 
                           '-preset', 'medium', '-crf', '18', '-c:a', 'copy', 
                           '-shortest', '-movflags', '+faststart', self.output_path]
                    
                    # Komutu string olarak değil liste olarak güvenli çalıştır
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    self.log.emit("BAŞARILI! Video hazır.")
                    os.remove(temp_file)
                    self.finished.emit(True)
                except subprocess.CalledProcessError as e:
                    self.log.emit(f"FFMPEG Hatası: {e.stderr or e.stdout}")
                    self.finished.emit(False)
                except Exception as e:
                    self.log.emit(f"Ses Birleştirme Hatası: {str(e)}")
                    self.finished.emit(False)
        except Exception as e:
            self.log.emit(f"KRİTİK HATA: {str(e)}")
            self.finished.emit(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = 'TR'
        self.resize(650, 550)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0b10; }
            QLabel { color: #00f2ff; font-family: 'Segoe UI', sans-serif; font-size: 12px; font-weight: bold; }
            QLineEdit { background-color: #161b22; border: 1px solid #00f2ff; border-radius: 2px; color: #ffffff; padding: 8px; }
            QPushButton { background-color: #1a1a2e; color: #00f2ff; border: 1px solid #00f2ff; padding: 10px; font-weight: bold; }
            QPushButton:hover { background-color: #00f2ff; color: #000000; }
            QProgressBar { border: 1px solid #ff00ff; text-align: center; color: white; font-weight: bold; }
            QProgressBar::chunk { background-color: #ff00ff; }
            QComboBox { background-color: #161b22; color: #ffffff; border: 1px solid #00f2ff; padding: 5px; }
            QScrollBar:vertical { border: none; background: #0a0b10; width: 10px; }
            QScrollBar::handle:vertical { background: #1f2833; border-radius: 5px; }
        """)

        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        splitter = QSplitter(Qt.Orientation.Vertical)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["TR", "EN"])
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        top_layout.addWidget(self.lang_combo, 0, Qt.AlignmentFlag.AlignRight)

        self.path_edit = QLineEdit()
        self.btn_sel = QPushButton()

        self.out_path_edit = QLineEdit()
        self.btn_out_sel = QPushButton()

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "4x"])
        self.scale_combo.setCurrentText("4x")

        self.scale_label = QLabel()
        self.sharpen_label = QLabel()
        self.sharpen_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpen_slider.setRange(0, 15) 
        self.sharpen_slider.setValue(0)
        self.sharpen_slider.valueChanged.connect(self.update_labels)

        self.contrast_label = QLabel()
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 20) 
        self.contrast_slider.setValue(10)
        self.contrast_slider.valueChanged.connect(self.update_labels)

        self.denoise_label = QLabel()
        self.denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self.denoise_slider.setRange(0, 15) 
        self.denoise_slider.setValue(0) 
        self.denoise_slider.valueChanged.connect(self.update_labels)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.btn_check_gpu_status = QPushButton() 
        self.btn_check_gpu_status = QPushButton()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #000; color: #0f0; font-family: Consolas;")
        
        self.btn_start = QPushButton()
        self.btn_start.setStyleSheet("background-color: #6200ea; color: white; font-size: 14px; border: 2px solid #ff00ff;")
        
        h1 = QHBoxLayout()
        h1.addWidget(self.path_edit)
        h1.addWidget(self.btn_sel)
        top_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(self.out_path_edit)
        h2.addWidget(self.btn_out_sel)
        top_layout.addLayout(h2)

        top_layout.addWidget(self.scale_label)
        top_layout.addWidget(self.scale_combo)
        top_layout.addWidget(self.denoise_label)
        top_layout.addWidget(self.denoise_slider)
        top_layout.addWidget(self.sharpen_label)
        top_layout.addWidget(self.sharpen_slider)
        top_layout.addWidget(self.contrast_label)
        top_layout.addWidget(self.contrast_slider)
        top_layout.addWidget(self.btn_start)
        top_layout.addWidget(self.progress_bar) # Yeni buton eklendi
        top_layout.addWidget(self.btn_check_gpu_status)

        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        
        terminal_header = QHBoxLayout()
        self.label_terminal = QLabel()
        bottom_layout.addWidget(self.label_terminal)
        self.btn_clear_log = QPushButton()
        self.btn_clear_log.setFixedWidth(80)
        self.btn_clear_log.setStyleSheet("font-size: 10px; padding: 2px;")
        terminal_header.addWidget(self.label_terminal)
        terminal_header.addStretch()
        terminal_header.addWidget(self.btn_clear_log)
        
        bottom_layout.addLayout(terminal_header)
        bottom_layout.addWidget(self.log_box)

        splitter.addWidget(top_container)
        splitter.addWidget(bottom_container)
        layout.addWidget(splitter)
        
        self.btn_sel.clicked.connect(self.select_file)
        self.btn_out_sel.clicked.connect(self.select_output_file)
        self.btn_check_gpu_status.clicked.connect(self.check_gpu_status) # Yeni butonun sinyali bağlandı
        self.btn_clear_log.clicked.connect(self.log_box.clear)
        self.btn_start.clicked.connect(self.run_task)
        
        self.retranslate_ui()

    def add_log(self, message):
        now = datetime.now().strftime("%H:%M:%S")
        msg_upper = message.upper()
        color = "#00ff00" # Varsayılan Yeşil
        
        # İçeriğe göre renk belirleme
        if any(x in msg_upper for x in ["HATA", "ERROR", "KRİTİK", "FAIL", "EKSİK"]):
            color = "#ff4444" # Kırmızı
        elif any(x in msg_upper for x in ["BAŞARILI", "SUCCESS", "HAZIR", "READY", "BİTTİ"]):
            color = "#55ff55" # Parlak Yeşil
        elif any(x in msg_upper for x in ["İŞLENİYOR", "PROCESSING", "İNDİRİLİYOR"]):
            color = "#00d2ff" # Mavi
            
        formatted_text = f"<span style='color:#666;'>[{now}]</span> <span style='color:{color};'>{message}</span>"
        self.log_box.append(formatted_text)
        self.log_box.ensureCursorVisible() # Otomatik kaydır

    def change_language(self, index):
        self.current_lang = self.lang_combo.currentText()
        self.retranslate_ui()

    def update_labels(self):
        t = TRANSLATIONS[self.current_lang]
        self.sharpen_label.setText(f"{t['sharpen_lbl']}: {self.sharpen_slider.value()/10:.1f}")
        self.contrast_label.setText(f"{t['contrast_lbl']}: {self.contrast_slider.value()/10:.1f}")
        self.denoise_label.setText(f"{t['denoise_lbl']}: {self.denoise_slider.value()}")

    def retranslate_ui(self):
        t = TRANSLATIONS[self.current_lang]
        self.setWindowTitle(t['title'])
        self.path_edit.setPlaceholderText(t['path_ph'])
        self.out_path_edit.setPlaceholderText(t['out_ph'])
        self.btn_sel.setText(t['btn_sel'])
        self.btn_out_sel.setText(t['btn_out_sel'])
        self.scale_label.setText(t['scale_lbl'])
        self.btn_start.setText(t['btn_start'])
        self.btn_check_gpu_status.setText(t['btn_gpu'])
        self.label_terminal.setText(t['terminal'])
        self.btn_clear_log.setText(t['btn_clear'])
        self.update_labels()

    def select_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "", "Medya (*.mp4 *.avi *.png *.jpg *.jpeg)")
        if f: 
            self.path_edit.setText(f)
            self.out_path_edit.setText(os.path.join(os.path.dirname(f), f"PRO_4k_{os.path.basename(f)}"))

    def select_output_file(self):
        f, _ = QFileDialog.getSaveFileName(self, "Kayıt Yeri", "", "Medya (*.mp4 *.png *.jpg)")
        if f: self.out_path_edit.setText(f)

    def run_task(self):
        path = self.path_edit.text()
        output = self.out_path_edit.text()
        if not path or not output: return
        
        scale_text = self.scale_combo.currentText()
        target_scale = int(scale_text.replace("x", ""))
        is_image = os.path.splitext(path)[1].lower() in ['.png', '.jpg', '.jpeg']

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.worker = UpscaleWorker(path, output, True, is_image, target_scale, 
                                    self.sharpen_slider.value()/10.0, 
                                    self.contrast_slider.value()/10.0, 
                                    self.denoise_slider.value(), lang=self.current_lang)
        self.worker.log.connect(self.add_log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(lambda s: self.progress_bar.setValue(100 if s else 0))
        self.worker.start()
        
    def check_gpu_status(self):
        """
        CUDA ve GPU kullanım durumunu kontrol eder ve log_box'a yazar.
        """
        t = TRANSLATIONS[self.current_lang]
        self.add_log(t['gpu_info_title'])
        if torch.cuda.is_available():
            torch_ver = torch.__version__
            device_name = torch.cuda.get_device_name(0)
            self.add_log(t['gpu_active_yes'])
            self.add_log(f"{t['gpu_used']}: {device_name}")
            self.add_log(f"PyTorch: {torch_ver}")
            self.add_log(t['gpu_ready'])
        else:
            torch_ver = torch.__version__
            self.add_log(t['gpu_active_no'])
            self.add_log(f"PyTorch: {torch_ver}")
            self.add_log(t['gpu_err'])
            if "+cpu" in torch_ver:
                self.add_log(t['gpu_diag'])
                self.add_log("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            else:
                self.add_log(t['gpu_fix'])
        self.add_log("----------------------------------")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())