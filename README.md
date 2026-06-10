# AI UPSCALER PROTOTYPE V2

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-purple.svg)
![AI](https://img.shields.io/badge/AI-Real--ESRGAN-orange.svg)

**TR:** Görselleri ve videoları yapay zeka (Real-ESRGAN) kullanarak 2x veya 4x çözünürlüğe çıkaran, Python ile geliştirilmiş bir araçtır.

**EN:** A tool developed with Python that upscales images and videos to 2x or 4x resolution using artificial intelligence (Real-ESRGAN).

## Özellikler / Features
- **AI Upscaling:** Real-ESRGAN motoru
- **Post-Processing:** Keskinlik (Sharpen), Kontrast ve Denoise ayarları.
- **Dual Language:** Türkçe ve İngilizce arayüz desteği.
- **GPU/CPU Support:** CUDA desteği
- **Auto-Model Download:** Model dosyası eksikse otomatik olarak indiriyor.

## Kurulum / Installation

1. Projeyi klonlayın / Clone the project:
   ```bash
   git clone https://github.com/Luxionss/PYTHON-AI-UPSCALER-PROTOTYPE-.git
   ```
2. Kütüphaneleri kurun / Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. **FFmpeg:** Video işlemleri için bilgisayarınızda FFmpeg kurulu olmalıdır.

## Sorun Giderme / Troubleshooting
Eğer `basicsr` hatası alırsanız, şu komutları sırayla çalıştırın: if you have `basicsr` error, run the following commands one by one:
```bash
pip uninstall basicsr realesrgan -y
pip install setuptools wheel
pip install basicsr --no-deps
pip install realesrgan
```

## Author
**Luxions**

---
*Gojo ile takas olurmuş abi*