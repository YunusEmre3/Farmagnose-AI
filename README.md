# Farmagnose-AI - Yapay Zeka Destekli Bitki Hastalık Teşhis Platformu 

Bu proje, TİM-İnovaTİM İnovasyon Yarışması için geliştirilmiş, çiftçilere ve ziraat mühendislerine yönelik iki aşamalı bir yapay zeka destekli bitki hastalık teşhis platformudur.

## 🚀 Projenin Amacı

Projemiz, cep telefonuyla çekilen yaprak fotoğraflarını analiz ederek, öncelikle görüntüde ilgili bitkinin varlığını **nesne tespiti (YOLO)** ile doğrular. Ardından, tespit edilen yaprak üzerinde **semantik segmentasyon (DeepLabV3+)** kullanarak hastalıklı bölgeleri piksel seviyesinde belirler. Bu sayede kullanıcılara sadece teşhis sunmakla kalmaz, aynı zamanda hastalığın yaygınlığı hakkında kantitatif veriler de sağlar.

## ✨ Özellikler

- **İki Aşamalı Analiz:** Hızlı YOLO tespiti ve detaylı DeepLabV3+ segmentasyonu.
- **Esnek Mod:** Sadece "Domates" veya "Tüm Bitkiler" için analiz yapabilme seçeneği.
- **Çoklu Dil Desteği:** Türkçe ve İngilizce arayüz.
- **Etkileşimli Arayüz:** Streamlit ile geliştirilmiş, çoklu resim yükleme ve kamera desteği sunan kullanıcı dostu arayüz.
- **Ekosistem Modülleri:** Topluluk Forumu, Uzman Desteği ve Malzeme Siparişi gibi mock-up sayfalar.

## 🛠️ Kurulum ve Çalıştırma

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/YunusEmre3/Farmagnose-AI.git
    cd Farmagnose-AI
    ```
2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model Dosyalarını İndirin:**
    - `models/best_model.pth` segmentasyon modelini bu klasöre yerleştirin.

4.  **Uygulamayı Başlatın:**
    ```bash
    streamlit run app.py
    ```
