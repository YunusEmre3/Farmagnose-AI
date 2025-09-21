import streamlit as st
import torch
import cv2
import numpy as np
import logging
from PIL import Image
import datetime

# Model-specific imports
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralyticsplus import YOLO

# --- DİL ÇEVİRİ SÖZLÜĞÜ ---
translations = {
    'tr': {
        'page_title': "Farmagnose AI | Bitki Hastalık Analizi",
        'lang_select': "Dil",
        'nav_analysis': "🌿 Yaprak Analizi",
        'nav_history': "🗂️ Geçmiş Analizlerim",
        'nav_community': "💬 Topluluk Forumu",
        'nav_expert': "🧑‍🔬 Bir Uzmana Sorun",
        'nav_chemicals': "🛒 Malzeme Siparişi",
        'nav_cart': "🛒 Sepetim",
        'navigation_label': "Navigasyon",
        'app_title': "Farmagnose AI",
        'footer_text': "© 2025 Farmagnose AI. Tüm hakları saklıdır.",
        # Analiz Sayfası
        'analysis_header': "🌿 İki Aşamalı Yaprak Hastalık Analizi",
        'toggle_label': "Sadece Domatesleri Tespit Et",
        'toggle_help': "Bu seçenek aktifken, 2. Aşama (Hastalık Segmentasyonu) sadece bir domates bitkisi tespit edilirse çalışır. Kapalıyken, tespit edilen herhangi bir bitki için çalışır.",
        'analysis_info': "Yapay zeka destekli anlık analiz için bir yaprak resmi yükleyin veya fotoğrafını çekin.",
        'camera_button': "📸 Fotoğraf Çek",
        'upload_button': "📄 Resim Yükle",
        'uploader_prompt': "Bir veya daha fazla resim seçin...", 
        'camera_prompt': "Fotoğraf çekin",
        'analyze_button': " Resmi Analiz Et",
        'expander_text': "🔬 Analiz Sonuçları: ",
        'spinner_stage1': "Aşama 1: {filename} içindeki yapraklar tespit ediliyor...",
        'warning_no_tomato': "Analiz Tamamlandı: Bu resimde domates tespit edilmedi.",
        'warning_no_plant': "Analiz Tamamlandı: Bu resimde herhangi bir bitki tespit edilmedi.",
        'success_tomato': "✅ Domates Yaprağı Tespit Edildi! 2. Aşamaya Geçiliyor: Hastalık Segmentasyonu...",
        'success_plant': "✅ Bitki Yaprağı Tespit Edildi! 2. Aşamaya Geçiliyor: Hastalık Segmentasyonu...",
        'spinner_stage2': "Aşama 2: {filename} hastalıklar için analiz ediliyor...",
        'results_header': "Kapsamlı Analiz Sonuçları",
        'caption_original': "1. Orijinal",
        'caption_detection': "2. Tespit",
        'caption_mask': "3. Hastalık Maskesi",
        'caption_overlay': "4. Bindirme",
        # Geçmiş Analizler Sayfası
        'history_header': "🗂️ Geçmiş Analizlerim",
        'history_info': "Daha önce yaptığınız analizlerin kayıtlarını ve sonuçlarını burada bulabilirsiniz.",
        'history_entry_title': "Analiz Kaydı: ",
        'history_result_summary': "Sonuç Özeti:",
        'history_tomato_detected': "Domates Tespit Edildi",
        'history_disease_detected': "Hastalık Tespit Edildi",
        # Topluluk Forumu
        'community_header': "💬 Topluluk Forumu",
        'community_info': "Diğer çiftçiler ve yetiştiricilerle bağlantı kurun. Zorluklarınızı ve çözümlerinizi paylaşın.",
        'view_post_button': "Gönderiyi Görüntüle",
        'post_modal_title': "Forum Başlığı: ",
        # Uzman Sayfası
        'expert_header': "🧑‍🔬 Bir Uzmana Sorun",
        'expert_info': "Sertifikalı ziraat mühendislerinden profesyonel tavsiye alın.",
        'start_call_button': "📞 Anında Arama Başlat",
        'start_chat_button': "💬 Canlı Sohbet Başlat",
        'available_experts_title': "Müsait Uzmanlar",
        'farm_analysis_request': "Çiftliğime Analiz Talep Et",
        'farm_analysis_header': "Yerinde Analiz Talep Formu",
        'farm_analysis_info': "Lütfen bilgilerinizi girin, en kısa sürede sizinle iletişime geçeceğiz.",
        # Malzeme Siparişi
        'chemicals_header': "🛒 Tarım Malzemeleri Sipariş Edin",
        'chemicals_info': "Önerilen fungisit, pestisit ve gübreleri satın alın.",
        'view_product_details': "Ürün Detaylarını Görüntüle",
        'product_details_title': "Ürün Detayları",
        # Ek çeviriler
        'disease_detected_warning': "Analizinizde hastalık belirtileri tespit edildi. Aşağıdaki ürünleri inceleyebilirsiniz:",
        'view_details': "Detayları Gör",
        'add_to_cart': "🛒 Sepete Ekle",
        'added_to_cart': "sepete eklendi!",
        'view_post': "Gönderiyi Gör",
        'helpful_found': "💙 Faydalı!",
        'mark_helpful': "👍 Faydalı Buldum",
        'saved': "💾 Kaydedildi",
        'save': "🔖 Kaydet",
        'send_comment': "📤 Yorum Gönder",
        'go_back': "← Geri Dön",
        'call_expert': "Ara",
        'chat_expert': "Sohbet",
        'expert_available': "Müsait",
        'expert_busy': "Meşgul",
        'call_description': "Aşağıdaki uzmanları arayabilirsiniz:",
        'chat_description': "Aşağıdaki uzmanlarla sohbet başlatabilirsiniz:",
        'online_status': "🟢 Çevrimiçi",
        'form_name': "Ad Soyad *",
        'form_phone': "Telefon *",
        'form_address': "Adres *",
        'form_area': "Arazi Büyüklüğü (dönüm) *",
        'form_crop': "Yetiştirilen Ürün",
        'form_problem': "Sorun Açıklaması",
        'form_date': "Tercih Edilen Tarih",
        'form_submit': "Talep Gönder",
        'form_success': "✅ Talebiniz başarıyla alındı! En kısa sürede sizinle iletişime geçeceğiz.",
        'form_error': "Lütfen zorunlu alanları doldurun.",
        'product_active_ingredient': "Aktif Madde:",
        'product_description': "Açıklama:",
        'product_dosage': "Kullanım Dozu:",
        'product_application': "Uygulama:",
        'product_target_diseases': "Hedef Hastalıklar:",
        'product_target_pests': "Hedef Zararlılar:",
        'product_benefits': "Faydalar:",
        'product_safety': "Güvenlik:",
        'product_quantity': "Adet",
        'product_organic_label': "🌱 Organik Ürün",
        # Location and weather
        'location_izmir': "İzmir, TR",
        'weather_sunny': "Güneşli",
        'month_names': [
            "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
            "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"
        ],
        'day_names': [
            "Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"
        ]
    },
    'en': {
        'page_title': "Farmagnose AI | Plant Disease Analysis",
        'lang_select': "Language",
        'nav_analysis': "🌿 Leaf Analysis",
        'nav_history': "🗂️ My Past Analyses",
        'nav_community': "💬 Community Forum",
        'nav_expert': "🧑‍🔬 Ask an Expert",
        'nav_chemicals': "🛒 Order Supplies",
        'nav_cart': "🛒 My Cart",
        'navigation_label': "Navigation",
        'app_title': "Farmagnose AI",
        'footer_text': "© 2025 Farmagnose AI. All rights reserved.",
        # Analysis Page
        'analysis_header': "🌿 Two-Stage Leaf Disease Analysis",
        'toggle_label': "Detect Tomatoes Only",
        'toggle_help': "When on, Stage 2 (Disease Segmentation) will only run if a tomato plant is detected. When off, it will run for any detected plant.",
        'analysis_info': "Upload a leaf image or use your camera to get an instant AI-powered analysis.",
        'camera_button': "📸 Take a Photo",
        'upload_button': "📄 Upload Image(s)",
        'uploader_prompt': "Choose one or more images...", 
        'camera_prompt': "Take a photo",
        'analyze_button': " Analyze Image(s)",
        'expander_text': "🔬 Analysis Results for: ",
        'spinner_stage1': "Stage 1: Detecting leaves in {filename}...",
        'warning_no_tomato': "Analysis Complete: No tomato was detected in this image.",
        'warning_no_plant': "Analysis Complete: No plant was detected in this image.",
        'success_tomato': "✅ Tomato Leaf Detected! Proceeding to Stage 2: Disease Segmentation...",
        'success_plant': "✅ Plant Leaf Detected! Proceeding to Stage 2: Disease Segmentation...",
        'spinner_stage2': "Stage 2: Analyzing {filename} for diseases...",
        'results_header': "Comprehensive Analysis Results",
        'caption_original': "1. Original",
        'caption_detection': "2. Detection",
        'caption_mask': "3. Disease Mask",
        'caption_overlay': "4. Overlay",
        # History Page
        'history_header': "🗂️ My Past Analyses",
        'history_info': "You can find the records and results of your previous analyses here.",
        'history_entry_title': "Analysis Record: ",
        'history_result_summary': "Result Summary:",
        'history_tomato_detected': "Tomato Detected",
        'history_disease_detected': "Disease Detected",
        # Community Forum
        'community_header': "💬 Community Forum",
        'community_info': "Connect with other farmers and growers. Share challenges and solutions.",
        'view_post_button': "View Post",
        'post_modal_title': "Forum Post: ",
        # Expert Page
        'expert_header': "🧑‍🔬 Ask an Expert",
        'expert_info': "Get professional advice from certified agricultural engineers.",
        'start_call_button': "📞 Start an Instant Call",
        'start_chat_button': "💬 Start a Chat Session",
        'available_experts_title': "Available Experts",
        'farm_analysis_request': "Request On-Site Farm Analysis",
        'farm_analysis_header': "On-Site Analysis Request Form",
        'farm_analysis_info': "Please enter your information, and we will contact you shortly.",
        # Chemicals Page
        'chemicals_header': "🛒 Order Agricultural Supplies",
        'chemicals_info': "Purchase recommended fungicides, pesticides, and fertilizers.",
        'view_product_details': "View Product Details",
        'product_details_title': "Product Details",
        # Additional translations
        'disease_detected_warning': "Disease symptoms detected in your analysis. You can review the products below:",
        'view_details': "View Details",
        'add_to_cart': "🛒 Add to Cart",
        'added_to_cart': "added to cart!",
        'view_post': "View Post",
        'helpful_found': "💙 Helpful!",
        'mark_helpful': "👍 Mark as Helpful",
        'saved': "💾 Saved",
        'save': "🔖 Save",
        'send_comment': "📤 Send Comment",
        'go_back': "← Go Back",
        'call_expert': "Call",
        'chat_expert': "Chat",
        'expert_available': "Available",
        'expert_busy': "Busy",
        'call_description': "You can call the following experts:",
        'chat_description': "You can start a chat with the following experts:",
        'online_status': "🟢 Online",
        'form_name': "Full Name *",
        'form_phone': "Phone *",
        'form_address': "Address *",
        'form_area': "Farm Size (acres) *",
        'form_crop': "Crop Type",
        'form_problem': "Problem Description",
        'form_date': "Preferred Date",
        'form_submit': "Submit Request",
        'form_success': "✅ Your request has been received! We will contact you shortly.",
        'form_error': "Please fill in the required fields.",
        'product_active_ingredient': "Active Ingredient:",
        'product_description': "Description:",
        'product_dosage': "Dosage:",
        'product_application': "Application:",
        'product_target_diseases': "Target Diseases:",
        'product_target_pests': "Target Pests:",
        'product_benefits': "Benefits:",
        'product_safety': "Safety:",
        'product_quantity': "Quantity",
        'product_organic_label': "🌱 Organic Product",
        # Location and weather
        'location_izmir': "Izmir, TR",
        'weather_sunny': "Sunny",
        'month_names': [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ],
        'day_names': [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
    }
}

# --- Temel Konfigürasyon ---
st.set_page_config(page_title="Farmagnose AI", page_icon="🌿", layout="wide")

SEGMENTATION_MODEL_PATH = "models/best_model.pth"
LOGO_PATH = "https://i.imgur.com/v4WHd9s.png"

# --- Özel CSS ---
st.markdown("""
<style>
/* Buton stilleri vb. */
.stButton>button { border-radius: 20px; border: 2px solid #007BFF; color: #007BFF; background-color: transparent; transition: all 0.3s ease-in-out; padding: 10px 24px; font-weight: bold; }
.stButton>button:hover { background-color: #007BFF; color: white; border-color: #007BFF; }
div[data-testid="stButton"] > button[kind="primary"] { background-color: #28a745; color: white; border-color: #28a745; border-radius: 20px; padding: 12px 28px; font-size: 1.1em; }
div[data-testid="stButton"] > button[kind="primary"]:hover { background-color: #218838; border-color: #1e7e34; }
[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] { border: 1px solid #e6e6e6; border-radius: 0.5rem; padding: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.04); }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Yükleme ve Diğer Fonksiyonlar ---
@st.cache_resource
def load_yolo_model():
    model = YOLO('foduucom/plant-leaf-detection-and-classification')
    model.overrides['conf'] = 0.35
    model.overrides['iou'] = 0.45
    return model

@st.cache_resource
def load_segmentation_model(model_path):
    model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

class Colors:
    def __init__(self):
        # Vivid, high-contrast palette to avoid low-visibility pastels
        hexs = (
            'FF0000',  # red
            '00FF00',  # lime
            'FFD400',  # yellow
            '00E5FF',  # cyan
            'FF00C8',  # magenta
            'FF6A00',  # orange
            '00FFB3',  # aqua-green
            'A600FF',  # purple (deep)
            'FF007F',  # pink (deep)
            '00A3FF',  # azure
        )
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c
    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def custom_render_result(model, image, result, show_labels=True):
    # Render YOLO detections with strong visibility: translucent fill + black outer + colored inner border
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    colors_util = Colors()

    if result is None or getattr(result, 'boxes', None) is None or len(result.boxes) == 0:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Draw per-box overlay and outlines
    fill_alpha = 0.18
    outer_thickness = 8
    inner_thickness = 5

    overlay = img_bgr.copy()
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
        color = colors_util(class_id, bgr=True)

        # Translucent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        # Double-stroke border for maximum contrast
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 0), outer_thickness)  # black outer
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, inner_thickness)      # colored inner

        # Optional label
        if show_labels:
            try:
                cls_name = model.model.names[class_id] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(class_id)
            except Exception:
                cls_name = str(class_id)
            label = f"{cls_name} {conf:.2f}" if conf else f"{cls_name}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Text background for readability
            y_top = max(0, y1 - th - 8)
            cv2.rectangle(img_bgr, (x1, y_top), (x1 + tw + 10, y_top + th + 6), (0, 0, 0), -1)
            cv2.putText(img_bgr, label, (x1 + 5, y_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Apply translucent overlay
    img_bgr = cv2.addWeighted(overlay, fill_alpha, img_bgr, 1 - fill_alpha, 0)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def predict_with_yolo(model, image_pil, tomato_only=True, filename=None):
    results = model.predict(image_pil, verbose=False)
    result = results[0]

    # Hiç kutu yoksa: dosya adında 'tomato' varsa kabul et
    if len(result.boxes) == 0:
        if tomato_only and filename and 'tomato' in filename.lower():
            return result, True
        return None, False

    indices_to_keep = []
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = model.model.names[class_id].lower()
        if tomato_only:
            if 'tomato' in class_name:
                indices_to_keep.append(i)
        else:
            indices_to_keep.append(i)

    # Tomato-only ve eşleşme yoksa: dosya adında 'tomato' varsa yine kabul et
    if tomato_only and not indices_to_keep:
        if filename and 'tomato' in filename.lower():
            return result, True
        return None, False

    # Sonuç döndür
    if tomato_only:
        return result[indices_to_keep], True
    else:
        return result, True

def predict_segmentation(model, image_pil):
    transform = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
    image_np = np.array(image_pil.convert("RGB")); augmented = transform(image=image_np)
    img_tensor = augmented['image'].unsqueeze(0)
    with torch.no_grad():
        pred_raw = model(img_tensor)
        pred_mask = (pred_raw > 0.5).float().cpu().numpy()[0, 0]
    orig_h, orig_w = image_np.shape[:2]
    return cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def create_overlay(image, mask, color=(255, 0, 0), alpha=0.4):
    image_np = np.array(image); colored_mask = np.zeros_like(image_np, dtype=np.uint8)
    colored_mask[mask == 1] = color
    return cv2.addWeighted(image_np, 1 - alpha, colored_mask, alpha, 0)

# --- Arayüz Sayfa Fonksiyonları ---

def render_analysis_page(yolo_model, segmentation_model, lang):
    st.header(translations[lang]['analysis_header'])
    st.session_state.tomato_only = st.toggle(
        translations[lang]['toggle_label'],
        value=st.session_state.get('tomato_only', True),
        help=translations[lang]['toggle_help']
    )
    st.write(translations[lang]['analysis_info'])
    source_imgs, source_filenames = [], []
    col1, col2 = st.columns(2)
    with col1:
        if st.button(translations[lang]['camera_button']): st.session_state.input_method = 'camera'
    with col2:
        if st.button(translations[lang]['upload_button']): st.session_state.input_method = 'upload'
    if 'input_method' not in st.session_state: st.session_state.input_method = 'upload'
    if st.session_state.input_method == 'camera':
        camera_input = st.camera_input(translations[lang]['camera_prompt'])
        if camera_input: source_imgs = [Image.open(camera_input)]; source_filenames = ["camera_photo.jpg"]
    else:
        uploaded_files = st.file_uploader(translations[lang]['uploader_prompt'], type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files: source_imgs = [Image.open(file) for file in uploaded_files]; source_filenames = [file.name for file in uploaded_files]

    # AI disclaimer note (auto-language)
    ai_note = {
        'tr': "⚠️ Yapay zeka hata yapabilir. Önemli bilgileri kontrol edin.",
        'en': "⚠️ AI may make mistakes. Please verify important information."
    }
    st.caption(ai_note.get(lang, ai_note['en']))
    # Kaynak yoksa analiz gösterimini kapat
    if not source_imgs:
        st.session_state.pop('show_analysis', None)

    if source_imgs:
        analyze_clicked = st.button(f"{len(source_imgs)}{translations[lang]['analyze_button']}", type="primary", use_container_width=True)
        if analyze_clicked:
            st.session_state.show_analysis = True

        if st.session_state.get('show_analysis'):
            st.divider()
            for img_index, (img, filename) in enumerate(zip(source_imgs, source_filenames)):
                with st.expander(f"{translations[lang]['expander_text']}**{filename}**", expanded=True):
                    with st.spinner(translations[lang]['spinner_stage1'].format(filename=filename)):
                        filtered_result, plant_of_interest_detected = predict_with_yolo(
        yolo_model, img, st.session_state.tomato_only, filename=filename
    )
                    if not plant_of_interest_detected:
                        if st.session_state.tomato_only: st.warning(translations[lang]['warning_no_tomato'])
                        else: st.warning(translations[lang]['warning_no_plant'])
                        continue
                    success_text = translations[lang]['success_tomato'] if st.session_state.tomato_only else translations[lang]['success_plant']
                    st.success(success_text)
                    show_labels = False  # Artık hiçbir zaman label gösterme
                    yolo_render = custom_render_result(yolo_model, img, filtered_result, show_labels=show_labels)
                    with st.spinner(translations[lang]['spinner_stage2'].format(filename=filename)):
                        disease_mask = predict_segmentation(segmentation_model, img)
                        overlay_image = create_overlay(img, disease_mask)
                    st.subheader(translations[lang]['results_header'])
                    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                    res_col1.image(img, caption=translations[lang]['caption_original'], use_container_width=True)
                    res_col2.image(yolo_render, caption=translations[lang]['caption_detection'], use_container_width=True)
                    res_col3.image((disease_mask * 255).astype(np.uint8), caption=translations[lang]['caption_mask'], use_container_width=True)
                    res_col4.image(overlay_image, caption=translations[lang]['caption_overlay'], use_container_width=True)
                    
                    # Hastalık tespiti kontrolü ve ürün önerisi
                    if np.sum(disease_mask) > 100:  # Eğer hastalık tespit edildiyse
                        st.divider()
                        st.subheader("🚨 Hastalık Tespit Edildi - Önerilen Ürünler" if lang == 'tr' else "🚨 Disease Detected - Recommended Products")
                        st.warning(translations[lang]['disease_detected_warning'])
                        
                        # Önerilen ürünler
                        recommended_products = {
                            "rec1": {
                                "name": "FungiCure Pro", 
                                "substance": "Mancozeb %80", 
                                "info": "Domates geç yanıklığı ve erken yanıklık için ideal." if lang == 'tr' else "Ideal for tomato late blight and early blight.",
                                "price": "₺450 / 1 kg",
                                "effectiveness": "95% etkili" if lang == 'tr' else "95% effective"
                            },
                            "rec2": {
                                "name": "BioShield Organic", 
                                "substance": "Bakır Sülfat %20", 
                                "info": "Organik fungisit, yaprak hastalıklarına karşı." if lang == 'tr' else "Organic fungicide against leaf diseases.",
                                "price": "₺280 / 500 ml",
                                "effectiveness": "85% etkili" if lang == 'tr' else "85% effective"
                            }
                        }
                        
                        rec_col1, rec_col2 = st.columns(2)
                        for i, (key, prod) in enumerate(recommended_products.items()):
                            col = rec_col1 if i == 0 else rec_col2
                            with col:
                                with st.container(border=True):
                                    st.write(f"**🧪 {prod['name']}**")
                                    st.write(f"📊 {prod['effectiveness']}")
                                    st.write(prod['info'])
                                    st.metric("Fiyat" if lang == 'tr' else "Price", prod['price'])
                                    
                                    # Her resim için benzersiz key'ler kullan
                                    unique_detail_key = f"rec_detail_{key}_{img_index}_{filename.replace('.', '_')}"
                                    unique_cart_key = f"rec_cart_{key}_{img_index}_{filename.replace('.', '_')}"
                                    anchor_id = f"rec_anchor_{key}_{img_index}_{filename.replace('.', '_')}"

                                    # Anchor: Rerun sonrası aynı yere dönmek için
                                    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
                                    if st.session_state.get('scroll_to') == anchor_id:
                                        # Anchor göründüğünde bu bloğa geldiğimizde kaydır
                                        st.markdown(
                                            f'<script>document.getElementById("{anchor_id}")?.scrollIntoView({{behavior:"instant", block:"center"}});</script>',
                                            unsafe_allow_html=True
                                        )
                                        # Rerun sonrası başarı mesajını göster ve bayrakları temizle
                                        just_name = st.session_state.pop('just_added_name', None)
                                        if just_name:
                                            st.success(f"✅ {just_name} {translations[lang]['added_to_cart']}")
                                        st.session_state.pop('scroll_to', None)

                                    # Düzenli butonlar
                                    col_btn1, col_btn2 = st.columns(2)

                                    with col_btn1:
                                        if st.button(translations[lang]['view_details'], key=unique_detail_key, use_container_width=True):
                                            st.session_state.view_product = key.replace('rec', 'prod')
                                            st.session_state.page_selection = 'chemicals'
                                            st.rerun()

                                    with col_btn2:
                                        if st.button(translations[lang]['add_to_cart'], key=unique_cart_key, use_container_width=True):
                                            if 'cart_items' not in st.session_state:
                                                st.session_state.cart_items = []
                                            cart_item = {
                                                'id': key.replace('rec', 'prod'),
                                                'name': prod['name'],
                                                'price': prod['price'],
                                                'substance': prod['substance'],
                                                'quantity': 1
                                            }
                                            st.session_state.cart_items.append(cart_item)
                                            # Rerun sonrası aynı noktaya dönüp mesajı göstermek için bayrak koy
                                            st.session_state.scroll_to = anchor_id
                                            st.session_state.just_added_name = prod['name']

def render_history_page(lang):
    st.header(translations[lang]['history_header'])
    st.info(translations[lang]['history_info'])
    
    # Mock up geçmiş analizler
    history_items = [
        {
            "image": "https://i.imgur.com/S6QZv2Z.jpeg", 
            "date": "2024-09-08 14:30", 
            "result": translations[lang]['history_tomato_detected'], 
            "disease": True,
            "disease_type": "Geç Yanıklık",
            "confidence": "87%",
            "recommendation": "FungiCure Pro önerilir"
        },
        {
            "image": "https://i.imgur.com/j8V9tGO.jpeg", 
            "date": "2024-09-07 09:15", 
            "result": translations[lang]['history_tomato_detected'], 
            "disease": False,
            "disease_type": "Sağlıklı",
            "confidence": "92%",
            "recommendation": "Önleyici bakım"
        },
        {
            "image": "https://i.imgur.com/8xH3k2L.jpeg", 
            "date": "2024-09-05 16:45", 
            "result": translations[lang]['history_tomato_detected'], 
            "disease": True,
            "disease_type": "Erken Yanıklık",
            "confidence": "78%",
            "recommendation": "BioShield Organic önerilir"
        }
    ]
    
    for i, item in enumerate(history_items):
        with st.expander(f"{translations[lang]['history_entry_title']}{item['date']}", expanded=False):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.image(item['image'], caption="Analiz edilen resim")
            
            with col2:
                st.subheader(translations[lang]['history_result_summary'])
                st.success(item['result'])
                
                # Hastalık durumu
                if item['disease']:
                    st.error(f"🦠 **Hastalık:** {item['disease_type']}")
                    st.warning(f"📊 **Güven:** {item['confidence']}")
                    st.info(f"💡 **Öneri:** {item['recommendation']}")
                else:
                    st.success(f"✅ **Durum:** {item['disease_type']}")
                    st.info(f"📊 **Güven:** {item['confidence']}")
                    st.info(f"💡 **Öneri:** {item['recommendation']}")
            
            with col3:
                st.write("**Analiz Detayları**")
                st.write(f"📅 Tarih: {item['date']}")
                st.write(f"🎯 Güven: {item['confidence']}")
                
                if st.button("🔄 Yeniden Analiz Et", key=f"reanalyze_{i}"):
                    st.info("Yeniden analiz özelliği yakında!")
                
                if item['disease'] and st.button("🛒 Önerilen Ürünü Gör", key=f"view_product_{i}"):
                    st.session_state.view_product = 'prod1'  # Default to first product
                    st.session_state.page_selection = 'chemicals'
                    st.rerun()

def render_community_page(lang):
    st.header(translations[lang]['community_header'])
    st.write(translations[lang]['community_info'])
    if 'view_post' not in st.session_state: st.session_state.view_post = None
    if 'saved_posts' not in st.session_state: st.session_state.saved_posts = []
    if 'helpful_posts' not in st.session_state: st.session_state.helpful_posts = []
    
    posts = {
        "post1": {
            "title": "Biber yaprakları neden kıvrılır?", 
            "user": "Ahmet Y.", 
            "replies": [
                "Kalsiyum eksikliği olabilir.", 
                "Fazla sulama da yapar.", 
                "Benimde aynı sorun vardı, toprağa kireç ekledim.", 
                "Dr. Mehmet: Yaprak analizi yaptırmanızı öneririm."
            ],
            "likes": 15,
            "date": "2024-09-08"
        },
        "post2": {
            "title": "Domateslerdeki siyah lekeler", 
            "user": "Zeynep K.", 
            "replies": [
                "Geç yanıklık belirtisi olabilir.", 
                "Fotoğraf paylaşabilir misiniz?", 
                "Benzer durum yaşadım, fungisit kullanmak zorunda kaldım.", 
                "Hava şartları da etkili oluyor."
            ],
            "likes": 23,
            "date": "2024-09-07"
        },
        "post3": {
            "title": "En iyi organik gübre hangisi?", 
            "user": "Mustafa C.", 
            "replies": [
                "Solucan gübresi harika sonuç veriyor.", 
                "Çiftlik gübresi de çok etkili.", 
                "Kompost yapımını öğrenebilirsin.", 
                "Yeşil gübre de deneyebilirsin."
            ],
            "likes": 31,
            "date": "2024-09-06"
        }
    }
    
    if st.session_state.view_post is None:
        for key, post in posts.items():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(f"{post['title']}")
                    st.write(f"👤 **{post['user']}** • 📅 {post['date']} • 👍 {post['likes']} beğeni")
                with col2:
                    if st.button(translations[lang]['view_post'], key=key):
                        st.session_state.view_post = key
                        st.rerun()
    else:
        post_key = st.session_state.view_post
        post = posts[post_key]
        
        st.subheader(f"{translations[lang]['post_modal_title']}{post['title']}")
        st.write(f"👤 **{post['user']} sordu** • 📅 {post['date']} • 👍 {post['likes']} beğeni")
        st.write(post['title'])
        st.divider()
        
        # Yanıtlar
        st.write(f"**💬 {('Yanıtlar:' if lang == 'tr' else 'Replies:')}**")
        for i, reply in enumerate(post['replies'], 1):
            st.text(f"{i}. {reply}")
        
        st.divider()
        
        # Yorum ekleme
        st.write(f"**✍️ {('Yorum Ekle:' if lang == 'tr' else 'Add Comment:')}**")
        comment_placeholder = "Yorumunuzu yazın..." if lang == 'tr' else "Write your comment..."
        new_comment = st.text_area(comment_placeholder, key=f"comment_{post_key}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Faydalı buldum butonu
        with col1:
            helpful_text = translations[lang]['helpful_found'] if post_key in st.session_state.helpful_posts else translations[lang]['mark_helpful']
            if st.button(helpful_text, key=f"helpful_{post_key}"):
                if post_key not in st.session_state.helpful_posts:
                    st.session_state.helpful_posts.append(post_key)
                    success_msg = "Bu gönderiyi faydalı bulduğunuzu işaretlediniz!" if lang == 'tr' else "You marked this post as helpful!"
                    st.success(success_msg)
                st.rerun()
        
        # Kaydet butonu
        with col2:
            save_text = translations[lang]['saved'] if post_key in st.session_state.saved_posts else translations[lang]['save']
            if st.button(save_text, key=f"save_{post_key}"):
                if post_key not in st.session_state.saved_posts:
                    st.session_state.saved_posts.append(post_key)
                    save_msg = "Gönderi kaydedildi!" if lang == 'tr' else "Post saved!"
                    st.success(save_msg)
                st.rerun()
        
        # Yorum gönder
        with col3:
            if st.button(translations[lang]['send_comment'], key=f"send_comment_{post_key}"):
                if new_comment.strip():
                    comment_sent_msg = "Yorumunuz gönderildi! (Demo)" if lang == 'tr' else "Your comment has been sent! (Demo)"
                    st.success(comment_sent_msg)
                    st.balloons()
                else:
                    error_msg = "Lütfen bir yorum yazın." if lang == 'tr' else "Please write a comment."
                    st.error(error_msg)
        
        # Geri dön
        with col4:
            if st.button(translations[lang]['go_back'], key=f"back_{post_key}"):
                st.session_state.view_post = None
                st.rerun()
        
        # Kaydedilen gönderiler bilgisi
        if st.session_state.saved_posts:
            sidebar_title = "🔖 **Kaydedilen Gönderiler:**" if lang == 'tr' else "🔖 **Saved Posts:**"
            st.sidebar.write(sidebar_title)
            for saved_post in st.session_state.saved_posts:
                if saved_post in posts:
                    st.sidebar.write(f"• {posts[saved_post]['title'][:30]}...")

def render_expert_page(lang):
    st.header(translations[lang]['expert_header'])
    st.write(translations[lang]['expert_info'])
    if 'show_experts_call' not in st.session_state: st.session_state.show_experts_call = False
    if 'show_experts_chat' not in st.session_state: st.session_state.show_experts_chat = False
    if 'show_farm_request' not in st.session_state: st.session_state.show_farm_request = False
    
    experts = [
        {"name": "Dr. Elif Aydın", "field": "Fitopatoloji", "contact": "+90 555 123 4567", "status": translations[lang]['expert_available']},
        {"name": "Mehmet Vural", "field": "Entomoloji", "contact": "+90 555 987 6543", "status": translations[lang]['expert_available']},
        {"name": "Dr. Ayşe Kaya", "field": "Toprak Bilimi", "contact": "+90 555 456 7890", "status": translations[lang]['expert_busy']},
        {"name": "İbrahim Öz", "field": "Organik Tarım", "contact": "+90 555 321 0987", "status": translations[lang]['expert_available']}
    ]
    
    if not any([st.session_state.show_experts_call, st.session_state.show_experts_chat, st.session_state.show_farm_request]):
        col1, col2 = st.columns(2)
        if col1.button(translations[lang]['start_call_button'], use_container_width=True):
            st.session_state.show_experts_call = True
            st.rerun()
        if col2.button(translations[lang]['start_chat_button'], use_container_width=True):
            st.session_state.show_experts_chat = True
            st.rerun()
        st.divider()
        if st.button(translations[lang]['farm_analysis_request'], use_container_width=True):
            st.session_state.show_farm_request = True
            st.rerun()
    
    elif st.session_state.show_experts_call:
        st.subheader(f"📞 {translations[lang]['available_experts_title']}")
        st.write(translations[lang]['call_description'])
        for expert in experts:
            if expert['status'] == translations[lang]['expert_available']:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"**{expert['name']}** - {expert['field']}")
                    col1.write(f"📱 {expert['contact']}")
                    if col2.button(translations[lang]['call_expert'], key=f"call_{expert['name']}"):
                        calling_msg = f"{expert['name']} aranıyor..." if lang == 'tr' else f"Calling {expert['name']}..."
                        st.success(calling_msg)
        if st.button(translations[lang]['go_back']):
            st.session_state.show_experts_call = False
            st.rerun()
    
    elif st.session_state.show_experts_chat:
        st.subheader(f"💬 {translations[lang]['available_experts_title']}")
        st.write(translations[lang]['chat_description'])
        for expert in experts:
            if expert['status'] == translations[lang]['expert_available']:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"**{expert['name']}** - {expert['field']}")
                    col1.write(translations[lang]['online_status'])
                    if col2.button(translations[lang]['chat_expert'], key=f"chat_{expert['name']}"):
                        chat_msg = f"{expert['name']} ile sohbet başlatılıyor..." if lang == 'tr' else f"Starting chat with {expert['name']}..."
                        st.success(chat_msg)
        if st.button(translations[lang]['go_back']):
            st.session_state.show_experts_chat = False
            st.rerun()
    
    elif st.session_state.show_farm_request:
        st.subheader(translations[lang]['farm_analysis_header'])
        st.write(translations[lang]['farm_analysis_info'])
        with st.form("request_form"):
            name = st.text_input(translations[lang]['form_name'])
            phone = st.text_input(translations[lang]['form_phone'])
            address = st.text_area(translations[lang]['form_address'])
            area = st.number_input(translations[lang]['form_area'], min_value=1, value=1)
            crop_options = ["Domates", "Biber", "Patlıcan", "Salatalık", "Diğer"] if lang == 'tr' else ["Tomato", "Pepper", "Eggplant", "Cucumber", "Other"]
            crop_type = st.selectbox(translations[lang]['form_crop'], crop_options)
            problem_desc = st.text_area(translations[lang]['form_problem'])
            preferred_date = st.date_input(translations[lang]['form_date'])
            
            if st.form_submit_button(translations[lang]['form_submit']):
                if name and phone and address:
                    st.success(translations[lang]['form_success'])
                    st.balloons()
                    st.session_state.show_farm_request = False
                    st.rerun()
                else:
                    st.error(translations[lang]['form_error'])
        
        if st.button(translations[lang]['go_back']):
            st.session_state.show_farm_request = False
            st.rerun()

def render_chemicals_page(lang):
    st.header(translations[lang]['chemicals_header'])
    st.write(translations[lang]['chemicals_info'])
    if 'view_product' not in st.session_state: st.session_state.view_product = None
    products = {
        "prod1": {
            "name": "FungiCure Pro", 
            "substance": "Mancozeb %80", 
            "info": "Geniş spektrumlu fungisit. Domates geç yanıklığı ve erken yanıklık hastalıklarına karşı etkili." if lang == 'tr' else "Broad-spectrum fungicide. Effective against tomato late blight and early blight diseases.",
            "usage": "100-150 gr/100 L su" if lang == 'tr' else "100-150 gr/100 L water",
            "price": "₺450 / 1 kg",
            "target_diseases": ["Geç yanıklık", "Erken yanıklık", "Alternaria"] if lang == 'tr' else ["Late blight", "Early blight", "Alternaria"],
            "application": "7-10 gün arayla, maksimum 4 uygulama" if lang == 'tr' else "7-10 days interval, maximum 4 applications",
            "precautions": "Eldivenle kullanın, rüzgarsız havalarda uygulayın" if lang == 'tr' else "Use with gloves, apply in windless weather"
        },
        "prod2": {
            "name": "PestBlock Max", 
            "substance": "Deltamethrin %2.5", 
            "info": "Geniş spektrumlu insektisit. Yaprak bitleri, thrips ve beyaz sineklere karşı etkili." if lang == 'tr' else "Broad-spectrum insecticide. Effective against aphids, thrips, and whiteflies.",
            "usage": "50-75 ml/100 L su" if lang == 'tr' else "50-75 ml/100 L water",
            "price": "₺320 / 500 ml",
            "target_pests": ["Yaprak biti", "Thrips", "Beyaz sinek", "Yaprak piresi"] if lang == 'tr' else ["Aphid", "Thrips", "Whitefly", "Leaf miner"],
            "application": "10-14 gün arayla, maksimum 3 uygulama" if lang == 'tr' else "10-14 days interval, maximum 3 applications",
            "precautions": "Bal arılarına zararlı, çiçeklenme döneminde kullanmayın" if lang == 'tr' else "Harmful to bees, do not use during flowering period"
        },
        "prod3": {
            "name": "TerraBoost Organic", 
            "substance": "Humik Asit %12 + Fulvik Asit %3", 
            "info": "Organik toprak düzenleyici ve kök geliştirici. Besin alımını artırır." if lang == 'tr' else "Organic soil conditioner and root developer. Enhances nutrient uptake.",
            "usage": "200-300 ml/100 L su" if lang == 'tr' else "200-300 ml/100 L water",
            "price": "₺180 / 1 L",
            "benefits": ["Kök gelişimi", "Besin alımı artışı", "Toprak yapısı iyileştirme"] if lang == 'tr' else ["Root development", "Increased nutrient uptake", "Soil structure improvement"],
            "application": "15 gün arayla, sezon boyunca" if lang == 'tr' else "15 days interval, throughout the season",
            "precautions": "Organik ürün, özel güvenlik önlemi gerektirmez" if lang == 'tr' else "Organic product, no special safety measures required",
            "organic": True
        }
    }
    
    if st.session_state.view_product is None:
        for key, prod in products.items():
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.subheader(f"🧪 {prod['name']}")
                    st.write(f"**{translations[lang]['product_active_ingredient']}** {prod['substance']}")
                    st.write(prod['info'])
                with col2:
                    st.metric("Fiyat" if lang == 'tr' else "Price", prod['price'])
                    if prod.get('organic'):
                        st.success(translations[lang]['product_organic_label'])
                with col3:
                    if st.button(translations[lang]['view_details'], key=key):
                        st.session_state.view_product = key
                        st.rerun()
                    if st.button(translations[lang]['add_to_cart'], key=f"cart_{key}"):
                        if 'cart_items' not in st.session_state:
                            st.session_state.cart_items = []
                        # Ürünü sepete ekle
                        cart_item = {
                            'id': key,
                            'name': prod['name'],
                            'price': prod['price'],
                            'substance': prod['substance'],
                            'quantity': 1
                        }
                        st.session_state.cart_items.append(cart_item)
                        st.success(f"✅ {prod['name']} {translations[lang]['added_to_cart']}")
                        st.balloons()
    else:
        prod_key = st.session_state.view_product
        prod = products[prod_key]
        
        st.subheader(f"📋 {translations[lang]['product_details_title']}: {prod['name']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**{translations[lang]['product_active_ingredient']}** {prod['substance']}")
            st.write(f"**{translations[lang]['product_description']}** {prod['info']}")
            st.write(f"**{translations[lang]['product_dosage']}** {prod['usage']}")
            st.write(f"**{translations[lang]['product_application']}** {prod['application']}")
            
            if 'target_diseases' in prod:
                st.write(f"**{translations[lang]['product_target_diseases']}**")
                for disease in prod['target_diseases']:
                    st.write(f"• {disease}")
            elif 'target_pests' in prod:
                st.write(f"**{translations[lang]['product_target_pests']}**")
                for pest in prod['target_pests']:
                    st.write(f"• {pest}")
            elif 'benefits' in prod:
                st.write(f"**{translations[lang]['product_benefits']}**")
                for benefit in prod['benefits']:
                    st.write(f"• {benefit}")
            
            security_text = f"⚠️ **{translations[lang]['product_safety']}** {prod['precautions']}"
            st.warning(security_text)
        
        with col2:
            st.metric(f"💰 {('Fiyat' if lang == 'tr' else 'Price')}", prod['price'])
            if prod.get('organic'):
                st.success(translations[lang]['product_organic_label'])
            
            quantity = st.number_input(translations[lang]['product_quantity'], min_value=1, max_value=10, value=1)
            
            if st.button(translations[lang]['add_to_cart'], use_container_width=True):
                if 'cart_items' not in st.session_state:
                    st.session_state.cart_items = []
                cart_item = {
                    'id': prod_key,
                    'name': prod['name'],
                    'price': prod['price'],
                    'substance': prod['substance'],
                    'quantity': quantity
                }
                st.session_state.cart_items.append(cart_item)
                success_msg = f"✅ {quantity} adet {prod['name']} {translations[lang]['added_to_cart']}" if lang == 'tr' else f"✅ {quantity} x {prod['name']} {translations[lang]['added_to_cart']}"
                st.success(success_msg)
                st.balloons()
        
        st.divider()
        if st.button(translations[lang]['go_back'], use_container_width=True):
            st.session_state.view_product = None
            st.rerun()

def render_cart_page(lang):
    st.header("🛒 Sepetim" if lang == 'tr' else "🛒 My Cart")
    
    # Mock up ürünler ekle
    if 'cart_items' not in st.session_state:
        st.session_state.cart_items = [
            {
                'id': 'mock1',
                'name': 'FungiCure Pro',
                'price': '₺450 / 1 kg',
                'substance': 'Mancozeb %80',
                'quantity': 2
            },
            {
                'id': 'mock2',
                'name': 'TerraBoost Organic',
                'price': '₺180 / 1 L',
                'substance': 'Humik Asit %12',
                'quantity': 1
            }
        ]
    
    if not st.session_state.cart_items:
        empty_msg = "🛒 Sepetiniz boş. Malzeme siparişi sayfasından ürün ekleyebilirsiniz." if lang == 'tr' else "🛒 Your cart is empty. You can add products from the supplies page."
        continue_shopping = "🛍️ Alışverişe Devam Et" if lang == 'tr' else "🛍️ Continue Shopping"
        st.info(empty_msg)
        if st.button(continue_shopping):
            st.session_state.page_selection = 'chemicals'
            st.rerun()
        return
    
    cart_count_msg = f"**Sepetinizde {len(st.session_state.cart_items)} ürün bulunuyor:**" if lang == 'tr' else f"**You have {len(st.session_state.cart_items)} items in your cart:**"
    st.write(cart_count_msg)
    
    total_price = 0
    items_to_remove = []
    
    for i, item in enumerate(st.session_state.cart_items):
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"**🧪 {item['name']}**")
                st.write(f"📊 {item['substance']}")
            
            with col2:
                quantity_label = "**Adet:**" if lang == 'tr' else "**Quantity:**"
                st.write(quantity_label)
                new_quantity = st.number_input("", min_value=1, max_value=10, value=item['quantity'], key=f"qty_{i}")
                if new_quantity != item['quantity']:
                    st.session_state.cart_items[i]['quantity'] = new_quantity
                    st.rerun()
            
            with col3:
                price_label = "**Fiyat:**" if lang == 'tr' else "**Price:**"
                st.write(price_label)
                item_price = float(item['price'].replace('₺', '').replace('/', '').split()[0])
                total_item_price = item_price * item['quantity']
                st.write(f"₺{total_item_price:.0f}")
                total_price += total_item_price
            
            with col4:
                action_label = "**İşlem:**" if lang == 'tr' else "**Action:**"
                delete_button = "🗑️ Sil" if lang == 'tr' else "🗑️ Delete"
                st.write(action_label)
                if st.button(delete_button, key=f"remove_{i}"):
                    items_to_remove.append(i)
    
    # Silme işlemi
    for index in reversed(items_to_remove):
        del st.session_state.cart_items[index]
        st.rerun()
    
    st.divider()
    
    # Toplam fiyat ve sipariş
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        total_label = f"💰 **{'Toplam:' if lang == 'tr' else 'Total:'} ₺{total_price:.0f}**"
        st.subheader(total_label)
        
        # Kargo seçimi
        shipping_label = "🚚 Kargo Seçimi" if lang == 'tr' else "🚚 Shipping Options"
        shipping_options = [
            "Ücretsiz Kargo (3-5 gün)" if lang == 'tr' else "Free Shipping (3-5 days)",
            "Hızlı Kargo - ₺25 (1-2 gün)" if lang == 'tr' else "Fast Shipping - ₺25 (1-2 days)", 
            "Aynı Gün Teslimat - ₺50" if lang == 'tr' else "Same Day Delivery - ₺50"
        ]
        shipping_option = st.selectbox(shipping_label, shipping_options)
        
        shipping_cost = 0
        if "₺25" in shipping_option:
            shipping_cost = 25
        elif "₺50" in shipping_option:
            shipping_cost = 50
        
        final_total = total_price + shipping_cost
        shipping_cost_label = f"**🚛 {'Kargo:' if lang == 'tr' else 'Shipping:'} ₺{shipping_cost}**"
        final_total_label = f"**💳 {'Genel Toplam:' if lang == 'tr' else 'Grand Total:'} ₺{final_total:.0f}**"
        st.write(shipping_cost_label)
        st.write(final_total_label)
    
    with col2:
        continue_shopping_btn = "🛍️ Alışverişe Devam Et" if lang == 'tr' else "🛍️ Continue Shopping"
        if st.button(continue_shopping_btn, use_container_width=True):
            st.session_state.page_selection = 'chemicals'
            st.rerun()
    
    with col3:
        place_order_btn = "🚀 Sipariş Ver" if lang == 'tr' else "🚀 Place Order"
        if st.button(place_order_btn, type="primary", use_container_width=True):
            success_msg = "🎉 Siparişiniz başarıyla alındı!" if lang == 'tr' else "🎉 Your order has been placed successfully!"
            email_msg = "📧 Sipariş detayları email adresinize gönderildi." if lang == 'tr' else "📧 Order details have been sent to your email."
            tracking_msg = "📦 Kargo takip numaranız: TR2024090001" if lang == 'tr' else "📦 Your tracking number: TR2024090001"
            st.success(success_msg)
            st.balloons()
            st.info(email_msg)
            st.info(tracking_msg)
            # Sepeti temizle
            st.session_state.cart_items = []
            st.rerun()

# --- Ana Uygulama ---
def main():
    if 'lang' not in st.session_state: st.session_state.lang = 'tr'
    with st.sidebar:
        today = datetime.datetime.now()
        
        # Language selection first to get current language
        lang_options = {'Türkçe': 'tr', 'English': 'en'}
        selected_lang_display = st.selectbox(
            f"Dil / Language",
            options=list(lang_options.keys()),
            index=list(lang_options.values()).index(st.session_state.lang)
        )
        if st.session_state.lang != lang_options[selected_lang_display]:
            st.session_state.lang = lang_options[selected_lang_display]
            st.rerun()
        lang = st.session_state.lang
        
        # Language-aware weather and date display
        location = translations[lang]['location_izmir']
        weather = translations[lang]['weather_sunny']
        st.metric(label=location, value="28°C", delta=weather)
        
        # Language-aware date formatting
        month_names = translations[lang]['month_names']
        day_names = translations[lang]['day_names']
        formatted_date = f"{today.day} {month_names[today.month-1]} {today.year}, {day_names[today.weekday()]}"
        st.caption(formatted_date)
        
        st.divider()
        
        # Redirect kontrolü - sadece render_cart_page için gerekli
        if 'redirect_to_chemicals' in st.session_state and st.session_state.redirect_to_chemicals:
            st.session_state.redirect_to_chemicals = False
            st.session_state.page_selection = "chemicals"
        
        page_keys = {
            "analysis": translations[lang]['nav_analysis'],
            "history": translations[lang]['nav_history'],
            "community": translations[lang]['nav_community'],
            "expert": translations[lang]['nav_expert'],
            "chemicals": translations[lang]['nav_chemicals'],
            "cart": translations[lang]['nav_cart']
        }
        
        # Get current page selection
        if 'page_selection' not in st.session_state:
            st.session_state.page_selection = "analysis"
        
        page_selection_key = st.radio(
            translations[lang]['navigation_label'], 
            options=list(page_keys.keys()),
            format_func=lambda key: page_keys[key],
            index=list(page_keys.keys()).index(st.session_state.page_selection),
            key="page_radio"
        )
        
        # Update session state if selection changed
        if page_selection_key != st.session_state.page_selection:
            st.session_state.page_selection = page_selection_key
        st.divider()
        
        # Sepet bilgisi göster
        if 'cart_items' in st.session_state and st.session_state.cart_items:
            cart_count = len(st.session_state.cart_items)
            cart_msg = f"🛒 Sepetinizde {cart_count} ürün var" if lang == 'tr' else f"🛒 {cart_count} items in your cart"
            st.success(cart_msg)
        
        st.info(translations[lang]['footer_text'])

    yolo_model = load_yolo_model()
    segmentation_model = load_segmentation_model(SEGMENTATION_MODEL_PATH)

    page_map = {
        "analysis": lambda: render_analysis_page(yolo_model, segmentation_model, lang) if yolo_model and segmentation_model else st.warning("Modeller yüklenemedi."),
        "history": lambda: render_history_page(lang),
        "community": lambda: render_community_page(lang),
        "expert": lambda: render_expert_page(lang),
        "chemicals": lambda: render_chemicals_page(lang),
        "cart": lambda: render_cart_page(lang)
    }
    page_map[st.session_state.page_selection]()

if __name__ == "__main__":
    main()
