# Gelişmiş Metinden Sese Dönüştürücü Web Uygulaması 🗣️

Python, Gradio ve Hugging Face Transformers kütüphaneleri kullanılarak oluşturulmuş, kullanıcı tarafından girilen metni gerçekçi bir insan sesiyle seslendiren modern ve kullanıcı dostu bir metinden sese dönüştürme (Text-to-Speech) web uygulamasıdır.

Bu proje, **#30DayAIMarathon** kapsamında Süleyman Toklu tarafından geliştirilmiştir.

## ✨ Özellikler

- **Yüksek Kaliteli Ses:** `microsoft/speecht5_tts` ve `microsoft/speecht5_hifigan` modelleri ile doğal ve akıcı ses üretimi.
- **Dinamik Ses Kimliği:** Uygulama, harici bir ses dosyasına ihtiyaç duymadan, başlangıçta Hugging Face `datasets` kütüphanesinden güvenilir bir ses kimliği (speaker embedding) oluşturur.
- **Kullanıcı Dostu Arayüz:** Gradio ile oluşturulmuş basit, modern ve etkileşimli bir web arayüzü.
- **Donanım Desteği:** Hem GPU (CUDA) hem de CPU üzerinde çalışacak şekilde tasarlanmıştır.
- **Kolay Kullanım:** Denemeler için hazır örnek metinler sunar.

## 🛠️ Kullanılan Teknolojiler

- **Backend:** Python
- **Web Arayüzü:** Gradio
- **Yapay Zeka Modelleri:** PyTorch, Hugging Face Transformers
- **Modeller:**
  - **Metinden Spektrograma:** `microsoft/speecht5_tts`
  - **Spektrogramdan Sese (Vokoder):** `microsoft/speecht5_hifigan`

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.


## ⚙️ Nasıl Çalışır?

1.  **Başlangıç:** Uygulama başlatıldığında, `microsoft/speecht5_tts` ve `microsoft/speecht5_hifigan` modelleri Hugging Face'ten yüklenir.
2.  **Ses Kimliği Oluşturma:** Sesin karakterini belirlemek için `datasets` kütüphanesi üzerinden standart bir konuşmacı vektörü (speaker embedding) çekilir.
3.  **Metin Girişi:** Kullanıcı, Gradio arayüzündeki metin kutusuna bir cümle girer ve "Generate Speech" butonuna tıklar.
4.  **Sentezleme:**
    - Girilen metin, `SpeechT5Processor` ile modelin anlayacağı token'lara dönüştürülür.
    - `SpeechT5` modeli, bu token'ları ve önceden oluşturulan ses kimliğini kullanarak bir spektrogram üretir.
    - `HifiGan` vokoder'ı, bu spektrogramı dinlenebilir bir ses dalgasına çevirir.
5.  **Ses Çıktısı:** Oluşturulan ses, arayüzdeki ses oynatıcı bileşeninde kullanıcıya sunulur.

---

*Created by Süleyman Toklu for the #30DayAIMarathon.*