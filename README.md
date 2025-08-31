title: Metinden Sese Dönüştürücü emoji: 🗣️ colorFrom: blue colorTo: green sdk: gradio sdk_version: 4.12.0 app_file: app.py pinned: false license: mit
🗣️ Metinden Sese Dönüştürücü (Text-to-Speech)

30 Günlük Yapay Zeka Maratonu'nun 19. Günü için geliştirilen bu uygulama, yazdığınız metinleri yapay zeka kullanarak sese dönüştürür. Microsoft'un SpeechT5 modelini temel alır ve çeşitli ses tonları sunar.
🚀 Nasıl Kullanılır?

Uygulamayı kullanmak çok kolay:

    Metin Girin: Yukarıdaki metin kutusuna seslendirmek istediğiniz herhangi bir şeyi yazın.

    Ses Seçin: "Choose a Voice" menüsünden Awesome-Amy veya Deep-David gibi farklı ses tonlarından birini seçin.

    Oluştur'a Tıklayın: "Generate Speech" butonuna basarak sesi oluşturun.

    Dinleyin: Birkaç saniye içinde oluşturulan ses, sağdaki ses çalar üzerinde belirecektir. Play tuşuna basarak dinleyebilirsiniz!

🌟 Temel Özellikler

    Yüksek Kaliteli Ses: Microsoft'un son teknoloji SpeechT5 modelini kullanarak doğal ve akıcı sesler üretir.

    Farklı Ses Tonları: Farklı "konuşmacı kimlikleri" sayesinde çeşitli sesler arasından seçim yapma imkanı sunar.

    Etkileşimli Arayüz: Gradio ile oluşturulmuş basit ve kullanıcı dostu bir arayüze sahiptir.

🛠️ Teknik Detaylar

    Ana Model: microsoft/speecht5_tts

    Vocoder (Ses Sentezleyici): microsoft/speecht5_hifigan

    Ses Kimlikleri Veri Seti: Matthijs/cmu-arctic-xvectors

Bu uygulama, Süleyman Toklu tarafından #30DayAIMarathon projesi kapsamında geliştirilmiştir.