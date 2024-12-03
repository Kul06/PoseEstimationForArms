# Kol Poz Tahmini Uygulaması

Bu proje, fizik tedavi süreçlerini desteklemek için tasarlanmış bir **Kol Poz Tahmini** uygulamasıdır. Yapay zeka tabanlı bu uygulama, hareketleri analiz ederek doğru açılarda gerçekleştirilen egzersizleri tespit eder ve geri bildirim sağlar.

## 📌 **Proje Amacı**
Fizik tedavi sırasında yapılan egzersizlerin doğruluğunu analiz etmek ve hastalara gerçek zamanlı geri bildirim sunarak tedavi sürecini optimize etmektir. Bu uygulama, özellikle kol açısı hareketlerini (ör. dirsek bükme/açma, kol kaldırma) değerlendirmek için tasarlanmıştır.

---

## 🛠️ **Özellikler**
- **Gerçek Zamanlı Analiz:** Videolar üzerinden kol hareketlerini analiz eder.
- **Açı Hesaplama:** Dirsek ve omuz açısını hesaplayarak egzersizlerin doğruluğunu değerlendirir.
- **Hareket Tanıma:** 
  - **Dirsek Bükme/Açma**
  - **Kol Kaldırma**
  - **Yatay Kol Hareketleri (Abdüksiyon/Addüksiyon)**
- **Çıktı Kaydetme:** Analiz edilen videolar üzerine görsel açı verileri ekleyerek çıktı olarak kaydeder.
- **Karanlık Tema Desteği:** Kullanıcı dostu ve modern bir arayüz.

---

## 📋 **Kullanılan Teknolojiler**
- **Python**: Proje dili.
- **MediaPipe**: İnsan vücudu poz tahmini ve anahtar noktaları çıkarma.
- **OpenCV**: Görüntü işleme ve video analizi.
- **PyQt5**: Kullanıcı arayüzü tasarımı.
- **NumPy**: Matematiksel hesaplamalar.

---

## 🚀 **Kurulum ve Çalıştırma**
### Gereksinimler