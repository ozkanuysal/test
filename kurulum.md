# İlan Öneri Sistemi - Hızlı Kurulum Kılavuzu

## 🎯 Özet

Bu proje, kullanıcı davranışları (graph-based) ve ilan içeriği (text-based) embedding'lerini birleştiren hibrit bir iş ilanı öneri sistemidir.

## 📦 Gereksinimler

- Python 3.8+
- Docker (Qdrant için)
- 4GB+ RAM
- ~2GB disk alanı

## 🚀 5 Dakikada Kurulum

### 1. Repoyu klonlayın ve klasöre girin

```bash
git clone <repo-url>
cd job-recommendation-system
```

### 2. Otomatik kurulum scriptini çalıştırın

```bash
chmod +x setup.sh
./setup.sh
```

Bu script otomatik olarak:
- Virtual environment oluşturur
- Python paketlerini yükler
- Qdrant'ı başlatır
- Gerekli klasörleri oluşturur

### 3. Veri dosyalarınızı ekleyin

Ana dizine koyun:
-- `user_event_data.csv`
-- `item_information.csv`

### 4. Sistemi çalıştırın

```bash
python main