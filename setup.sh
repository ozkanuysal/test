#!/bin/bash

# İlan Öneri Sistemi - Hızlı Kurulum Script'i
# Bu script sistemi otomatik olarak kurar

set -e  # Hata durumunda dur

echo "======================================================================"
echo "İLAN ÖNERİ SİSTEMİ - HIZLI KURULUM"
echo "======================================================================"
echo ""

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python versiyonu kontrolü
echo -e "${YELLOW}Python versiyonu kontrol ediliyor...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}HATA: Python3 bulunamadı!${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python $PYTHON_VERSION bulundu${NC}"

# Virtual environment oluştur
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Virtual environment oluşturuluyor...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment oluşturuldu${NC}"
else
    echo -e "\n${GREEN}✓ Virtual environment mevcut${NC}"
fi

# Virtual environment aktif et
echo -e "\n${YELLOW}Virtual environment aktif ediliyor...${NC}"
source venv/bin/activate

# Requirements yükle
echo -e "\n${YELLOW}Python paketleri yükleniyor...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Paketler yüklendi${NC}"

# Docker kontrolü
echo -e "\n${YELLOW}Docker kontrol ediliyor...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}⚠ Docker bulunamadı!${NC}"
    echo "Docker'ı manuel olarak yüklemeniz gerekiyor:"
    echo "  https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker bulundu${NC}"

# Docker Compose kontrolü
echo -e "\n${YELLOW}Docker Compose kontrol ediliyor...${NC}"
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠ docker-compose bulunamadı, 'docker compose' kullanılacak${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
    echo -e "${GREEN}✓ Docker Compose bulundu${NC}"
fi

# Qdrant başlat
echo -e "\n${YELLOW}Qdrant başlatılıyor...${NC}"
$DOCKER_COMPOSE up -d

# Qdrant'ın hazır olmasını bekle
echo -e "${YELLOW}Qdrant'ın hazır olması bekleniyor...${NC}"
sleep 5

# Health check
max_attempts=10
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:6333/health > /dev/null; then
        echo -e "${GREEN}✓ Qdrant hazır${NC}"
        break
    fi
    attempt=$((attempt+1))
    echo "  Deneme $attempt/$max_attempts..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}HATA: Qdrant başlatılamadı!${NC}"
    echo "Manuel olarak kontrol edin: docker ps"
    exit 1
fi

# Outputs klasörü oluştur
echo -e "\n${YELLOW}Çıktı klasörleri oluşturuluyor...${NC}"
mkdir -p outputs
mkdir -p logs
echo -e "${GREEN}✓ Klasörler oluşturuldu${NC}"

# Veri dosyalarını kontrol et
echo -e "\n${YELLOW}Veri dosyaları kontrol ediliyor...${NC}"
if [ ! -f "user_event_data.csv" ] || [ ! -f "item_information.csv" ]; then
    echo -e "${RED}⚠ UYARI: Veri dosyaları bulunamadı!${NC}"
    echo "Lütfen aşağıdaki dosyaları ana dizine ekleyin:"
    echo "  - user_event_data.csv"
    echo "  - item_information.csv"
    echo ""
    echo "Devam etmek için Enter'a basın..."
    read
else
    echo -e "${GREEN}✓ Veri dosyaları bulundu${NC}"
fi

# Kurulum tamamlandı
echo ""
echo "======================================================================"
echo -e "${GREEN}KURULUM TAMAMLANDI!${NC}"
echo "======================================================================"
echo ""
echo "Sistemi çalıştırmak için:"
echo -e "  ${YELLOW}python main.py${NC}"
echo ""
echo "Test etmek için:"
echo -e "  ${YELLOW}python test_system.py${NC}"
echo ""
echo "Qdrant UI:"
echo -e "  ${YELLOW}http://localhost:6333/dashboard${NC}"
echo ""
echo "Qdrant'ı durdurmak için:"
echo -e "  ${YELLOW}$DOCKER_COMPOSE down${NC}"
echo ""
echo "======================================================================"