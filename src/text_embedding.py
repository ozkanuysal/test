import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import re
from typing import Dict, List

class JobTextEmbedding:
    """
    İlan metinlerinden (başlık + açıklama) embedding oluşturma sınıfı
    
    Seçilen Model: BERTurk (Turkish BERT)
    
    Seçim Sebepleri:
    1. Türkçe Dil Desteği: İlan verileri Türkçe olduğu için Türkçe pre-trained model şart
    2. Contextual Embeddings: BERT, kelimelerin cümle içindeki anlamını yakalar
    3. Transfer Learning: Geniş korpusta eğitilmiş, domain adaptation kolay
    4. Production-Ready: Transformers kütüphanesi ile kolay deploy
    5. State-of-the-art: NLP task'lerinde yüksek performans
    
    Alternatifler:
    - Word2Vec: Contextual değil, eski teknoloji
    - TF-IDF: Basit, semantik anlam yakalamıyor
    - multilingual-BERT: Türkçe için özelleşmiş model daha iyi
    """
    
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased'):
        """
        Args:
            model_name: HuggingFace model adı
                Önerilen: 'dbmdz/bert-base-turkish-cased' (BERTurk)
        """
        print(f"Model yükleniyor: {model_name}")
        
        # Device seçimi (GPU varsa kullan)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Tokenizer ve model yükle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Evaluation mode
        
        self.embeddings = {}
        
        print(f"Model başarıyla yüklendi")
    
    def clean_text(self, text):
        """
        Metin temizleme işlemleri
        
        Temizleme Adımları:
        1. None/NaN kontrolü
        2. HTML taglerini kaldır
        3. Özel karakterleri temizle
        4. Fazla boşlukları düzelt
        5. Küçük harfe çevir (opsiyonel - BERT büyük/küçük harf duyarlı)
        
        Args:
            text: Ham metin
            
        Returns:
            Temizlenmiş metin
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # HTML taglerini kaldır
        text = re.sub(r'<[^>]+>', '', text)
        
        # URL'leri kaldır
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Email adreslerini kaldır
        text = re.sub(r'\S+@\S+', '', text)
        
        # Özel karakterleri temizle (Türkçe karakterler hariç)
        text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?-]', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        # Baş ve son boşlukları kaldır
        text = text.strip()
        
        return text
    
    def create_combined_text(self, title, description, title_weight=2):
        """
        Başlık ve açıklamayı birleştirir
        
        Strategi:
        - Başlığa daha fazla önem ver (title_weight parametresi)
        - Başlığı tekrarla veya özel token ekle
        
        Args:
            title: İlan başlığı
            description: İlan açıklaması
            title_weight: Başlığın tekrar sayısı (daha fazla = daha önemli)
            
        Returns:
            Birleştirilmiş metin
        """
        title = self.clean_text(title)
        description = self.clean_text(description)
        
        # Başlığı birkaç kez tekrarla (daha fazla önem vermek için)
        combined = " ".join([title] * title_weight)
        
        if description:
            combined += " " + description
        
        return combined
    
    def get_bert_embedding(self, text, max_length=512, pooling='mean'):
        """
        BERT kullanarak metin embedding'i üretir
        
        Pooling Stratejileri:
        1. CLS Token: [CLS] token'ının embedding'i (cümle temsili)
        2. Mean Pooling: Tüm token embedding'lerinin ortalaması
        3. Max Pooling: Her boyut için maksimum değer
        
        Seçilen: Mean Pooling
        Sebep: Tüm kelimelerin bilgisini kullanır, daha robust
        
        Args:
            text: Input metni
            max_length: Maksimum token sayısı
            pooling: Pooling stratejisi ('cls', 'mean', 'max')
            
        Returns:
            Embedding vektörü (numpy array)
        """
        if not text or len(text.strip()) == 0:
            # Boş metin için sıfır vektör
            return np.zeros(768)  # BERT base hidden size
        
        # Tokenize et
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Forward pass (gradient hesaplama)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pooling uygula
        if pooling == 'cls':
            # CLS token (ilk token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
        elif pooling == 'mean':
            # Mean pooling (attention mask kullan)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Padding token'ları maskeleyerek ortalama al
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
            
        elif pooling == 'max':
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            embedding = torch.max(token_embeddings, dim=1)[0].cpu().numpy()[0]
        
        else:
            raise ValueError(f"Geçersiz pooling: {pooling}")
        
        return embedding
    
    def create_embeddings(self, job_data_df, batch_size=32, 
                         title_weight=2, pooling='mean'):
        """
        Tüm ilanlar için embedding oluşturur
        
        Args:
            job_data_df: İlan verileri DataFrame
            batch_size: Batch boyutu (bellek optimizasyonu)
            title_weight: Başlık ağırlığı
            pooling: Pooling stratejisi
            
        Returns:
            embeddings: Dictionary {item_id: embedding_vector}
        """
        print(f"\n{len(job_data_df)} ilan için embedding oluşturuluyor...")
        print(f"Parametreler: batch_size={batch_size}, title_weight={title_weight}, pooling={pooling}")
        
        self.embeddings = {}
        
        # Batch'ler halinde işle
        for i in tqdm(range(0, len(job_data_df), batch_size)):
            batch_df = job_data_df.iloc[i:i+batch_size]
            
            for _, row in batch_df.iterrows():
                item_id = row['item_id']
                title = row.get('pozisyon_adi', '')
                description = row.get('item_id_aciklama', '')
                
                # Metni birleştir ve temizle
                combined_text = self.create_combined_text(title, description, title_weight)
                
                # Embedding oluştur
                embedding = self.get_bert_embedding(combined_text, pooling=pooling)
                self.embeddings[item_id] = embedding
        
        print(f"Toplam {len(self.embeddings)} embedding oluşturuldu")
        print(f"Embedding boyutu: {len(next(iter(self.embeddings.values())))}")
        
        return self.embeddings
    
    def compute_similarity(self, item_id1, item_id2):
        """
        İki ilan arasındaki cosine similarity hesaplar
        
        Args:
            item_id1, item_id2: İlan ID'leri
            
        Returns:
            Similarity skoru (0-1 arası)
        """
        if item_id1 not in self.embeddings or item_id2 not in self.embeddings:
            return 0.0
        
        emb1 = self.embeddings[item_id1]
        emb2 = self.embeddings[item_id2]
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return similarity
    
    def get_most_similar(self, item_id, top_k=10):
        """
        Verilen ilana en benzer ilanları bulur
        
        Args:
            item_id: İlan ID'si
            top_k: Döndürülecek benzer ilan sayısı
            
        Returns:
            List of (item_id, similarity_score)
        """
        if item_id not in self.embeddings:
            print(f"Hata: {item_id} için embedding bulunamadı")
            return []
        
        similarities = []
        target_emb = self.embeddings[item_id]
        
        for other_id, other_emb in self.embeddings.items():
            if other_id != item_id:
                sim = np.dot(target_emb, other_emb) / (
                    np.linalg.norm(target_emb) * np.linalg.norm(other_emb)
                )
                similarities.append((other_id, sim))
        
        # Benzerliğe göre sırala
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def evaluate_semantic_quality(self, job_data_df, sample_size=10):
        """
        Semantic embedding kalitesini değerlendirir
        
        Değerlendirme Yöntemi:
        - Aynı başlığa sahip ilanlar yüksek benzerlik göstermeli
        - Farklı kategorilerdeki ilanlar düşük benzerlik göstermeli
        
        Args:
            job_data_df: İlan verileri
            sample_size: Örnek sayısı
            
        Returns:
            quality_metrics: Kalite metrikleri
        """
        print("\nSemantik kalite değerlendirmesi...")
        
        # Aynı başlığa sahip ilan çiftleri bul
        title_groups = job_data_df.groupby('pozisyon_adi')['item_id'].apply(list).to_dict()
        
        same_title_sims = []
        diff_title_sims = []
        
        # Aynı başlık benzerliği
        for title, items in list(title_groups.items())[:sample_size]:
            if len(items) >= 2:
                for i in range(min(len(items), 5)):
                    for j in range(i+1, min(len(items), 5)):
                        if items[i] in self.embeddings and items[j] in self.embeddings:
                            sim = self.compute_similarity(items[i], items[j])
                            same_title_sims.append(sim)
        
        # Farklı başlık benzerliği
        all_items = list(self.embeddings.keys())
        np.random.seed(42)
        for _ in range(min(100, len(all_items))):
            idx1, idx2 = np.random.choice(len(all_items), 2, replace=False)
            sim = self.compute_similarity(all_items[idx1], all_items[idx2])
            diff_title_sims.append(sim)
        
        metrics = {
            'avg_same_title_similarity': np.mean(same_title_sims) if same_title_sims else 0,
            'avg_diff_title_similarity': np.mean(diff_title_sims) if diff_title_sims else 0,
            'semantic_gap': np.mean(same_title_sims) - np.mean(diff_title_sims) if same_title_sims and diff_title_sims else 0
        }
        
        print(f"Aynı başlık ortalama benzerlik: {metrics['avg_same_title_similarity']:.4f}")
        print(f"Farklı başlık ortalama benzerlik: {metrics['avg_diff_title_similarity']:.4f}")
        print(f"Semantik fark: {metrics['semantic_gap']:.4f}")
        
        return metrics
    
    def save_embeddings(self, path='job_embeddings_text.npz'):
        """Embedding'leri kaydet"""
        np.savez(path, **{str(k): v for k, v in self.embeddings.items()})
        print(f"Text embeddings kaydedildi: {path}")
    
    def load_embeddings(self, path='job_embeddings_text.npz'):
        """Embedding'leri yükle"""
        data = np.load(path)
        self.embeddings = {int(k): v for k, v in data.items()}
        print(f"Text embeddings yüklendi: {path} ({len(self.embeddings)} ilan)")
        return self.embeddings


# Kullanım örneği
if __name__ == "__main__":
    # İlan verilerini yükle
    job_data = pd.read_csv('item_information.csv')
    
    # Text embedding sınıfını oluştur
    text_embedder = JobTextEmbedding(
        model_name='dbmdz/bert-base-turkish-cased'
    )
    
    # Embedding'leri oluştur
    embeddings = text_embedder.create_embeddings(
        job_data_df=job_data,
        batch_size=16,  # GPU belleğine göre ayarla
        title_weight=2,
        pooling='mean'
    )
    
    # Kalite değerlendirmesi
    metrics = text_embedder.evaluate_semantic_quality(job_data)
    
    # Örnek: Benzer ilanları bul
    sample_item_id = job_data['item_id'].iloc[0]
    similar_jobs = text_embedder.get_most_similar(sample_item_id, top_k=5)
    
    print(f"\nİlan {sample_item_id} için en benzer 5 ilan:")
    for job_id, similarity in similar_jobs:
        job_title = job_data[job_data['item_id'] == job_id]['pozisyon_adi'].values[0]
        print(f"  {job_id}: {job_title[:50]}... (benzerlik: {similarity:.4f})")
    
    # Embedding'leri kaydet
    text_embedder.save_embeddings('job_embeddings_text.npz')