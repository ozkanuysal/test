import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Tuple
from tqdm import tqdm

class HybridJobRecommendationSystem:
    """
    Graph-based ve Text-based embedding'leri birleştiren hibrit öneri sistemi
    
    Sistem Tasarımı:
    1. İki farklı embedding kaynağı:
       - Graph-based (Node2Vec): Kullanıcı davranışlarından öğrenen
       - Text-based (BERT): İlan içeriğinden öğrenen
    
    2. Birleştirme Stratejisi: Weighted Combination
       - Her iki embedding'i birleştirerek tek bir vektör oluştur
       - Ağırlıklar: alpha * graph_emb + (1-alpha) * text_emb
       - alpha parametresi: Davranış vs içerik dengesi
    
    3. Vector Database: Qdrant
       - Hızlı similarity search
       - Scalable
       - Production-ready
    
    Seçim Sebepleri:
    - Cold Start Problem: Yeni ilanlar için text embedding yeter
    - Warm Items: Mevcut ilanlar için graph embedding davranış bilgisi ekler
    - Flexibility: Alpha parametresiyle davranış-içerik dengesi ayarlanabilir
    - Diversity: İki farklı kaynak farklı perspektifler sunar
    """
    
    def __init__(self, qdrant_url="localhost", qdrant_port=6333, collection_name="job_recommendations"):
        """
        Args:
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            collection_name: Qdrant collection adı
        """
        self.collection_name = collection_name
        
        # Qdrant client'ı başlat
        self.client = QdrantClient(url=qdrant_url, port=qdrant_port)
        
        # Embedding'leri saklayacak dictionary'ler
        self.graph_embeddings = {}
        self.text_embeddings = {}
        self.hybrid_embeddings = {}
        self.job_metadata = {}
        
        print(f"Qdrant client başlatıldı: {qdrant_url}:{qdrant_port}")
    
    def load_embeddings(self, graph_emb_path, text_emb_path):
        """
        Graph ve text embedding'leri yükler
        
        Args:
            graph_emb_path: Graph embedding NPZ dosya yolu
            text_emb_path: Text embedding NPZ dosya yolu
        """
        print("Embedding'ler yükleniyor...")
        
        # Graph embeddings
        graph_data = np.load(graph_emb_path)
        self.graph_embeddings = {int(k): v for k, v in graph_data.items()}
        print(f"  Graph embeddings: {len(self.graph_embeddings)} ilan")
        
        # Text embeddings
        text_data = np.load(text_emb_path)
        self.text_embeddings = {int(k): v for k, v in text_data.items()}
        print(f"  Text embeddings: {len(self.text_embeddings)} ilan")
        
        # Ortak ilanları bul
        common_items = set(self.graph_embeddings.keys()) & set(self.text_embeddings.keys())
        print(f"  Ortak ilan sayısı: {len(common_items)}")
    
    def normalize_embedding(self, embedding):
        """
        Embedding'i L2 normalize eder
        
        Sebep: Farklı kaynaklardan gelen embedding'lerin ölçeklerini eşitlemek
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def create_hybrid_embeddings(self, alpha=0.5, normalize=True):
        """
        Graph ve text embedding'leri birleştirir
        
        Birleştirme Stratejileri:
        1. Concatenation: [graph_emb; text_emb] - Boyut ikiye katlanır
        2. Weighted Sum: alpha * graph + (1-alpha) * text - Boyut sabit kalır
        3. Learned Combination: Neural network ile öğren - Kompleks
        
        Seçilen: Weighted Sum
        - Basit ve etkili
        - Boyut artmaz (önemli: Qdrant'ta daha az depolama)
        - Alpha ile fine-tuning yapılabilir
        
        Args:
            alpha: Graph embedding ağırlığı (0-1 arası)
                  - 0: Sadece text embedding
                  - 1: Sadece graph embedding
                  - 0.5: Eşit ağırlık
            normalize: Embedding'leri normalize et mi?
            
        Returns:
            hybrid_embeddings: Dictionary {item_id: hybrid_embedding}
        """
        print(f"\nHibrit embedding'ler oluşturuluyor (alpha={alpha})...")
        
        # Embedding boyutlarını kontrol et
        sample_graph = next(iter(self.graph_embeddings.values()))
        sample_text = next(iter(self.text_embeddings.values()))
        
        graph_dim = len(sample_graph)
        text_dim = len(sample_text)
        
        print(f"  Graph embedding boyutu: {graph_dim}")
        print(f"  Text embedding boyutu: {text_dim}")
        
        if graph_dim != text_dim:
            print(f"  UYARI: Embedding boyutları farklı! Boyut eşitleme yapılıyor...")
            # Daha küçük olanı pad et veya daha büyük olanı truncate et
            target_dim = min(graph_dim, text_dim)
        else:
            target_dim = graph_dim
        
        self.hybrid_embeddings = {}
        
        # Her ilan için hibrit embedding oluştur
        all_items = set(self.graph_embeddings.keys()) | set(self.text_embeddings.keys())
        
        for item_id in tqdm(all_items):
            # Graph embedding'i al (yoksa sıfır vektör)
            if item_id in self.graph_embeddings:
                graph_emb = self.graph_embeddings[item_id][:target_dim]
            else:
                graph_emb = np.zeros(target_dim)
            
            # Text embedding'i al (yoksa sıfır vektör)
            if item_id in self.text_embeddings:
                text_emb = self.text_embeddings[item_id][:target_dim]
            else:
                text_emb = np.zeros(target_dim)
            
            # Normalize (opsiyonel)
            if normalize:
                graph_emb = self.normalize_embedding(graph_emb)
                text_emb = self.normalize_embedding(text_emb)
            
            # Weighted combination
            hybrid_emb = alpha * graph_emb + (1 - alpha) * text_emb
            
            # Final normalization
            if normalize:
                hybrid_emb = self.normalize_embedding(hybrid_emb)
            
            self.hybrid_embeddings[item_id] = hybrid_emb
        
        print(f"Toplam {len(self.hybrid_embeddings)} hibrit embedding oluşturuldu")
        print(f"Hibrit embedding boyutu: {target_dim}")
        
        return self.hybrid_embeddings
    
    def load_job_metadata(self, job_data_df):
        """
        İlan metadata'larını yükler (başlık, açıklama vb.)
        
        Args:
            job_data_df: İlan verileri DataFrame
        """
        print("İlan metadata'ları yükleniyor...")
        
        for _, row in job_data_df.iterrows():
            self.job_metadata[row['item_id']] = {
                'title': row.get('pozisyon_adi', ''),
                'description': row.get('item_id_aciklama', ''),
                'category': row.get('kategori', '') if 'kategori' in row else ''
            }
        
        print(f"  {len(self.job_metadata)} ilan metadata'sı yüklendi")
    
    def setup_qdrant_collection(self, vector_dim, recreate=False):
        """
        Qdrant collection'ını oluşturur veya yeniden oluşturur
        
        Args:
            vector_dim: Vector boyutu
            recreate: Mevcut collection'ı sil ve yeniden oluştur
        """
        # Mevcut collection'ı kontrol et
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)
        
        if collection_exists:
            if recreate:
                print(f"Mevcut '{self.collection_name}' collection siliniyor...")
                self.client.delete_collection(collection_name=self.collection_name)
            else:
                print(f"'{self.collection_name}' collection zaten mevcut")
                return
        
        # Yeni collection oluştur
        print(f"'{self.collection_name}' collection oluşturuluyor...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE  # Cosine similarity kullan
            )
        )
        print(f"Collection oluşturuldu (vector_dim={vector_dim}, distance=COSINE)")
    
    def upload_to_qdrant(self, batch_size=100):
        """
        Hibrit embedding'leri Qdrant'a yükler
        
        Args:
            batch_size: Batch boyutu (bellek optimizasyonu)
        """
        if not self.hybrid_embeddings:
            raise ValueError("Önce hibrit embedding'ler oluşturulmalı (create_hybrid_embeddings)")
        
        # Collection'ı setup et
        vector_dim = len(next(iter(self.hybrid_embeddings.values())))
        self.setup_qdrant_collection(vector_dim, recreate=True)
        
        print(f"\n{len(self.hybrid_embeddings)} vektör Qdrant'a yükleniyor...")
        
        # Batch'ler halinde yükle
        points = []
        for item_id, embedding in tqdm(self.hybrid_embeddings.items()):
            # Metadata hazırla
            payload = {
                'item_id': int(item_id),
                'title': self.job_metadata.get(item_id, {}).get('title', ''),
                'category': self.job_metadata.get(item_id, {}).get('category', '')
            }
            
            # Point oluştur
            point = PointStruct(
                id=int(item_id),
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
            
            # Batch dolduğunda yükle
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []
        
        # Kalan point'leri yükle
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"Yükleme tamamlandı!")
        
        # Collection info
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        print(f"Collection info: {collection_info.points_count} vektör")
    
    def recommend_similar_jobs(self, item_id: int, top_n: int = 10, 
                              filter_category: str = None) -> List[Dict]:
        """
        Verilen ilan ID'sine benzer ilanları önerir
        
        Args:
            item_id: İlan ID'si
            top_n: Öneri sayısı
            filter_category: Sadece belirli kategoriden öner (opsiyonel)
            
        Returns:
            List of dictionaries içeren öneriler:
            [
                {
                    'item_id': 123,
                    'score': 0.95,
                    'title': 'Yazılım Geliştirici',
                    'category': 'IT'
                },
                ...
            ]
        """
        # Embedding'i al
        if item_id not in self.hybrid_embeddings:
            raise ValueError(f"İlan ID {item_id} için embedding bulunamadı")
        
        query_vector = self.hybrid_embeddings[item_id].tolist()
        
        # Filter oluştur (opsiyonel)
        query_filter = None
        if filter_category:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=filter_category)
                    )
                ]
            )
        
        # Qdrant'tan benzer vektörleri ara
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_n + 1,  # +1 çünkü kendisi de dönebilir
            query_filter=query_filter
        )
        
        # Sonuçları formatla
        recommendations = []
        for hit in search_result:
            # Kendisini atla
            if hit.id == item_id:
                continue
            
            recommendations.append({
                'item_id': hit.id,
                'score': hit.score,
                'title': hit.payload.get('title', ''),
                'category': hit.payload.get('category', '')
            })
        
        return recommendations[:top_n]
    
    def batch_recommend(self, item_ids: List[int], top_n: int = 10) -> Dict[int, List[Dict]]:
        """
        Birden fazla ilan için toplu öneri
        
        Args:
            item_ids: İlan ID'leri listesi
            top_n: Her ilan için öneri sayısı
            
        Returns:
            Dictionary {item_id: recommendations_list}
        """
        results = {}
        
        print(f"\n{len(item_ids)} ilan için öneri üretiliyor...")
        for item_id in tqdm(item_ids):
            try:
                recommendations = self.recommend_similar_jobs(item_id, top_n)
                results[item_id] = recommendations
            except Exception as e:
                print(f"  Hata (İlan {item_id}): {e}")
                results[item_id] = []
        
        return results
    
    def get_statistics(self):
        """Sistem istatistiklerini döndürür"""
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        
        stats = {
            'total_embeddings': len(self.hybrid_embeddings),
            'graph_embeddings': len(self.graph_embeddings),
            'text_embeddings': len(self.text_embeddings),
            'qdrant_points': collection_info.points_count,
            'vector_dimension': collection_info.config.params.vectors.size,
            'distance_metric': collection_info.config.params.vectors.distance.name
        }
        
        return stats
    
    def evaluate_recommendations(self, user_events_df, test_size=100):
        """
        Öneri sistemini kullanıcı davranışlarına göre değerlendirir
        
        Metrik: Hit Rate
        - Kullanıcının başvurduğu ilanlar için öneriler üret
        - Bu öneriler arasında kullanıcının daha sonra başvurduğu ilanlar var mı?
        
        Args:
            user_events_df: Kullanıcı hareket verileri
            test_size: Test edilecek örnek sayısı
            
        Returns:
            metrics: Değerlendirme metrikleri
        """
        print("\nÖneri sistemi değerlendirmesi başlıyor...")
        
        # Purchase eventlerini al
        purchases = user_events_df[user_events_df['event_type'] == 'purchase'].copy()
        
        # Her kullanıcının başvurduğu ilanları grupla
        user_purchases = purchases.groupby('client_id')['item_id'].apply(list).to_dict()
        
        # En az 2 başvurusu olan kullanıcıları filtrele
        user_purchases = {k: v for k, v in user_purchases.items() if len(v) >= 2}
        
        if len(user_purchases) == 0:
            print("Yeterli veri yok")
            return {}
        
        # Rastgele kullanıcı seç
        import random
        test_users = random.sample(list(user_purchases.keys()), 
                                   min(test_size, len(user_purchases)))
        
        hits = 0
        total = 0
        
        for user_id in tqdm(test_users):
            user_items = user_purchases[user_id]
            
            # İlk ilandan öneri üret
            query_item = user_items[0]
            target_items = set(user_items[1:])
            
            try:
                recommendations = self.recommend_similar_jobs(query_item, top_n=20)
                recommended_ids = set([r['item_id'] for r in recommendations])
                
                # Hit var mı?
                if len(recommended_ids & target_items) > 0:
                    hits += 1
                
                total += 1
                
            except Exception as e:
                continue
        
        hit_rate = hits / total if total > 0 else 0
        
        print(f"\n{'='*60}")
        print("ÖNERİ SİSTEMİ DEĞERLENDİRME RAPORU")
        print(f"{'='*60}")
        print(f"Test edilen kullanıcı sayısı: {total}")
        print(f"Hit sayısı: {hits}")
        print(f"Hit Rate: {hit_rate:.4f}")
        print(f"{'='*60}")
        
        return {
            'hit_rate': hit_rate,
            'total_tests': total,
            'hits': hits
        }


# Ana fonksiyon: Sistemi kurmak için
def build_recommendation_system(
    user_events_path='user_event_data.csv',
    job_data_path='item_information.csv',
    graph_emb_path='job_embeddings_graph.npz',
    text_emb_path='job_embeddings_text.npz',
    alpha=0.5,
    qdrant_url='localhost',
    qdrant_port=6333
):
    """
    Tam öneri sistemini kurar
    
    Args:
        user_events_path: Kullanıcı hareket verileri
        job_data_path: İlan verileri
        graph_emb_path: Graph embedding dosyası
        text_emb_path: Text embedding dosyası
        alpha: Graph-text dengesi (0-1)
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        
    Returns:
        recommender: HybridJobRecommendationSystem instance
    """
    print("="*60)
    print("İLAN ÖNERİ SİSTEMİ KURULUMU")
    print("="*60)
    
    # 1. Öneri sistemini başlat
    recommender = HybridJobRecommendationSystem(
        qdrant_url=qdrant_url,
        qdrant_port=qdrant_port
    )
    
    # 2. Embedding'leri yükle
    recommender.load_embeddings(graph_emb_path, text_emb_path)
    
    # 3. İlan metadata'larını yükle
    job_data = pd.read_csv(job_data_path)
    recommender.load_job_metadata(job_data)
    
    # 4. Hibrit embedding'ler oluştur
    recommender.create_hybrid_embeddings(alpha=alpha, normalize=True)
    
    # 5. Qdrant'a yükle
    recommender.upload_to_qdrant(batch_size=100)
    
    # 6. İstatistikleri göster
    stats = recommender.get_statistics()
    print(f"\n{'='*60}")
    print("SİSTEM İSTATİSTİKLERİ")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"{'='*60}")
    
    # 7. Değerlendirme (opsiyonel)
    user_events = pd.read_csv(user_events_path)
    recommender.evaluate_recommendations(user_events, test_size=50)
    
    print("\n✓ Öneri sistemi başarıyla kuruldu!")
    
    return recommender


# Kullanım örneği
if __name__ == "__main__":
    # Sistemi kur
    recommender = build_recommendation_system(
    user_events_path='user_event_data.csv',
    job_data_path='item_information.csv',
        graph_emb_path='job_embeddings_graph.npz',
        text_emb_path='job_embeddings_text.npz',
        alpha=0.6,  # %60 graph, %40 text
        qdrant_url='localhost',
        qdrant_port=6333
    )
    
    # Örnek öneri al
    print("\n" + "="*60)
    print("ÖRNEK ÖNERİLER")
    print("="*60)
    
    test_item_id = 123
    recommendations = recommender.recommend_similar_jobs(
        item_id=test_item_id,
        top_n=10
    )
    
    print(f"\nİlan {test_item_id} için öneriler:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. İlan {rec['item_id']}")
        print(f"   Başlık: {rec['title'][:60]}...")
        print(f"   Benzerlik: {rec['score']:.4f}")
        print(f"   Kategori: {rec['category']}")
        print()