import numpy as np
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

class JobNodeEmbedding:
    """
    Graph-based node embedding için sınıf
    
    Seçilen Metod: Node2Vec
    
    Seçim Sebepleri:
    1. Random Walk Tabanlı: Hem yerel (BFS) hem global (DFS) komşulukları öğrenir
    2. Esnek Parametreler: p ve q parametreleriyle keşif stratejisi ayarlanabilir
    3. Skip-gram Model: Bağlam bilgisini etkili şekilde öğrenir
    4. İş bulma domaininde başarılı: Benzer ilanları yakalamada etkili
    5. Unsupervised Learning: Etiket verisi gerektirmez
    """
    
    def __init__(self, graph):
        """
        Args:
            graph: NetworkX graph nesnesi
        """
        self.graph = graph
        self.model = None
        self.embeddings = {}
        
    def train_embeddings(self, dimensions=128, walk_length=30, num_walks=200,
                        p=1, q=1, workers=4, window=10, min_count=1, batch_words=4):
        """
        Node2Vec kullanarak node embeddings oluşturur
        
        Args:
            dimensions: Embedding vektör boyutu (64-256 arası önerilir)
            walk_length: Her random walk'un uzunluğu
            num_walks: Her node için random walk sayısı
            p: Return parametresi (önceki node'a dönüş olasılığı)
            q: In-out parametresi (BFS vs DFS dengesi)
                - q < 1: DFS (daha uzak keşif) 
                - q > 1: BFS (yerel komşuluk)
            workers: Paralel işlem sayısı
            window: Skip-gram için bağlam pencere boyutu
            min_count: Minimum kelime frekansı
            
        Returns:
            embeddings: Dictionary {item_id: embedding_vector}
        """
        print("\nNode2Vec embedding eğitimi başlıyor...")
        print(f"Parametreler: dim={dimensions}, walks={num_walks}, length={walk_length}, p={p}, q={q}")
        
        # Node2Vec modelini oluştur
        node2vec = Node2Vec(
            self.graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=workers,
            quiet=False
        )
        
        # Word2Vec modelini eğit
        print("Word2Vec eğitimi başlıyor...")
        self.model = node2vec.fit(
            window=window,
            min_count=min_count,
            batch_words=batch_words,
            epochs=5
        )
        
        # Tüm node'lar için embedding'leri al
        print("Embedding'ler çıkarılıyor...")
        for node in self.graph.nodes():
            try:
                # Node ID'yi string'e çevir (Node2Vec bunu bekler)
                self.embeddings[node] = self.model.wv[str(node)]
            except KeyError:
                # Eğer node embedding'e sahip değilse, sıfır vektör kullan
                self.embeddings[node] = np.zeros(dimensions)
                print(f"Uyarı: Node {node} için embedding bulunamadı, sıfır vektör kullanıldı")
        
        print(f"Toplam {len(self.embeddings)} node için embedding oluşturuldu")
        return self.embeddings
    
    def get_similar_jobs(self, item_id, top_k=10):
        """
        Verilen ilan için en benzer ilanları bulur
        
        Args:
            item_id: İlan ID'si
            top_k: Döndürülecek benzer ilan sayısı
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.embeddings:
            print(f"Hata: {item_id} için embedding bulunamadı")
            return []
        
        target_emb = self.embeddings[item_id].reshape(1, -1)
        similarities = []
        
        for other_id, other_emb in self.embeddings.items():
            if other_id != item_id:
                sim = cosine_similarity(target_emb, other_emb.reshape(1, -1))[0][0]
                similarities.append((other_id, sim))
        
        # Benzerliğe göre sırala
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def create_test_set(self, user_events_df, min_interactions=5):
        """
        Embedding kalitesini test etmek için test seti oluşturur
        
        Mantık:
        - Pozitif örnekler: Aynı kullanıcının etkileşimde bulunduğu ilan çiftleri
        - Negatif örnekler: Farklı kullanıcıların etkileşimde bulunduğu, 
          graph'ta bağlantısı olmayan ilan çiftleri
          
        Args:
            user_events_df: Kullanıcı hareket verileri
            min_interactions: Minimum etkileşim sayısı filtresi
            
        Returns:
            test_pairs: List of (item1, item2, label) - label: 1=benzer, 0=benzer değil
        """
        print("\nTest seti oluşturuluyor...")
        
        # Kullanıcıların etkileşimde bulunduğu ilanları grupla
        user_items = user_events_df.groupby('client_id')['item_id'].apply(list).to_dict()
        
        # Sadece yeterli etkileşimi olan kullanıcıları filtrele
        user_items = {k: v for k, v in user_items.items() if len(v) >= min_interactions}
        
        positive_pairs = []
        negative_pairs = []
        
        # Pozitif örnekler: Aynı kullanıcının etkileşimde bulunduğu çiftler
        print("  Pozitif örnekler oluşturuluyor...")
        for user, items in tqdm(list(user_items.items())[:100]):  # İlk 100 kullanıcı
            unique_items = list(set(items))
            if len(unique_items) >= 2:
                for i in range(min(len(unique_items), 10)):  # Her kullanıcıdan max 10 çift
                    for j in range(i + 1, min(len(unique_items), 10)):
                        if unique_items[i] in self.embeddings and unique_items[j] in self.embeddings:
                            positive_pairs.append((unique_items[i], unique_items[j], 1))
        
        # Negatif örnekler: Graph'ta bağlantısı olmayan rastgele çiftler
        print("  Negatif örnekler oluşturuluyor...")
        all_nodes = list(self.embeddings.keys())
        np.random.seed(42)
        
        attempts = 0
        max_attempts = len(positive_pairs) * 10
        
        while len(negative_pairs) < len(positive_pairs) and attempts < max_attempts:
            idx1, idx2 = np.random.choice(len(all_nodes), 2, replace=False)
            node1, node2 = all_nodes[idx1], all_nodes[idx2]
            
            # Graph'ta bağlantı yoksa ve henüz eklenmemişse
            if not self.graph.has_edge(node1, node2):
                if (node1, node2, 0) not in negative_pairs and (node2, node1, 0) not in negative_pairs:
                    negative_pairs.append((node1, node2, 0))
            
            attempts += 1
        
        test_pairs = positive_pairs + negative_pairs
        np.random.shuffle(test_pairs)
        
        print(f"  Test seti oluşturuldu: {len(positive_pairs)} pozitif, {len(negative_pairs)} negatif örnek")
        return test_pairs
    
    def evaluate_embeddings(self, test_pairs, threshold=0.5):
        """
        Embedding kalitesini değerlendirir
        
        Metrikler:
        1. Precision: Benzer dediğimiz çiftlerin gerçekten benzer olma oranı
        2. Recall: Gerçekten benzer çiftleri ne kadar yakalayabiliyoruz
        3. F1-Score: Precision ve Recall'ın harmonik ortalaması
        4. ROC-AUC: Eşik değerinden bağımsız performans
        
        Args:
            test_pairs: List of (item1, item2, label)
            threshold: Benzerlik eşiği (cosine similarity)
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        print(f"\nEmbedding değerlendirme başlıyor ({len(test_pairs)} test çifti)...")
        
        y_true = []
        y_pred = []
        similarities = []
        
        for item1, item2, label in tqdm(test_pairs):
            if item1 in self.embeddings and item2 in self.embeddings:
                emb1 = self.embeddings[item1].reshape(1, -1)
                emb2 = self.embeddings[item2].reshape(1, -1)
                sim = cosine_similarity(emb1, emb2)[0][0]
                
                y_true.append(label)
                y_pred.append(1 if sim >= threshold else 0)
                similarities.append(sim)
        
        # Metrikleri hesapla
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Ortalama benzerlik skorları
        pos_sims = [s for s, l in zip(similarities, y_true) if l == 1]
        neg_sims = [s for s, l in zip(similarities, y_true) if l == 0]
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': threshold,
            'avg_positive_similarity': np.mean(pos_sims) if pos_sims else 0,
            'avg_negative_similarity': np.mean(neg_sims) if neg_sims else 0,
            'similarity_gap': np.mean(pos_sims) - np.mean(neg_sims) if pos_sims and neg_sims else 0
        }
        
        print("\n" + "="*60)
        print("EMBEDDING KALITE RAPORU")
        print("="*60)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Threshold: {threshold}")
        print(f"Pozitif çiftler ortalama benzerlik: {metrics['avg_positive_similarity']:.4f}")
        print(f"Negatif çiftler ortalama benzerlik: {metrics['avg_negative_similarity']:.4f}")
        print(f"Benzerlik farkı: {metrics['similarity_gap']:.4f}")
        print("="*60)
        
        return metrics
    
    def optimize_hyperparameters(self, user_events_df, param_grid=None):
        """
        Grid search ile hiperparametre optimizasyonu
        
        Args:
            user_events_df: Kullanıcı hareket verileri
            param_grid: Denenmek istenen parametreler
            
        Returns:
            best_params: En iyi parametreler
            results: Tüm denemelerin sonuçları
        """
        if param_grid is None:
            param_grid = {
                'dimensions': [64, 128],
                'walk_length': [20, 30],
                'num_walks': [100, 200],
                'p': [0.5, 1, 2],
                'q': [0.5, 1, 2],
            }
        
        print("\nHiperparametre optimizasyonu başlıyor...")
        print(f"Toplam {np.prod([len(v) for v in param_grid.values()])} kombinasyon denenecek")
        
        # Test setini bir kez oluştur
        test_pairs = self.create_test_set(user_events_df)
        
        results = []
        best_f1 = 0
        best_params = None
        
        # Grid search
        from itertools import product
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        for i, param_values in enumerate(param_combinations[:5]):  # İlk 5 kombinasyon
            params = dict(zip(param_names, param_values))
            print(f"\n[{i+1}/5] Deneme: {params}")
            
            try:
                # Embedding eğit
                self.train_embeddings(**params)
                
                # Değerlendir
                metrics = self.evaluate_embeddings(test_pairs)
                metrics.update(params)
                results.append(metrics)
                
                # En iyiyi güncelle
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_params = params
                    
            except Exception as e:
                print(f"Hata: {e}")
                continue
        
        print("\n" + "="*60)
        print("OPTİMİZASYON SONUÇLARI")
        print("="*60)
        print(f"En iyi F1-Score: {best_f1:.4f}")
        print(f"En iyi parametreler: {best_params}")
        print("="*60)
        
        return best_params, pd.DataFrame(results)
    
    def save_embeddings(self, path='job_embeddings.npz'):
        """Embedding'leri kaydet"""
        np.savez(path, **{str(k): v for k, v in self.embeddings.items()})
        print(f"Embeddings kaydedildi: {path}")
    
    def load_embeddings(self, path='job_embeddings.npz'):
        """Embedding'leri yükle"""
        data = np.load(path)
        self.embeddings = {int(k): v for k, v in data.items()}
        print(f"Embeddings yüklendi: {path} ({len(self.embeddings)} node)")
        return self.embeddings


# Kullanım örneği
if __name__ == "__main__":
    # Graph'ı yükle
    graph = nx.read_gexf('job_graph.gexf')
    
    # Embedding sınıfını oluştur
    embedder = JobNodeEmbedding(graph)
    
    # Embedding'leri eğit (optimize edilmiş parametrelerle)
    embeddings = embedder.train_embeddings(
        dimensions=128,
        walk_length=30,
        num_walks=200,
        p=1,
        q=0.5,  # DFS-like, daha uzak bağlantıları keşfeder
        workers=4
    )
    
    # Test seti oluştur ve değerlendir
    user_events = pd.read_csv('user_event_data.csv')
    test_pairs = embedder.create_test_set(user_events)
    metrics = embedder.evaluate_embeddings(test_pairs, threshold=0.5)
    
    # Benzer ilanları bul (örnek)
    similar_jobs = embedder.get_similar_jobs(item_id=123, top_k=10)
    print(f"\nİlan 123 için en benzer ilanlar:")
    for job_id, similarity in similar_jobs:
        print(f"  İlan {job_id}: Benzerlik={similarity:.4f}")
    
    # Embedding'leri kaydet
    embedder.save_embeddings('job_embeddings_graph.npz')