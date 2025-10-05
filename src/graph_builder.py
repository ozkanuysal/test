import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

class JobGraphBuilder:
    """
    İlan öneri sistemi için Graph oluşturma ve analiz sınıfı
    """
    
    def __init__(self, user_events_path, job_data_path):
        """
        Args:
            user_events_path: Kullanıcı hareket verileri CSV dosya yolu
            job_data_path: İlan verileri CSV dosya yolu
        """
        self.user_events = pd.read_csv(user_events_path)
        self.job_data = pd.read_csv(job_data_path)
        self.graph = nx.Graph()
        
    def build_graph(self, weight_method='co_interaction'):
        """
        Kullanıcı etkileşimlerinden graph oluşturur
        
        Graph Kurulum Mantığı:
        - Node'lar: Her bir ilan (item_id) bir node olarak temsil edilir
        - Edge'ler: İki ilan arasında edge oluşturma kriterleri:
          1. Aynı kullanıcı tarafından etkileşim (click veya purchase)
          2. Aynı arama oturumunda (ds_search_id) görüntüleme
          3. Edge ağırlığı: Ortak etkileşim sayısı
        
        Sebepler:
        - Collaborative Filtering mantığı: Aynı kullanıcının ilgilendiği ilanlar benzerdir
        - Session-based yaklaşım: Aynı aramada görülen ilanlar ilişkilidir
        - Ağırlıklı yapı: Daha çok ortak etkileşim = daha güçlü benzerlik
        
        Args:
            weight_method: Ağırlık hesaplama yöntemi
                - 'co_interaction': Ortak etkileşim sayısı
                - 'weighted': Purchase'a daha fazla ağırlık verir
        """
        print("Graph oluşturuluyor...")
        
        # Tüm ilanları node olarak ekle
        all_items = self.job_data['item_id'].unique()
        self.graph.add_nodes_from(all_items)
        
        # Node attribute olarak ilan bilgilerini ekle
        for _, row in self.job_data.iterrows():
            if row['item_id'] in self.graph.nodes():
                self.graph.nodes[row['item_id']]['title'] = row.get('pozisyon_adi', '')
                self.graph.nodes[row['item_id']]['description'] = row.get('item_id_aciklama', '')
        
        edge_weights = {}
        
        # 1. Kullanıcı bazlı ortak etkileşimler
        print("  - Kullanıcı bazlı kenarlar oluşturuluyor...")
        user_groups = self.user_events.groupby('client_id')['item_id'].apply(list)
        
        for user_items in user_groups:
            # Kullanıcının etkileşimde bulunduğu her ilan çifti için edge oluştur
            for i in range(len(user_items)):
                for j in range(i + 1, len(user_items)):
                    item1, item2 = sorted([user_items[i], user_items[j]])
                    edge = (item1, item2)
                    edge_weights[edge] = edge_weights.get(edge, 0) + 1
        
        # 2. Arama oturumu bazlı ortak görüntülemeler
        print("  - Arama oturumu bazlı kenarlar oluşturuluyor...")
        search_groups = self.user_events.groupby('ds_search_id')['item_id'].apply(list)
        
        for search_items in search_groups:
            # Aynı aramada görülen her ilan çifti için edge oluştur
            for i in range(len(search_items)):
                for j in range(i + 1, len(search_items)):
                    item1, item2 = sorted([search_items[i], search_items[j]])
                    edge = (item1, item2)
                    # Arama bazlı ilişkiye daha düşük ağırlık ver
                    edge_weights[edge] = edge_weights.get(edge, 0) + 0.5
        
        # 3. Purchase'lara ekstra ağırlık (opsiyonel)
        if weight_method == 'weighted':
            print("  - Purchase'lara ekstra ağırlık ekleniyor...")
            purchase_events = self.user_events[self.user_events['event_type'] == 'purchase']
            purchase_groups = purchase_events.groupby('client_id')['item_id'].apply(list)
            
            for user_items in purchase_groups:
                for i in range(len(user_items)):
                    for j in range(i + 1, len(user_items)):
                        item1, item2 = sorted([user_items[i], user_items[j]])
                        edge = (item1, item2)
                        # Purchase ilişkisine ekstra ağırlık
                        edge_weights[edge] = edge_weights.get(edge, 0) + 1.5
        
        # Edge'leri graph'a ekle
        for (item1, item2), weight in edge_weights.items():
            if item1 in self.graph.nodes() and item2 in self.graph.nodes():
                self.graph.add_edge(item1, item2, weight=weight)
        
        print(f"Graph oluşturuldu: {self.graph.number_of_nodes()} node, {self.graph.number_of_edges()} edge")
        return self.graph
    
    def analyze_graph(self):
        """
        Graph'ı detaylı analiz eder ve istatistikler üretir
        """
        print("\n" + "="*60)
        print("GRAPH ANALİZ RAPORU")
        print("="*60)
        
        # Temel istatistikler
        print(f"\n1. TEMEL İSTATİSTİKLER:")
        print(f"   - Toplam node sayısı: {self.graph.number_of_nodes()}")
        print(f"   - Toplam edge sayısı: {self.graph.number_of_edges()}")
        print(f"   - Graph yoğunluğu: {nx.density(self.graph):.6f}")
        
        # Bağlantı durumu
        print(f"\n2. BAĞLANTILIK ANALİZİ:")
        if nx.is_connected(self.graph):
            print("   - Graph bağlantılı (connected)")
            print(f"   - Ortalama en kısa yol uzunluğu: {nx.average_shortest_path_length(self.graph):.2f}")
            print(f"   - Diameter (en uzak node mesafesi): {nx.diameter(self.graph)}")
        else:
            print("   - Graph bağlantılı değil (disconnected)")
            components = list(nx.connected_components(self.graph))
            print(f"   - Bağlantılı bileşen sayısı: {len(components)}")
            print(f"   - En büyük bileşen boyutu: {len(max(components, key=len))}")
            print(f"   - İzole node sayısı: {len([c for c in components if len(c) == 1])}")
        
        # Derece (degree) analizi
        print(f"\n3. DERECE (DEGREE) ANALİZİ:")
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        print(f"   - Ortalama derece: {np.mean(degree_values):.2f}")
        print(f"   - Medyan derece: {np.median(degree_values):.2f}")
        print(f"   - Maksimum derece: {np.max(degree_values)}")
        print(f"   - Minimum derece: {np.min(degree_values)}")
        
        # En yüksek dereceli node'lar (hub'lar)
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n   En bağlantılı 5 ilan (Hub'lar):")
        for item_id, degree in top_nodes:
            title = self.graph.nodes[item_id].get('title', 'N/A')
            print(f"   - İlan {item_id} ({title[:50]}...): {degree} bağlantı")
        
        # Ağırlık analizi
        print(f"\n4. EDGE AĞIRLIK ANALİZİ:")
        weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        print(f"   - Ortalama edge ağırlığı: {np.mean(weights):.2f}")
        print(f"   - Medyan edge ağırlığı: {np.median(weights):.2f}")
        print(f"   - Maksimum edge ağırlığı: {np.max(weights):.2f}")
        
        # Kümelenme katsayısı
        print(f"\n5. KÜMELENME ANALİZİ:")
        clustering_coef = nx.average_clustering(self.graph, weight='weight')
        print(f"   - Ortalama clustering coefficient: {clustering_coef:.4f}")
        print(f"   - Yorum: {self._interpret_clustering(clustering_coef)}")
        
        # Centralite analizi
        print(f"\n6. CENTRALITE ANALİZİ:")
        print("   Betweenness centrality hesaplanıyor...")
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   En yüksek betweenness centrality'ye sahip ilanlar:")
        for item_id, score in top_betweenness:
            title = self.graph.nodes[item_id].get('title', 'N/A')
            print(f"   - İlan {item_id} ({title[:50]}...): {score:.4f}")
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': np.mean(degree_values),
            'clustering_coef': clustering_coef,
            'is_connected': nx.is_connected(self.graph)
        }
    
    def _interpret_clustering(self, coef):
        """Clustering coefficient yorumlama"""
        if coef < 0.1:
            return "Düşük - İlanlar arası zayıf kümelenme, çeşitli kullanıcı davranışları"
        elif coef < 0.3:
            return "Orta - Makul kümelenme, bazı ilan grupları oluşmuş"
        else:
            return "Yüksek - Güçlü kümelenme, belirgin ilan kümeleri var"
    
    def visualize_graph_sample(self, num_nodes=50, save_path='graph_visualization.png'):
        """
        Graph'ın örnek bir kısmını görselleştirir
        """
        # En yüksek dereceli node'ları seç
        degrees = dict(self.graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:num_nodes]
        top_node_ids = [node for node, _ in top_nodes]
        
        # Subgraph oluştur
        subgraph = self.graph.subgraph(top_node_ids)
        
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Node boyutlarını dereceye göre ayarla
        node_sizes = [degrees[node] * 20 for node in subgraph.nodes()]
        
        # Edge kalınlıklarını ağırlığa göre ayarla
        edges = subgraph.edges()
        weights = [subgraph[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7)
        nx.draw_networkx_edges(subgraph, pos, width=weights, alpha=0.3)
        nx.draw_networkx_labels(subgraph, pos, font_size=6)
        
        plt.title(f'Job Recommendation Graph (Top {num_nodes} nodes)', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nGörselleştirme kaydedildi: {save_path}")
        plt.close()
    
    def get_graph(self):
        """Graph nesnesini döndürür"""
        return self.graph
    
    def save_graph(self, path='job_graph.gexf'):
        """Graph'ı dosyaya kaydeder"""
        nx.write_gexf(self.graph, path)
        print(f"Graph kaydedildi: {path}")
    
    def load_graph(self, path='job_graph.gexf'):
        """Graph'ı dosyadan yükler"""
        self.graph = nx.read_gexf(path)
        print(f"Graph yüklendi: {path}")
        return self.graph


# Kullanım örneği
if __name__ == "__main__":
    # Graph builder oluştur
    builder = JobGraphBuilder(
    user_events_path='user_event_data.csv',
        job_data_path='item_information.csv'
    )
    
    # Graph'ı kur
    graph = builder.build_graph(weight_method='weighted')
    
    # Graph'ı analiz et
    stats = builder.analyze_graph()
    
    # Görselleştir
    builder.visualize_graph_sample(num_nodes=50)
    
    # Kaydet
    builder.save_graph('job_graph.gexf')