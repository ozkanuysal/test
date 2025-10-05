"""
İlan Öneri Sistemi - Ana Pipeline

Bu script tüm sistemi sıfırdan kurar:
1. Graph oluşturur ve analiz eder
2. Graph-based embeddings üretir (Node2Vec)
3. Text-based embeddings üretir (BERTurk)
4. Hibrit embedding sistemi kurar
5. Qdrant'a yükler
6. Sistem değerlendirmesi yapar
"""

import argparse
import os
from src.graph_builder import JobGraphBuilder
from src.node_embedding import JobNodeEmbedding
from src.text_embedding import JobTextEmbedding
from src.recommendation_system import build_recommendation_system
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='İlan Öneri Sistemi Pipeline')
    
    # Veri dosyaları
    parser.add_argument('--user_events', type=str, default='user_event_data.csv',
                       help='Kullanıcı hareket verileri CSV dosyası')
    parser.add_argument('--job_data', type=str, default='item_information.csv',
                       help='İlan verileri CSV dosyası')
    
    # Graph parametreleri
    parser.add_argument('--weight_method', type=str, default='weighted',
                       choices=['co_interaction', 'weighted'],
                       help='Graph edge ağırlık yöntemi')
    
    # Node2Vec parametreleri
    parser.add_argument('--dimensions', type=int, default=128,
                       help='Embedding boyutu')
    parser.add_argument('--walk_length', type=int, default=30,
                       help='Random walk uzunluğu')
    parser.add_argument('--num_walks', type=int, default=200,
                       help='Her node için walk sayısı')
    parser.add_argument('--p', type=float, default=1.0,
                       help='Return parametresi')
    parser.add_argument('--q', type=float, default=0.5,
                       help='In-out parametresi')
    
    # Text embedding parametreleri
    parser.add_argument('--bert_model', type=str, 
                       default='dbmdz/bert-base-turkish-cased',
                       help='BERT model adı')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Text embedding batch size')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['cls', 'mean', 'max'],
                       help='BERT pooling stratejisi')
    
    # Hibrit sistem parametreleri
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Graph embedding ağırlığı (0-1)')
    
    # Qdrant parametreleri
    parser.add_argument('--qdrant_url', type=str, default='localhost',
                       help='Qdrant server URL')
    parser.add_argument('--qdrant_port', type=int, default=6333,
                       help='Qdrant server port')
    
    # Pipeline kontrol
    parser.add_argument('--skip_graph', action='store_true',
                       help='Graph oluşturmayı atla (mevcut graph kullan)')
    parser.add_argument('--skip_node_emb', action='store_true',
                       help='Node embedding oluşturmayı atla')
    parser.add_argument('--skip_text_emb', action='store_true',
                       help='Text embedding oluşturmayı atla')
    
    # Çıktı dosyaları
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Çıktı dosyaları klasörü')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Çıktı klasörünü oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("İLAN ÖNERİ SİSTEMİ - TAM PİPELINE")
    print("="*80)
    
    # Dosya yolları
    graph_path = os.path.join(args.output_dir, 'job_graph.gexf')
    graph_emb_path = os.path.join(args.output_dir, 'job_embeddings_graph.npz')
    text_emb_path = os.path.join(args.output_dir, 'job_embeddings_text.npz')
    
    # =========================================================================
    # PART 1: GRAPH OLUŞTURMA VE ANALİZ
    # =========================================================================
    if not args.skip_graph:
        print("\n" + "="*80)
        print("PART 1: GRAPH OLUŞTURMA VE ANALİZ")
        print("="*80)
        
        builder = JobGraphBuilder(
            user_events_path=args.user_events,
            job_data_path=args.job_data
        )
        
        # Graph'ı kur
        graph = builder.build_graph(weight_method=args.weight_method)
        
        # Graph'ı analiz et
        stats = builder.analyze_graph()
        
        # Görselleştir
        vis_path = os.path.join(args.output_dir, 'graph_visualization.png')
        builder.visualize_graph_sample(num_nodes=50, save_path=vis_path)
        
        # Kaydet
        builder.save_graph(graph_path)
    else:
        print(f"\n[ATLANIYOR] Graph mevcut dosyadan yükleniyor: {graph_path}")
        import networkx as nx
        graph = nx.read_gexf(graph_path)
        print(f"Graph yüklendi: {graph.number_of_nodes()} node, {graph.number_of_edges()} edge")
    
    # =========================================================================
    # PART 2: GRAPH-BASED EMBEDDING (Node2Vec)
    # =========================================================================
    if not args.skip_node_emb:
        print("\n" + "="*80)
        print("PART 2: GRAPH-BASED EMBEDDING (Node2Vec)")
        print("="*80)
        
        embedder = JobNodeEmbedding(graph)
        
        # Embedding'leri eğit
        embeddings = embedder.train_embeddings(
            dimensions=args.dimensions,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            p=args.p,
            q=args.q,
            workers=4
        )
        
        # Test seti oluştur ve değerlendir
        user_events = pd.read_csv(args.user_events)
        test_pairs = embedder.create_test_set(user_events, min_interactions=5)
        metrics = embedder.evaluate_embeddings(test_pairs, threshold=0.5)
        
        # Kaydet
        embedder.save_embeddings(graph_emb_path)
    else:
        print(f"\n[ATLANIYOR] Node embedding mevcut: {graph_emb_path}")
    
    # =========================================================================
    # PART 3: TEXT-BASED EMBEDDING (BERTurk)
    # =========================================================================
    if not args.skip_text_emb:
        print("\n" + "="*80)
        print("PART 3: TEXT-BASED EMBEDDING (BERTurk)")
        print("="*80)
        
        job_data = pd.read_csv(args.job_data)
        
        text_embedder = JobTextEmbedding(model_name=args.bert_model)
        
        # Embedding'leri oluştur
        embeddings = text_embedder.create_embeddings(
            job_data_df=job_data,
            batch_size=args.batch_size,
            title_weight=2,
            pooling=args.pooling
        )
        
        # Kalite değerlendirmesi
        metrics = text_embedder.evaluate_semantic_quality(job_data, sample_size=10)
        
        # Kaydet
        text_embedder.save_embeddings(text_emb_path)
    else:
        print(f"\n[ATLANIYOR] Text embedding mevcut: {text_emb_path}")
    
    # =========================================================================
    # PART 4: HİBRİT ÖNERİ SİSTEMİ
    # =========================================================================
    print("\n" + "="*80)
    print("PART 4: HİBRİT ÖNERİ SİSTEMİ")
    print("="*80)
    
    recommender = build_recommendation_system(
        user_events_path=args.user_events,
        job_data_path=args.job_data,
        graph_emb_path=graph_emb_path,
        text_emb_path=text_emb_path,
        alpha=args.alpha,
        qdrant_url=args.qdrant_url,
        qdrant_port=args.qdrant_port
    )
    
    # =========================================================================
    # ÖRNEK ÖNERİLER
    # =========================================================================
    print("\n" + "="*80)
    print("ÖRNEK ÖNERİLER")
    print("="*80)
    
    # İlk 3 ilan için öneri üret
    job_data = pd.read_csv(args.job_data)
    sample_items = job_data['item_id'].head(3).tolist()
    
    for item_id in sample_items:
        print(f"\n{'='*60}")
        try:
            # İlan bilgilerini al
            job_info = job_data[job_data['item_id'] == item_id].iloc[0]
            print(f"Kaynak İlan ID: {item_id}")
            print(f"Başlık: {job_info['pozisyon_adi'][:60]}...")
            
            # Önerileri al
            recommendations = recommender.recommend_similar_jobs(item_id, top_n=5)
            
            print(f"\nTop 5 Benzer İlan:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. İlan {rec['item_id']}")
                print(f"   Başlık: {rec['title'][:60]}...")
                print(f"   Benzerlik Skoru: {rec['score']:.4f}")
        except Exception as e:
            print(f"Hata: {e}")
    
    print("\n" + "="*80)
    print("PİPELINE TAMAMLANDI!")
    print("="*80)
    print(f"\nÇıktı dosyaları '{args.output_dir}' klasöründe:")
    print(f"  - Graph: {graph_path}")
    print(f"  - Graph Embeddings: {graph_emb_path}")
    print(f"  - Text Embeddings: {text_emb_path}")
    print(f"  - Görselleştirme: {os.path.join(args.output_dir, 'graph_visualization.png')}")
    print(f"\nQdrant Collection: job_recommendations")
    print(f"Qdrant URL: {args.qdrant_url}:{args.qdrant_port}")


if __name__ == "__main__":
    main()