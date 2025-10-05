"""
Öneri Sistemini Test Etme Script'i

Bu script kurulmuş sistemi test eder ve örnek öneriler üretir.
"""

from src.recommendation_system import HybridJobRecommendationSystem
import pandas as pd

def test_recommendations():
    """Öneri sistemini test eder"""
    
    print("="*80)
    print("ÖNERİ SİSTEMİ TEST SCRIPT'İ")
    print("="*80)
    
    # Sistem bağlan
    print("\nQdrant'a bağlanılıyor...")
    recommender = HybridJobRecommendationSystem(
        qdrant_url='localhost',
        qdrant_port=6333,
        collection_name='job_recommendations'
    )
    
    # İstatistikleri göster
    print("\nSistem İstatistikleri:")
    try:
        stats = recommender.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Hata: {e}")
        print("  Not: Embedding'ler yüklenmemiş olabilir")
    
    # İlan verilerini yükle
    print("\nİlan verileri yükleniyor...")
    try:
        job_data = pd.read_csv('item_information.csv')
        print(f"  {len(job_data)} ilan yüklendi")
    except FileNotFoundError:
        print("  HATA: item_information.csv bulunamadı!")
        return
    
    # Test ilanları seç
    test_items = job_data['item_id'].head(5).tolist()
    
    print(f"\n{len(test_items)} test ilanı için öneriler üretiliyor...\n")
    
    # Her test ilanı için öneri al
    for i, item_id in enumerate(test_items, 1):
        print("="*80)
        print(f"TEST {i}/{len(test_items)}")
        print("="*80)
        
        try:
            # İlan bilgilerini göster
            job_info = job_data[job_data['item_id'] == item_id].iloc[0]
            print(f"\nKaynak İlan:")
            print(f"  ID: {item_id}")
            print(f"  Başlık: {job_info['pozisyon_adi']}")
            if len(job_info.get('item_id_aciklama', '')) > 0:
                desc = str(job_info['item_id_aciklama'])[:100]
                print(f"  Açıklama: {desc}...")
            
            # Önerileri al
            recommendations = recommender.recommend_similar_jobs(
                item_id=item_id,
                top_n=5
            )
            
            if not recommendations:
                print("\n  ⚠️  Öneri bulunamadı!")
                continue
            
            print(f"\n  Top 5 Benzer İlan:")
            for j, rec in enumerate(recommendations, 1):
                print(f"\n  {j}. İlan {rec['item_id']}")
                print(f"     Başlık: {rec['title'][:60]}")
                print(f"     Benzerlik Skoru: {rec['score']:.4f}")
                if rec.get('category'):
                    print(f"     Kategori: {rec['category']}")
            
        except Exception as e:
            print(f"\n  ❌ Hata: {e}")
        
        print()
    
    print("="*80)
    print("TEST TAMAMLANDI")
    print("="*80)


def test_batch_recommendations():
    """Toplu öneri testi"""
    
    print("\n" + "="*80)
    print("TOPLU ÖNERİ TESTİ")
    print("="*80)
    
    # Sistem bağlan
    recommender = HybridJobRecommendationSystem(
        qdrant_url='localhost',
        qdrant_port=6333
    )
    
    # Test ilanları
    job_data = pd.read_csv('item_information.csv')
    test_items = job_data['item_id'].head(10).tolist()
    
    print(f"\n{len(test_items)} ilan için toplu öneri üretiliyor...")
    
    results = recommender.batch_recommend(
        item_ids=test_items,
        top_n=5
    )
    
    # Özet istatistik
    total_recs = sum(len(recs) for recs in results.values())
    avg_recs = total_recs / len(results) if results else 0
    
    print(f"\nSonuçlar:")
    print(f"  Toplam test ilan: {len(test_items)}")
    print(f"  Başarılı öneri: {len(results)}")
    print(f"  Toplam öneri sayısı: {total_recs}")
    print(f"  Ortalama öneri/ilan: {avg_recs:.2f}")


def test_similarity_scores():
    """Benzerlik skorlarını analiz eder"""
    
    print("\n" + "="*80)
    print("BENZERLİK SKORU ANALİZİ")
    print("="*80)
    
    recommender = HybridJobRecommendationSystem(
        qdrant_url='localhost',
        qdrant_port=6333
    )
    
    job_data = pd.read_csv('item_information.csv')
    test_items = job_data['item_id'].head(20).tolist()
    
    all_scores = []
    
    for item_id in test_items:
        try:
            recs = recommender.recommend_similar_jobs(item_id, top_n=10)
            scores = [r['score'] for r in recs]
            all_scores.extend(scores)
        except:
            continue
    
    if all_scores:
        import numpy as np
        print(f"\nBenzerlik Skor İstatistikleri:")
        print(f"  Toplam öneri: {len(all_scores)}")
        print(f"  Ortalama skor: {np.mean(all_scores):.4f}")
        print(f"  Medyan skor: {np.median(all_scores):.4f}")
        print(f"  Min skor: {np.min(all_scores):.4f}")
        print(f"  Max skor: {np.max(all_scores):.4f}")
        print(f"  Std skor: {np.std(all_scores):.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Öneri sistemi test script')
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'batch', 'analysis', 'all'],
                       help='Test modu')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'basic' or args.mode == 'all':
            test_recommendations()
        
        if args.mode == 'batch' or args.mode == 'all':
            test_batch_recommendations()
        
        if args.mode == 'analysis' or args.mode == 'all':
            test_similarity_scores()
            
    except KeyboardInterrupt:
        print("\n\nTest iptal edildi.")
    except Exception as e:
        print(f"\n\nBEKLENMEYEN HATA: {e}")
        import traceback
        traceback.print_exc()