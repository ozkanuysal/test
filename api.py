"""
FastAPI REST API - Öneri Sistemi Endpoint'leri

Production deployment için API örneği.
Kurulum: pip install fastapi uvicorn
Çalıştırma: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from src.recommendation_system import HybridJobRecommendationSystem
import pandas as pd

# FastAPI uygulaması
app = FastAPI(
    title="İlan Öneri Sistemi API",
    description="Graph + Text based hibrit öneri sistemi",
    version="1.0.0"
)

# CORS middleware (frontend için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da spesifik domain'ler belirtin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler
recommender = None
job_data = None

# Pydantic modeller
class RecommendationRequest(BaseModel):
    item_id: int = Field(..., description="İlan ID'si")
    top_n: int = Field(10, ge=1, le=100, description="Öneri sayısı")
    filter_category: Optional[str] = Field(None, description="Kategori filtresi")

class JobRecommendation(BaseModel):
    item_id: int
    title: str
    score: float
    category: Optional[str] = None

class RecommendationResponse(BaseModel):
    item_id: int
    recommendations: List[JobRecommendation]
    total: int

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    total_embeddings: int

class SystemStats(BaseModel):
    total_embeddings: int
    graph_embeddings: int
    text_embeddings: int
    qdrant_points: int
    vector_dimension: int


@app.on_event("startup")
async def startup_event():
    """API başlatma"""
    global recommender, job_data
    
    print("API başlatılıyor...")
    
    # Öneri sistemini yükle
    try:
        recommender = HybridJobRecommendationSystem(
            qdrant_url='localhost',
            qdrant_port=6333
        )
        print("✓ Öneri sistemi yüklendi")
    except Exception as e:
        print(f"⚠ Öneri sistemi yüklenemedi: {e}")
    
    # İlan verilerini yükle
    try:
        job_data = pd.read_csv('item_information.csv')
        print(f"✓ {len(job_data)} ilan yüklendi")
    except Exception as e:
        print(f"⚠ İlan verileri yüklenemedi: {e}")


@app.get("/", tags=["General"])
async def root():
    """Ana endpoint"""
    return {
        "message": "İlan Öneri Sistemi API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Sistem sağlık kontrolü"""
    
    if recommender is None:
        return {
            "status": "unhealthy",
            "qdrant_connected": False,
            "total_embeddings": 0
        }
    
    try:
        stats = recommender.get_statistics()
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "total_embeddings": stats.get('total_embeddings', 0)
        }
    except Exception as e:
        return {
            "status": "degraded",
            "qdrant_connected": False,
            "total_embeddings": 0
        }


@app.get("/stats", response_model=SystemStats, tags=["General"])
async def get_system_stats():
    """Sistem istatistikleri"""
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Öneri sistemi yüklenmedi")
    
    try:
        stats = recommender.get_statistics()
        return SystemStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Belirli bir ilan için öneriler üretir
    
    - **item_id**: Kaynak ilan ID'si
    - **top_n**: Kaç öneri döndürülecek (1-100)
    - **filter_category**: Opsiyonel kategori filtresi
    """
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Öneri sistemi yüklenmedi")
    
    try:
        recommendations = recommender.recommend_similar_jobs(
            item_id=request.item_id,
            top_n=request.top_n,
            filter_category=request.filter_category
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"İlan {request.item_id} için öneri bulunamadı"
            )
        
        return RecommendationResponse(
            item_id=request.item_id,
            recommendations=[JobRecommendation(**rec) for rec in recommendations],
            total=len(recommendations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/{item_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_get(
    item_id: int,
    top_n: int = Query(10, ge=1, le=100, description="Öneri sayısı"),
    category: Optional[str] = Query(None, description="Kategori filtresi")
):
    """
    GET endpoint ile öneriler (URL parametreleri ile)
    """
    
    request = RecommendationRequest(
        item_id=item_id,
        top_n=top_n,
        filter_category=category
    )
    
    return await get_recommendations(request)


@app.get("/jobs/{item_id}", tags=["Jobs"])
async def get_job_info(item_id: int):
    """Belirli bir ilanın detaylarını döndürür"""
    
    if job_data is None:
        raise HTTPException(status_code=503, detail="İlan verileri yüklenmedi")
    
    job = job_data[job_data['item_id'] == item_id]
    
    if job.empty:
        raise HTTPException(status_code=404, detail=f"İlan {item_id} bulunamadı")
    
    job_info = job.iloc[0].to_dict()
    
    return {
        "item_id": int(job_info['item_id']),
        "title": job_info.get('pozisyon_adi', ''),
        "description": job_info.get('item_id_aciklama', ''),
        "category": job_info.get('kategori', None)
    }


@app.get("/jobs", tags=["Jobs"])
async def search_jobs(
    query: str = Query(..., min_length=2, description="Arama sorgusu"),
    limit: int = Query(20, ge=1, le=100, description="Sonuç sayısı")
):
    """İlanlarda arama yapar (başlıkta)"""
    
    if job_data is None:
        raise HTTPException(status_code=503, detail="İlan verileri yüklenmedi")
    
    # Basit string arama
    query_lower = query.lower()
    filtered = job_data[
        job_data['pozisyon_adi'].str.lower().str.contains(query_lower, na=False)
    ]
    
    results = []
    for _, row in filtered.head(limit).iterrows():
        results.append({
            "item_id": int(row['item_id']),
            "title": row.get('pozisyon_adi', ''),
            "description": row.get('item_id_aciklama', '')[:100] + '...'
        })
    
    return {
        "query": query,
        "total": len(results),
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("İLAN ÖNERİ SİSTEMİ API")
    print("="*80)
    print("\nAPI Endpoints:")
    print("  - Swagger UI: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    print("  - Health Check: http://localhost:8000/health")
    print("\n" + "="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)