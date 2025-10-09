# Cost Analysis

## Cloud Deployment Cost Breakdown

### AWS Infrastructure Costs (Monthly)

#### Compute - GPU Instances

**Development Pool** (2 GPUs):
- Instance Type: `p3.8xlarge` (4x V100 GPUs, 32 vCPU, 244 GB RAM)
- On-Demand: $12.24/hour × 1 instance × 730 hours = **$8,935/month**
- Reserved (1-year): ~$6,250/month (-30%)
- Spot: ~$3,672/month (-60%)

**Production Pool** (8 GPUs):
- Instance Type: `p4d.24xlarge` (8x A100 GPUs, 96 vCPU, 1152 GB RAM)
- On-Demand: $32.77/hour × 1 instance × 730 hours = **$23,922/month**
- Reserved (1-year): ~$16,745/month (-30%)
- Spot: ~$9,569/month (-60%)

**Worker Coordination**:
- Instance Type: `c5.2xlarge` (8 vCPU, 16 GB RAM)
- Cost: $0.34/hour × 2 instances × 730 hours = **$496/month**

**Subtotal Compute**:
- On-Demand: $33,353/month
- Mixed (Prod Reserved + Dev Spot): $23,411/month **← Recommended**

#### Storage

**EBS Volumes** (SSD):
- 2 TB per GPU instance × $0.10/GB = **$200/month** per instance
- Total (2 instances): **$400/month**

**S3 Storage** (datasets, checkpoints):
- Standard: 5 TB × $0.023/GB = **$115/month**
- Infrequent Access (old checkpoints): 10 TB × $0.0125/GB = **$125/month**

**S3 Data Transfer**:
- Out to internet: 1 TB/month × $0.09/GB = **$90/month**

**Subtotal Storage**: **$730/month**

#### Networking

**Data Transfer**:
- Inter-AZ: 500 GB × $0.01/GB = **$5/month**
- VPC Endpoints (S3, CloudWatch): **$7.20/month**

**Subtotal Networking**: **$12/month**

#### Database & Caching

**Redis** (ElastiCache):
- Instance Type: `cache.r6g.xlarge` (4 vCPU, 26.32 GB)
- Cost: $0.252/hour × 730 hours = **$184/month**
- Multi-AZ: +100% = **$368/month**

#### Monitoring

**CloudWatch**:
- Metrics: 500 custom metrics × $0.30 = **$150/month**
- Logs: 100 GB × $0.50/GB = **$50/month**
- Alarms: 50 alarms × $0.10 = **$5/month**

**Subtotal Monitoring**: **$205/month**

### **Total AWS Cost (Monthly)**

| Configuration | Cost |
|---------------|------|
| On-Demand | $34,668/month |
| Optimized (Reserved + Spot) | **$24,726/month** |
| Annual Optimized | **$296,712/year** |
