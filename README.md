# Báo cáo ngắn — LAB 16: CPU Baseline (LightGBM) trên GCP

**MSSV:** 2A202600917 ·**Sinh viên:** NGUYEN THI BAO TRAN · **Project GCP:** `august-lamp-499804-h3` · **Ngày:** 18/06/2026
**Máy ảo:** `ai-gpu-node` — `e2-standard-8` (8 vCPU / 32 GB) · zone `us-central1-c`
**Bài toán:** Credit Card Fraud Detection (Kaggle `mlg-ulb/creditcardfraud`, 284,807 giao dịch, 492 gian lận = 0.173%).

## 1. Vì sao dùng CPU thay vì GPU?
- **Quota GPU:** GCP khóa quota NVIDIA T4 ở mức 0 cho tài khoản mới/Free Tier; ngoài ra khi triển khai còn gặp lỗi hết tài nguyên (`n2-standard-8` stockout ở cả 3 zone us-central1), nên chuyển sang `e2-standard-8`.
- **Bản chất bài toán:** LightGBM là gradient boosting trên **dữ liệu bảng (tabular)** — thuật toán tối ưu cho CPU đa nhân. GPU hầu như **không tăng tốc** loại workload này (thậm chí chậm hơn với dataset nhỏ do overhead copy dữ liệu CPU↔GPU). GPU chỉ thực sự đáng giá cho deep learning / LLM inference.

## 2. Kết quả benchmark (đo thực tế trên `e2-standard-8`)
| Chỉ số | Giá trị |
|---|---|
| Thời gian load data | 1.76 s |
| **Thời gian training** | **4.42 s** (early stopping ở cây 213) |
| **AUC-ROC** | **0.9787** |
| AUC-PR (average precision) | 0.8693 |
| F1 / Precision / Recall | 0.865 / 0.920 / 0.816 |
| Accuracy | 0.99956 |
| **Inference latency (1 dòng)** | **0.77 ms** |
| **Inference throughput (1000 dòng)** | **~283,000 dòng/s** |

## 3. So sánh CPU vs GPU
- **Training time:** 4.4s trên 8 vCPU là rất nhanh cho 227k mẫu × 30 đặc trưng. Với dataset cỡ này, GPU không rút ngắn được đáng kể vì thời gian chủ yếu nằm ở xây histogram/split — vốn đã song song hóa tốt trên CPU; overhead truyền dữ liệu sang GPU thường khiến tổng thời gian *bằng hoặc lâu hơn*.
- **Độ chính xác (AUC):** Chất lượng mô hình **không phụ thuộc** vào CPU hay GPU — cùng thuật toán cho cùng AUC ≈ 0.98. Phần cứng chỉ ảnh hưởng tốc độ.
- **Inference speed:** 0.77 ms/dòng và ~283k dòng/s trên CPU đã thừa cho hệ thống phát hiện gian lận realtime; GPU là dư thừa và tốn kém hơn.

## 4. Chi phí
| Phương án | Cấu hình | Chi phí/giờ |
|---|---|---|
| **CPU (đang dùng)** | `e2-standard-8` + NAT + LB | **~$0.32** |
| GPU | `n1-standard-4` + 1× T4 + NAT + LB | ~$0.54 |

→ Với workload tabular ML, **CPU rẻ hơn ~40% và hiệu năng tương đương** về thời gian — đây là bài học chọn hạ tầng đúng theo workload thay vì mặc định chọn GPU.

## 5. Kết luận
Chọn `e2-standard-8` (CPU) cho bài toán LightGBM là quyết định hợp lý cả về **chi phí lẫn hiệu năng**: train 4.4s, AUC-ROC 0.9787, inference < 1 ms/dòng, rẻ hơn phương án GPU mà không hề thua kém về tốc độ hay độ chính xác cho dữ liệu bảng.
