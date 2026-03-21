📊 HR Analytics Project – Employee Attrition Prediction
📌 1. Tổng quan dự án

Trong doanh nghiệp, nghỉ việc (Attrition) là một vấn đề lớn vì:

Tốn chi phí tuyển dụng lại
Mất nhân sự có kinh nghiệm
Ảnh hưởng đến hiệu suất tổ chức

👉 Vì vậy, dự án này tập trung vào:

🎯 Mục tiêu chính:
Phân tích nguyên nhân khiến nhân viên nghỉ việc
Dự đoán ai có nguy cơ nghỉ
Đưa ra gợi ý chính sách HR dựa trên dữ liệu
📂 2. Dataset & Bài toán
Dataset: HR Analytics (Kaggle)
Kích thước: 1480 nhân viên × 38 thuộc tính
📊 Phân bố mục tiêu:
1242 Stay (~84%)
238 Leave (~16%)

👉 Đây là bài toán mất cân bằng dữ liệu (imbalanced classification)
→ Nếu không xử lý, model sẽ thiên về dự đoán “Stay”

🧹 3. Data Preprocessing
Các bước xử lý:
Làm sạch dữ liệu
Không có missing lớn → giữ nguyên cấu trúc
Encode dữ liệu
Convert categorical → số (LabelEncoder)
Chuẩn hóa
StandardScaler giúp model học tốt hơn
Xử lý imbalance
Dùng class_weight = balanced

👉 Ý nghĩa:

Model “quan tâm hơn” đến nhóm Leave (thiểu số)
⚙️ 4. Feature Engineering (Rất quan trọng)

Thay vì dùng raw data, bạn đã tạo thêm 6 đặc trưng mới:

📌 Ví dụ:
IncomePerYearExp → đo hiệu quả lương theo kinh nghiệm
PromotionGap → số năm chưa được thăng chức
SatisfactionIndex → tổng hợp mức độ hài lòng

👉 Đây là bước tăng sức mạnh model mạnh nhất trong pipeline

🔍 Feature Selection (SelectKBest)

Chọn ra 20 features quan trọng nhất

🔥 Top features:
OverTime (quan trọng nhất)
TotalWorkingYears
JobLevel
MonthlyIncome
SatisfactionIndex

👉 Ý nghĩa:

Giảm nhiễu
Tăng tốc model
Tăng độ chính xác
📊 5. EDA – Phân tích dữ liệu
Insight chính:
🔥 OverTime = yếu tố mạnh nhất
💰 Lương thấp → nghỉ cao
👴 Tuổi lớn → ít nghỉ hơn

👉 Đây là insight thực tế rất giá trị cho HR

🔗 6. Association Rules (Luật kết hợp)
Mục tiêu:

Tìm pattern dạng:

“Nếu A xảy ra → khả năng cao B xảy ra”

📌 Ví dụ thực tế:
OverTime_No → Stay (Confidence ~0.896)
Income cao → Stay

👉 Diễn giải:

Nhân viên không làm OT gần như chắc chắn ở lại
Lương cao → giữ chân tốt
🧠 Ý nghĩa cho HR:
Giảm OT = giảm nghỉ việc
Tăng lương hợp lý = giữ người
🤖 7. Classification (Dự đoán nghỉ việc)
🎯 Mục tiêu:

Dự đoán:

Nhân viên này có nghỉ không?

🔹 Logistic Regression
AUC: 0.85
Recall (Leave): 0.79

👉 Rất tốt để:

Phát hiện người có nguy cơ nghỉ
🌲 Random Forest
AUC: 0.86
Recall thấp hơn

👉 Học pattern tốt nhưng:

Bỏ sót nhiều người nghỉ
⚡ Gradient Boosting (BEST)
AUC: 0.87
Accuracy cao nhất

👉 Model tốt nhất overall

📌 Kết luận:
Model	Dùng khi
Logistic	Cần bắt hết người nghỉ
GBM	Cần độ chính xác tổng thể
📊 8. Clustering (Phân cụm nhân viên)
Mục tiêu:

Không cần label → vẫn hiểu dữ liệu

Kết quả:
4 cụm (k=4)
Cluster 3: attrition ~22% (cao nhất)

👉 Đây là:

“Nhóm nhân viên rủi ro cao”

🧠 Ý nghĩa:

HR có thể:

Target đúng nhóm nguy hiểm
Không cần xử lý toàn bộ nhân viên
🧠 9. Explainability (Giải thích model)
Top yếu tố ảnh hưởng:
OverTime
StockOptionLevel
SatisfactionIndex
MonthlyIncome
🔥 Insight:
OT ↑ → nghỉ ↑
Stock option ↑ → nghỉ ↓

👉 Đây là phần rất quan trọng khi demo
(vì business cần hiểu model)

🔄 10. Semi-Supervised Learning
Vấn đề thực tế:
HR không thể label toàn bộ dữ liệu (tốn chi phí)
Giải pháp:
Dùng ít label + tận dụng data chưa label
📊 Kết quả:
Label %	Best Model	AUC
10%	Label Spread	0.71
20%	Label Spread	0.76
50%	Label Spread	0.81
❌ Self-Training:
Luôn kém hơn

👉 Vì:

Data imbalance → bias
✅ Kết luận:
≥20% label → dùng được
≥50% → gần model full
🧾 11. Regression (Dự đoán Satisfaction)
Mục tiêu:

Dự đoán:

Mức độ hài lòng (1–4)

⚠️ Data Leakage

Phát hiện:

SatisfactionIndex → chứa thông tin target

👉 Đã loại bỏ → đúng chuẩn ML

📊 Kết quả:
Best: Lasso
MAE ≈ 1.0

👉 Insight:

Rất khó dự đoán satisfaction
→ yếu tố này mang tính cảm xúc
🏁 12. Tổng kết Pipeline
🚀 Full pipeline:
Data
Preprocessing
Feature Engineering
EDA
Association Rules
Clustering
Classification
Explainability
Semi-supervised
Regression
📊 📦 Output
12 hình ảnh:
fig0 → fig11
💡 Business Insights (QUAN TRỌNG NHẤT)
🔥 1. OverTime là yếu tố nguy hiểm nhất

→ Giảm OT = giảm nghỉ việc

💰 2. Lương & Stock Option

→ Công cụ giữ người hiệu quả

👥 3. Phân nhóm nhân viên

→ Tập trung vào nhóm risk cao (Cluster 3)

🧠 4. Thiếu dữ liệu vẫn làm được ML

→ Semi-supervised giúp tiết kiệm chi phí

🚀 Hướng phát triển
XGBoost / LightGBM
Dashboard HR
Deploy web app
SHAP explainability
✅ KẾT LUẬN

👉 Project này không chỉ là ML mà còn:

✔️ Phân tích dữ liệu
✔️ Dự đoán
✔️ Giải thích
✔️ Đề xuất chính sách

