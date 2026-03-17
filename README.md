# 📊 HR Analytics Project – Full ML Pipeline (Final Version)

## 🎯 Mục tiêu

Xây dựng hệ thống **phân tích & dự đoán Attrition (nghỉ việc)** của nhân viên bằng Machine Learning, kết hợp:

* Feature Engineering
* Clustering
* Classification
* Semi-supervised Learning
* Association Rules
* Model Explainability
* Regression

---

# 📂 Cấu trúc project

```
HR_Analytics_Project/
│── HR_Analytics.csv
│── hr_pipeline.py
│── fig0 → fig11 (biểu đồ output)
│── README.md
```

---

# ⚙️ Pipeline tổng thể

## 1. Data Preprocessing

* Làm sạch dữ liệu (drop cột dư thừa)
* Fill missing (median)
* Encode categorical (LabelEncoder)
* StandardScaler

👉 Giúp dữ liệu **sạch + sẵn sàng cho ML**

---

## 2. Feature Engineering

Tạo 6 feature mới:

* IncomePerYearExp
* CompanyTenureRatio
* SatisfactionIndex
* PromotionGap
* IncomeSalaryDiff
* ExternalExperience

👉 Ý nghĩa:

* Biểu diễn insight thực tế HR (kinh nghiệm, thu nhập, hài lòng)

---

# 📊 GIẢI THÍCH 12 BIỂU ĐỒ 
---

## 📌 **fig0 – Feature Engineering & Selection**

![fig0](fig0_feature_engineering.png)

👉 Gồm 3 phần:

* F-score (SelectKBest): chọn feature quan trọng
* Random Forest Importance: xác nhận lại
* SatisfactionIndex vs Attrition

📌 Kết luận:

* Feature mới **có đóng góp thực sự**
* SatisfactionIndex liên quan mạnh tới nghỉ việc

---

## 📌 **fig1 – EDA (Exploratory Data Analysis)**

![fig1](fig1_eda.png)

👉 Gồm:

* Tỷ lệ nghỉ việc
* Age vs Attrition
* Salary vs Attrition
* Overtime vs Attrition
* Department vs Attrition
* Job Satisfaction

📌 Kết luận:

* Nhân viên OT nhiều → nghỉ cao
* Lương thấp → nghỉ nhiều
* Satisfaction thấp → nguy cơ nghỉ

---

## 📌 **fig2 – Correlation Heatmap**

![fig2](fig2_correlation.png)

👉 Thể hiện tương quan giữa các feature

📌 Kết luận:

* Không có multicollinearity quá mạnh
* Một số nhóm feature liên quan chặt

---

## 📌 **fig3 – Clustering (KMeans)**

![fig3](fig3_clustering.png)

👉 Gồm:

* Elbow Method (chọn k=4)
* PCA visualization
* Attrition theo cluster

📌 Kết luận:

* Có thể phân nhóm nhân viên rõ ràng
* Mỗi cluster có **risk nghỉ việc khác nhau**

---

## 📌 **fig4 – Model Evaluation**

![fig4](fig4_model_eval.png)

👉 Gồm:

* ROC Curve
* Precision-Recall
* Confusion Matrix

📌 Kết luận:

* Random Forest thường tốt nhất
* Model phân biệt tốt giữa Stay / Leave

---

## 📌 **fig5 – Hyperparameter Comparison**

![fig5](fig5_hyperparams.png)

👉 So sánh:

* AUC
* F1 Score
* Training Time

📌 Kết luận:

* Trade-off giữa hiệu năng và thời gian
* RF & GBM mạnh hơn Logistic

---

## 📌 **fig6 – Feature Importance**

![fig6](fig6_feature_importance.png)

👉 Top feature quan trọng

📌 Kết luận:

* Feature mới (màu đỏ) có ảnh hưởng lớn
* Income + Satisfaction là key driver

---

## 📌 **fig7 – Semi-Supervised Learning**

![fig7](fig7_semi_supervised.png)

👉 So sánh:

* Supervised vs Self-training vs Label Spreading

📌 Kết luận:

* Label Spreading tốt nhất khi thiếu nhãn
* Self-training dễ bị bias

---

## 📌 **fig8 – Association Rules**

![fig8](fig8_association_rules.png)

👉 Phân tích luật:

* Support
* Confidence
* Lift

📌 Kết luận:

* Nhận diện pattern nghỉ việc
* Ví dụ:

  * OT + lương thấp → nghỉ cao

---

## 📌 **fig9 – Model Explainability**

![fig9](fig9_model_explanation.png)

👉 Gồm:

* Permutation Importance
* Partial Dependence

📌 Kết luận:

* Hiểu được model “nghĩ gì”
* Feature ảnh hưởng trực tiếp đến xác suất nghỉ

---

## 📌 **fig10 – Regression (Job Satisfaction)**

![fig10](fig10_regression.png)

👉 Dự đoán mức hài lòng

📌 Kết luận:

* Ridge Regression tốt nhất
* Có thể dùng để:
  → dự đoán tâm lý nhân viên

---

## 📌 **fig11 – Leakage Check**

![fig11](fig11_leakage_check.png)

👉 Kiểm tra data leakage

📌 Kết luận:

* Đã loại bỏ feature gây leak
* Model đáng tin cậy

---

# 🤖 Model sử dụng

* Logistic Regression
* Random Forest ⭐ (best)
* Gradient Boosting

---

# 🔬 Kết quả chính

* ✔ AUC cao (~0.85+)
* ✔ Feature Engineering hiệu quả
* ✔ Clustering phát hiện nhóm risk
* ✔ Semi-supervised cải thiện khi thiếu data

---

# 💡 Insight quan trọng 

* OT là yếu tố mạnh nhất gây nghỉ việc
* Satisfaction thấp → nghỉ cao
* Lương thấp → rủi ro lớn
* Có thể dùng model để:

  * Predict nghỉ việc
  * Đưa ra chính sách giữ người

---

# 🚀 Run project

```bash
python hr_pipeline.py
```

---

# 🏆 Kết luận

Dự án đã xây dựng **pipeline ML end-to-end hoàn chỉnh**, bao gồm:

✔ Data processing
✔ Feature engineering
✔ Clustering
✔ Classification
✔ Semi-supervised learning
✔ Explainability
✔ Regression

👉 Có thể áp dụng thực tế trong HR Analytics

---

# 🔥 Tổng kết 

**"Đây là một hệ thống HR Analytics hoàn chỉnh, không chỉ dự đoán nghỉ việc mà còn giải thích nguyên nhân và hỗ trợ ra quyết định quản trị nhân sự."**
