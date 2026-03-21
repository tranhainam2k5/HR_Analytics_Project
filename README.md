# 🚀 HR Analytics Project – Employee Attrition Prediction

## 📌 1. Giới thiệu

Dự án này xây dựng một **pipeline Machine Learning hoàn chỉnh** nhằm:

* 🔍 Phân tích hành vi nhân viên
* 🎯 Dự đoán khả năng nghỉ việc (Attrition)
* 📊 Trích xuất insight phục vụ quyết định nhân sự

👉 Đây không chỉ là bài toán dự đoán, mà là **hệ thống hỗ trợ ra quyết định cho HR**.

---

## 📊 2. Dataset

* Nguồn: Kaggle HR Analytics
* Kích thước: **1480 × 38**
* Sau xử lý: **1480 × 31**

### Phân bố Attrition:

* Stay (No): 1242 (84%)
* Leave (Yes): 238 (16%)

⚠️ Đây là bài toán **mất cân bằng dữ liệu (imbalanced classification)**

---

## ⚙️ 3. Data Preprocessing

* Xử lý missing values
* Encode categorical (LabelEncoder)
* Scale dữ liệu (StandardScaler)
* Xử lý imbalance:

  * `class_weight='balanced'`

---

## 🧠 4. Feature Engineering

Tạo thêm 6 đặc trưng mới:

* IncomePerYearExp
* CompanyTenureRatio
* SatisfactionIndex
* PromotionGap
* IncomeSalaryDiff
* ExternalExperience

👉 Mục tiêu: **biến dữ liệu thô thành insight có ý nghĩa business**

---

## 🔍 5. Feature Selection

Sử dụng **SelectKBest (ANOVA F-test)**

### Top features:

* 🔥 OverTime
* 🔥 TotalWorkingYears
* 🔥 JobLevel
* 🔥 YearsInCurrentRole
* 🔥 SatisfactionIndex
* 🔥 MonthlyIncome

👉 Insight:

> Nhân viên làm OT nhiều + ít thăng tiến → nguy cơ nghỉ việc cao

---

## 🧩 6. Clustering – Phân cụm nhân viên

Sử dụng **K-Means (k=4)**

### Kết quả:

| Cluster | Attrition Rate |
| ------- | -------------- |
| 0       | 10.5%          |
| 1       | 11.1%          |
| 2       | 9.0%           |
| 3       | 🔥 22.0%       |

👉 Insight:

> Cluster 3 là nhóm rủi ro cao → cần can thiệp sớm

---

## 🤖 7. Classification – Dự đoán nghỉ việc

### So sánh mô hình:

| Model               | AUC    | F1 (Leave) |
| ------------------- | ------ | ---------- |
| Logistic Regression | 0.8503 | ✅ **0.51** |
| Random Forest       | 0.8616 | 0.39       |
| Gradient Boosting   | 0.8716 | 0.41       |

---

### 🎯 Kết luận:

* GBM có AUC cao nhất
* Nhưng Logistic Regression có **F1 tốt nhất cho lớp Leave**

👉 Chọn **Logistic Regression**

---

## ⚙️ Hyperparameters

* Logistic Regression:

  * C=1.0, max_iter=1000
* Random Forest:

  * n_estimators=200, max_depth=8
* Gradient Boosting:

  * learning_rate=0.05, n_estimators=200

---

## 🧠 8. Model Explainability

Sử dụng **Permutation Importance**

### Top Features:

1. 🔥 OverTime
2. StockOptionLevel
3. SatisfactionIndex
4. MonthlyIncome
5. Age

---

### 🎯 Insight quan trọng:

> OverTime là yếu tố ảnh hưởng mạnh nhất đến nghỉ việc

---

## 🔗 9. Association Rules (Apriori)

* 1480 transactions × 26 items
* 231 frequent itemsets
* 97 rules

### Ví dụ:

* OT cao + lương thấp → nghỉ việc
* Lương cao → ở lại

---

### 🎯 Ứng dụng:

* Rule-based alert system
* Hỗ trợ HR ra quyết định nhanh

---

## 🔬 10. Semi-Supervised Learning

So sánh:

| Method          | Hiệu quả |
| --------------- | -------- |
| Self-Training   | ❌ kém    |
| Label Spreading | ✅ tốt    |

---

### 🎯 Insight:

* Self-training bị bias do imbalance
* Label Spreading ổn định hơn

---

### 📌 Khuyến nghị:

| % Label | Hành động     |
| ------- | ------------- |
| 5%      | chỉ cảnh báo  |
| 10%     | khảo sát thêm |
| 20%     | can thiệp     |
| 50%     | deploy        |

---

## 📉 11. Regression – Job Satisfaction

* Model: Linear / Ridge / Lasso
* Kết quả: R² ≈ 0

👉 Insight:

> Job Satisfaction khó dự đoán → phụ thuộc yếu tố ẩn

---

## 🚨 12. Data Leakage

Phát hiện:

* SatisfactionIndex gây leakage

👉 Đã loại bỏ để đảm bảo tính chính xác

---

## 📊 13. Kết quả trực quan

Project sinh ra 12 biểu đồ:

* fig0 → Feature Engineering
* fig1 → EDA
* fig2 → Correlation
* fig3 → Clustering
* fig4 → Model Evaluation
* fig5 → Hyperparameters
* fig6 → Feature Importance
* fig7 → Semi-supervised
* fig8 → Association Rules
* fig9 → Explainability
* fig10 → Regression
* fig11 → Leakage Check

---

## 💡 14. Insight Business

Nhân viên dễ nghỉ việc khi:

* 🔥 Làm thêm giờ nhiều
* 💰 Lương thấp
* 📉 Ít thăng tiến
* 😞 Satisfaction thấp

---

## 🎯 15. Ứng dụng thực tế

* Dự đoán nhân viên nghỉ việc
* Dashboard HR
* Hệ thống cảnh báo sớm
* Cá nhân hóa giữ chân nhân viên

---

## 🛠️ 16. Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## ▶️ 17. Cách chạy project

```bash
pip install -r requirements.txt
python hr_pipeline.py
```

---

## 📌 18. Kết luận

Đây là một hệ thống:

* ✅ Phân tích dữ liệu
* ✅ Dự đoán
* ✅ Giải thích
* ✅ Hỗ trợ quyết định

👉 Có thể mở rộng thành:

* Dashboard (Power BI / Streamlit)
* Web App
* HR Decision System

---


