# 🚀 HR Analytics Project – Employee Attrition Prediction

> **Hệ thống Machine Learning end-to-end** giúp HR dự đoán và ngăn chặn nguy cơ nghỉ việc của nhân viên.

---

## 📌 1. Giới thiệu

Dự án xây dựng một **pipeline Machine Learning end-to-end** nhằm phân tích và dự đoán **nguy cơ nghỉ việc của nhân viên (Attrition)**.

Thay vì chỉ đơn thuần xây dựng một mô hình ML, dự án này hướng đến một **Decision Support System (Hệ thống hỗ trợ ra quyết định)** toàn diện cho bộ phận HR, bao gồm:

- 🔍 **Phân tích hành vi nhân viên** — hiểu rõ ai có nguy cơ cao và tại sao
- 🤖 **Dự đoán khả năng nghỉ việc** — cảnh báo sớm trước khi nhân viên quyết định rời đi
- 📊 **Trích xuất insight hỗ trợ HR** — đưa ra hành động cụ thể để giữ chân nhân tài

---

## 📊 2. Dataset

| Thông tin | Chi tiết |
|-----------|----------|
| Nguồn | Kaggle HR Analytics |
| Kích thước ban đầu | **1,480 hàng × 38 cột** |
| Sau xử lý | **1,480 hàng × 31 cột** (loại bỏ 7 cột dư thừa/rò rỉ) |

### 📈 Phân bố nhãn Attrition:

| Nhãn | Số lượng | Tỷ lệ |
|------|----------|-------|
| Stay – Ở lại (No) | 1,242 | 84% |
| Leave – Nghỉ việc (Yes) | 238 | 16% |

⚠️ **Mất cân bằng dữ liệu (Imbalanced Classification):** Tỷ lệ 84:16 có nghĩa là mô hình sẽ bị thiên về dự đoán "ở lại" nếu không xử lý. Giải pháp: sử dụng `class_weight='balanced'` để mô hình chú ý đến lớp thiểu số (nhân viên nghỉ việc) nhiều hơn.

---

## ⚙️ 3. Data Preprocessing

Dữ liệu thực tế thường bị bẩn và không đồng nhất. Pipeline tiền xử lý gồm các bước:

1. **Xử lý missing values** — điền giá trị trung bình/mode hoặc loại bỏ hàng không hợp lệ
2. **Encode categorical** — chuyển các biến hạng mục (e.g., `Department`, `JobRole`) thành số bằng `LabelEncoder`
3. **Chuẩn hóa (StandardScaler)** — đưa tất cả feature về cùng thang đo, tránh bias do chênh lệch đơn vị
4. **Xử lý imbalance** — dùng `class_weight='balanced'` để mô hình không bỏ qua nhóm nhân viên nghỉ việc

```python
class_weight = 'balanced'
```

---

## 🧠 4. Feature Engineering

### 📊 Biểu đồ: Feature Engineering Overview

![Feature Engineering](fig0_feature_engineering.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ hiển thị 6 đặc trưng mới được tạo ra từ các cột gốc, kèm theo phân bố giá trị và mức độ tương quan với biến mục tiêu (Attrition). Thanh màu đỏ/cam biểu thị nhân viên **nghỉ việc**, thanh xanh biểu thị nhân viên **ở lại**. Khi hai phân bố tách biệt rõ ràng → feature đó có sức phân tách tốt.

Thay vì chỉ dùng dữ liệu thô, chúng tôi tạo thêm **6 đặc trưng có ý nghĩa kinh doanh**:

| Feature mới | Công thức / Ý nghĩa |
|-------------|---------------------|
| `IncomePerYearExp` | `MonthlyIncome / TotalWorkingYears` — Thu nhập trên mỗi năm kinh nghiệm; thấp → cảm giác bị "trả chưa xứng" |
| `CompanyTenureRatio` | `YearsAtCompany / TotalWorkingYears` — Tỷ lệ thời gian gắn bó với công ty hiện tại so với tổng kinh nghiệm |
| `SatisfactionIndex` | Tổng hợp các chỉ số hài lòng (`JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`) |
| `PromotionGap` | `YearsAtCompany - YearsSinceLastPromotion` — Khoảng cách từ lần thăng tiến cuối; cao → cảm giác bị bỏ quên |
| `IncomeSalaryDiff` | Chênh lệch giữa thu nhập thực tế và mức kỳ vọng theo thị trường |
| `ExternalExperience` | `TotalWorkingYears - YearsAtCompany` — Số năm làm việc ở công ty khác trước khi gia nhập |

👉 **Mục tiêu:** Biến dữ liệu thô thành insight có ý nghĩa kinh doanh, giúp mô hình "hiểu" được động lực của nhân viên.

---

## 🔍 5. Feature Selection

### 📊 Biểu đồ 1: EDA – Exploratory Data Analysis

![EDA](fig1_eda.png)

> **Cách đọc biểu đồ này:**
> Đây là tập hợp các biểu đồ phân tích khám phá dữ liệu. Mỗi subplot thể hiện phân bố của một feature theo nhóm Attrition (Yes/No). Chú ý đến những feature có **đường phân bố tách biệt rõ** giữa hai nhóm — đó là những yếu tố dự báo quan trọng.
>
> Ví dụ: nếu phân bố `OverTime` của nhóm "Leave" lệch hẳn về phía "Yes" so với nhóm "Stay" → OverTime là dấu hiệu cảnh báo mạnh. Ngược lại, feature nào có hai đường phân bố chồng lên nhau hoàn toàn thì không có giá trị phân tách.

---

### 📊 Biểu đồ 2: Correlation Heatmap

![Correlation](fig2_correlation.png)

> **Cách đọc biểu đồ này:**
> Heatmap thể hiện mức độ tương quan (từ -1 đến +1) giữa các cặp feature.
>
> - **Màu đỏ đậm (+1):** Hai feature tăng/giảm cùng chiều — tương quan dương mạnh
> - **Màu xanh đậm (-1):** Một feature tăng thì feature kia giảm — tương quan âm mạnh
> - **Màu nhạt (≈0):** Hai feature hầu như không liên quan nhau
>
> Tập trung vào **hàng/cột `Attrition`** để thấy feature nào ảnh hưởng nhiều nhất đến nghỉ việc. Đồng thời, cần tránh dùng đồng thời hai feature có tương quan > 0.8 với nhau vì chúng mang thông tin trùng lặp (multicollinearity), làm giảm hiệu quả mô hình.

**Sử dụng SelectKBest (ANOVA F-test)** để chọn các feature có khả năng phân tách nhóm tốt nhất.

### 🔥 Top features được chọn:

| Rank | Feature | Ý nghĩa |
|------|---------|---------|
| 1 | `OverTime` | Làm thêm giờ — yếu tố mạnh nhất |
| 2 | `TotalWorkingYears` | Tổng số năm kinh nghiệm |
| 3 | `JobLevel` | Cấp bậc công việc |
| 4 | `YearsInCurrentRole` | Số năm giữ vị trí hiện tại |
| 5 | `SatisfactionIndex` | Chỉ số hài lòng tổng hợp |
| 6 | `MonthlyIncome` | Thu nhập hàng tháng |

> 🎯 **Insight:** Nhân viên làm OT nhiều + ít được thăng tiến → nguy cơ nghỉ việc cao. Đây là dấu hiệu của sự kiệt sức (burnout) kết hợp với cảm giác thiếu ghi nhận.

---

## 🧩 6. Clustering – Phân nhóm nhân viên

### 📊 Biểu đồ: K-Means Clustering

![Clustering](fig3_clustering.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ scatter plot 2D (sau khi giảm chiều bằng PCA) hiển thị 4 cụm nhân viên, mỗi cụm một màu khác nhau. Mỗi điểm là một nhân viên. **Điểm hình ✕ hoặc điểm sáng màu** trong mỗi cụm là các nhân viên đã nghỉ việc.
>
> Khi một cụm có mật độ điểm nghỉ việc cao → đó là **nhóm rủi ro cần ưu tiên**. Khoảng cách giữa các cụm trên biểu đồ thể hiện mức độ khác biệt về đặc điểm giữa các nhóm nhân viên — cụm nào càng cách xa nhau thì tính chất nhân viên càng khác nhau nhiều.

**Sử dụng K-Means với k=4** để phân chia nhân viên thành 4 nhóm hành vi:

| Cluster | Attrition Rate | Đặc điểm tiêu biểu |
|---------|---------------|---------------------|
| 0 | 10.5% | Nhân viên lâu năm, thu nhập ổn định |
| 1 | 11.1% | Nhân viên cấp trung, hài lòng vừa phải |
| 2 | 9.0% | Nhân viên mới, có định hướng phát triển |
| **3** | **🔥 22.0%** | **Làm OT nhiều, lương thấp, ít thăng tiến** |

> 🎯 **Insight:** **Cluster 3 có tỷ lệ nghỉ việc 22%** — gần gấp đôi mức trung bình toàn công ty. Đây là nhóm cần được HR **ưu tiên can thiệp ngay**: xem xét tăng lương, giảm OT, hoặc tạo lộ trình thăng tiến rõ ràng.

---

## 🤖 7. Classification – Dự đoán nghỉ việc

### 📊 Biểu đồ: Model Evaluation

![Model Evaluation](fig4_model_eval.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ gồm hai phần chính:
>
> **ROC Curve (trái):** Trục X = False Positive Rate (tỷ lệ cảnh báo nhầm), Trục Y = True Positive Rate (tỷ lệ phát hiện đúng). Đường cong càng gần góc trên-trái → mô hình càng tốt. **AUC** là diện tích dưới đường cong, dao động từ 0.5 (đoán ngẫu nhiên) đến 1.0 (hoàn hảo).
>
> **Confusion Matrix (phải):** Ma trận 2×2 cho thấy số ca dự đoán đúng/sai.
> - Ô trên-trái (True Negative): Dự đoán "ở lại" → đúng ✅
> - Ô dưới-phải (True Positive): Dự đoán "nghỉ" → đúng ✅
> - Ô trên-phải (False Positive): Dự đoán "nghỉ" nhưng thực ra ở lại ⚠️
> - Ô dưới-trái (False Negative): Dự đoán "ở lại" nhưng thực ra nghỉ ❌ — **sai lầm tốn kém nhất**

**So sánh 3 mô hình:**

| Model | AUC | F1 (Leave) | Nhận xét |
|-------|-----|------------|---------|
| Logistic Regression | 0.8503 | ✅ **0.51** | Cân bằng tốt giữa Precision và Recall |
| Random Forest | 0.8616 | 0.39 | AUC cao nhưng bỏ sót nhiều ca nghỉ việc |
| Gradient Boosting | **0.8716** | 0.41 | AUC cao nhất nhưng F1 cho lớp Leave thấp |

**🎯 Tại sao chọn Logistic Regression?**

Trong bài toán HR, **bỏ sót một nhân viên sắp nghỉ (False Negative) tốn kém hơn nhiều** so với cảnh báo nhầm. Chi phí tuyển dụng thay thế thường bằng 50–200% lương năm. Do đó, **F1-score cho lớp "Leave"** quan trọng hơn AUC tổng thể. Logistic Regression với F1=0.51 vượt trội các mô hình còn lại ở tiêu chí này.

---

## ⚙️ 8. Hyperparameter Tuning

### 📊 Biểu đồ: Hyperparameter Search

![Hyperparameters](fig5_hyperparams.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ heatmap hoặc line plot thể hiện sự thay đổi hiệu năng mô hình (AUC/F1) theo từng giá trị hyperparameter được thử nghiệm. **Màu đậm hơn / điểm cao hơn** = cấu hình hyperparameter tốt hơn.
>
> Chú ý đến vùng mà **train score và validation score gần nhau** — đây là vùng cân bằng tốt giữa underfitting và overfitting. Nếu train score cao nhưng validation score thấp → mô hình đang overfit, cần điều chỉnh (giảm độ phức tạp hoặc tăng regularization).

| Model | Hyperparameter tối ưu |
|-------|----------------------|
| Logistic Regression | `C=1.0`, `max_iter=1000` |
| Random Forest | `n_estimators=200`, `max_depth=8` |
| Gradient Boosting | `learning_rate=0.05`, `n_estimators=200` |

---

## 🧠 9. Model Explainability – Giải thích mô hình

### 📊 Biểu đồ 1: Feature Importance

![Feature Importance](fig6_feature_importance.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ **bar chart nằm ngang**, mỗi thanh thể hiện mức độ ảnh hưởng của một feature đến quyết định của mô hình. **Thanh càng dài = feature càng quan trọng**.
>
> Màu sắc thể hiện chiều ảnh hưởng:
> - **Màu đỏ/cam:** Feature này tăng → xác suất nghỉ việc tăng (yếu tố đẩy nhân viên rời đi)
> - **Màu xanh:** Feature này tăng → xác suất ở lại tăng (yếu tố giữ chân nhân viên)
>
> Kết quả từ phân tích hệ số Logistic Regression hoặc SHAP values giúp HR hiểu **cụ thể tại sao** mô hình đưa ra dự đoán đó, thay vì chỉ tin vào "hộp đen".

### 🔥 Top 5 yếu tố ảnh hưởng:

| Rank | Feature | Chiều ảnh hưởng |
|------|---------|----------------|
| 1 | `OverTime` | ↑ Làm OT nhiều → ↑ nghỉ việc (mạnh nhất) |
| 2 | `StockOptionLevel` | ↑ Stock options → ↓ nghỉ việc (giữ chân hiệu quả) |
| 3 | `SatisfactionIndex` | ↑ Hài lòng → ↓ nghỉ việc |
| 4 | `MonthlyIncome` | ↑ Lương → ↓ nghỉ việc |
| 5 | `Age` | Nhân viên trẻ hơn → ↑ nghỉ việc (linh hoạt hơn) |

> 🎯 **Insight:** OverTime là **yếu tố số 1**. Một chính sách đơn giản như giới hạn số giờ OT tối đa/tuần hoặc bù đắp OT thỏa đáng có thể giảm đáng kể tỷ lệ nghỉ việc.

---

### 📊 Biểu đồ 2: SHAP Summary Plot

![Model Explanation](fig9_model_explanation.png)

> **Cách đọc biểu đồ này:**
> Đây là **SHAP summary plot** — công cụ giải thích mô hình mạnh nhất hiện nay. Mỗi hàng là một feature, mỗi chấm là một nhân viên.
>
> - **Trục X (SHAP value):** Dương (+) = feature này đẩy xác suất nghỉ việc lên; Âm (-) = feature này kéo xác suất nghỉ việc xuống
> - **Màu chấm:** Đỏ = giá trị feature cao; Xanh = giá trị feature thấp
> - **Đọc cụ thể:** Hàng `OverTime` — chấm đỏ (OT nhiều) tập trung bên phải (+) → làm OT nhiều làm tăng xác suất nghỉ. Chấm xanh (ít OT) tập trung bên trái (-) → ít OT giúp nhân viên ở lại.
>
> SHAP cho phép giải thích **từng cá nhân nhân viên**: "Nhân viên A có xác suất nghỉ 78% vì OT cao (+0.4), lương thấp (+0.3), nhưng được bù lại một phần bởi stock option cao (-0.2)."

---

## 🔗 10. Association Rules – Luật kết hợp (Apriori)

### 📊 Biểu đồ: Association Rules Network

![Association Rules](fig8_association_rules.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ dạng **network graph** hoặc **bubble scatter plot** với 3 trục thông tin:
> - **Trục X = Support:** Tần suất pattern này xuất hiện trong dữ liệu (0-1). Support = 0.1 nghĩa là 10% nhân viên có pattern này.
> - **Trục Y = Confidence:** Khi điều kiện xảy ra, kết quả xảy ra với xác suất bao nhiêu. Confidence = 0.7 nghĩa là 70% nhân viên thỏa điều kiện sẽ nghỉ việc.
> - **Kích thước bong bóng = Lift:** Lift > 1 nghĩa là hai sự kiện có liên hệ thật sự, không phải ngẫu nhiên. Lift = 3 nghĩa là pattern này xảy ra nhiều gấp 3 lần so với kỳ vọng ngẫu nhiên.
>
> **Luật tốt nhất** nằm ở **góc trên phải, bong bóng lớn** — Support cao + Confidence cao + Lift cao.

### 🔥 Ví dụ luật quan trọng:

| Điều kiện (IF) | Kết quả (THEN) | Confidence |
|----------------|----------------|-----------|
| OT cao + Lương thấp | → Nghỉ việc | ~70% |
| Stock Option = 0 + JobLevel thấp | → Nghỉ việc | ~65% |
| Lương cao + JobLevel cao | → Ở lại | ~88% |
| Thăng tiến gần đây + Hài lòng cao | → Ở lại | ~82% |

### 🎯 Ứng dụng thực tế:
- **Rule-based alert system:** Tự động gắn cờ nhân viên khi thỏa mãn điều kiện nguy cơ
- **Hỗ trợ HR ra quyết định:** Đưa ra lý do cụ thể thay vì chỉ một con số xác suất mơ hồ

---

## 🔬 11. Semi-Supervised Learning

### 📊 Biểu đồ: Semi-Supervised Comparison

![Semi-Supervised](fig7_semi_supervised.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ so sánh hiệu năng giữa **Self-Training** và **Label Spreading** khi chỉ có một phần nhỏ dữ liệu được gán nhãn (tình huống thực tế: HR chỉ biết chắc một số ít nhân viên đã/chưa nghỉ).
>
> - **Trục X:** Tỷ lệ dữ liệu có nhãn (10%, 20%... 80%)
> - **Trục Y:** F1-score hoặc Accuracy trên tập test
> - **Đường phẳng và cao hơn** = phương pháp ổn định và hiệu quả hơn
>
> Self-Training bị bias vì nó tự gán nhãn dựa trên dự đoán ban đầu — nếu dự đoán đầu sai, sai lầm sẽ tích lũy theo vòng lặp. Label Spreading dùng cấu trúc đồ thị (graph) của dữ liệu để lan truyền nhãn, nên ít bị ảnh hưởng hơn bởi các nhãn sai ban đầu.

| Phương pháp | Kết quả | Lý do |
|-------------|---------|-------|
| Self-Training | ❌ Bị bias | Tự lặp lại và khuếch đại sai lầm khi gán nhãn |
| **Label Spreading** | ✅ Ổn định | Lan truyền nhãn qua cấu trúc đồ thị dữ liệu |

---

## 📉 12. Regression – Dự đoán Job Satisfaction

### 📊 Biểu đồ: Regression Analysis

![Regression](fig10_regression.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ **scatter plot** với đường hồi quy (regression line). Trục X = giá trị thực tế của Job Satisfaction, trục Y = giá trị mô hình dự đoán.
>
> Mô hình **tốt** → các điểm nằm sát đường chéo 45° (dự đoán = thực tế). Mô hình **kém** → các điểm phân tán ngẫu nhiên, không theo đường nào. R² ≈ 0 trong trường hợp này cho thấy mô hình không học được mối quan hệ nào có ý nghĩa.
>
> Đây **không phải lỗi kỹ thuật** mà là một **phát hiện có giá trị**: Job Satisfaction là chỉ số chủ quan, bị chi phối bởi yếu tố tâm lý và xã hội mà dữ liệu số không nắm bắt được.

**Kết quả:** R² ≈ 0

> 🎯 **Insight có giá trị:** Mức độ hài lòng của nhân viên **không thể dự đoán đơn giản** từ lương, cấp bậc hay số năm kinh nghiệm. Điều này nhấn mạnh tầm quan trọng của **khảo sát định tính**, **1-on-1 meeting** và văn hóa tổ chức trong việc quản lý nhân sự.

---

## 🚨 13. Phát hiện Data Leakage

### 📊 Biểu đồ: Leakage Detection

![Leakage Check](fig11_leakage_check.png)

> **Cách đọc biểu đồ này:**
> Biểu đồ so sánh **feature importance trước và sau khi loại bỏ feature gây rò rỉ dữ liệu (data leakage)**.
>
> **Dấu hiệu leakage:**
> - Một feature chiếm tỷ trọng importance bất thường cao (ví dụ > 50%) trong khi các feature khác gần bằng 0
> - Accuracy trên test set cao bất thường (> 95%) — mô hình "biết trước đáp án"
> - Sau khi loại bỏ feature đó, hiệu năng giảm xuống mức thực tế hơn nhưng **đáng tin cậy hơn** cho môi trường production
>
> Hai subplot trái/phải thể hiện distribution của feature trước và sau khi xử lý — phân bố tách biệt hoàn toàn giữa hai nhóm Yes/No là dấu hiệu rõ ràng của leakage.

**Vấn đề phát hiện:**

`SatisfactionIndex` được tạo từ tổng hợp nhiều chỉ số hài lòng — trong đó có thông tin gián tiếp phản ánh quyết định nghỉ việc đã biết trước (nhân viên đã nghỉ thường đánh giá thấp hơn do bias nhớ lại). Điều này khiến mô hình "gian lận" — học được đáp án thay vì học pattern thực sự.

**Hành động khắc phục:** Loại bỏ `SatisfactionIndex` khỏi feature set → mô hình tổng quát hóa tốt hơn trên dữ liệu nhân viên mới chưa có nhãn.

---

## 💡 14. Business Insights – Kết luận từ dữ liệu

Tổng hợp từ tất cả phân tích, **nhân viên có nguy cơ nghỉ việc cao** khi:

| Yếu tố | Dấu hiệu nguy hiểm | Hành động HR đề xuất |
|--------|-------------------|----------------------|
| 🔥 **OverTime** | Làm thêm giờ liên tục | Giới hạn OT, bổ sung phụ cấp OT |
| 💰 **Thu nhập** | Lương thấp hơn mặt bằng thị trường | Review và điều chỉnh lương định kỳ |
| 📉 **Thăng tiến** | Không được thăng tiến > 3 năm | Tạo lộ trình career rõ ràng |
| 😞 **Satisfaction** | Điểm hài lòng < 2/4 | 1-on-1 meeting, khảo sát ẩn danh định kỳ |
| 🎯 **Stock Option** | Không có cổ phần công ty | Cân nhắc chương trình ESOP |

---

## 🎯 15. Ứng dụng thực tế

- **Dự đoán nhân viên nghỉ việc** — Điểm rủi ro từ 0 đến 1 cho từng cá nhân, cập nhật hàng tháng
- **HR Dashboard** — Tổng quan theo phòng ban, cluster, và xu hướng theo thời gian
- **Hệ thống cảnh báo sớm** — Tự động gắn cờ khi nhân viên thỏa mãn luật rủi ro (OT + lương thấp...)
- **Cá nhân hóa giữ chân** — Đề xuất hành động phù hợp theo từng nhóm nhân viên

---

## 🛠️ 16. Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?logo=numpy&logoColor=white)

| Thư viện | Mục đích |
|----------|---------|
| `pandas`, `numpy` | Xử lý và biến đổi dữ liệu |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib`, `seaborn` | Visualization |
| `mlxtend` | Apriori association rules |

---

## ▶️ 17. Cách chạy project

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Chạy toàn bộ pipeline
python hr_pipeline.py

# 3. Output:
# - fig0_feature_engineering.png ... fig11_leakage_check.png  (12 biểu đồ)
# - model_results.csv                                          (số liệu đánh giá)
```

---

## 📌 18. Kết luận

Dự án đã xây dựng thành công một **hệ thống ML end-to-end** hoàn chỉnh:

| Module | Kết quả |
|--------|---------|
| ✅ Phân tích dữ liệu | EDA + Feature Engineering + Selection |
| ✅ Phân cụm nhân viên | K-Means phát hiện nhóm rủi ro 22% |
| ✅ Dự đoán nghỉ việc | Logistic Regression đạt F1=0.51 cho lớp nghỉ việc |
| ✅ Giải thích mô hình | SHAP xác nhận OverTime là yếu tố số 1 |
| ✅ Luật kết hợp | Apriori sinh rule-based alert system thực tế |
| ✅ Đảm bảo chất lượng | Phát hiện và xử lý data leakage |

**👉 Hướng mở rộng tiếp theo:**
- 📊 Dashboard — Power BI / Streamlit visualization
- 🌐 Web App — FastAPI + React cho HR sử dụng trực tiếp  
- 🔔 Alert System — Tích hợp email/Slack notification tự động
- 📱 HR Decision System — Gợi ý hành động cụ thể cho từng nhân viên rủi ro

---

<div align="center">

**⭐ Nếu project hữu ích, hãy để lại một Star! ⭐**

Made with ❤️ by [tranhainam2k5](https://github.com/tranhainam2k5)

</div>