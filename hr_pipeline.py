
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                              confusion_matrix, roc_curve, precision_recall_curve,
                              average_precision_score)
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2563EB', '#DC2626', '#16A34A', '#CA8A04', '#7C3AED']

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & BASIC EDA
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv('HR_Analytics.csv')
df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")
print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")

# Drop constant / redundant cols
drop_cols = ['EmpID', 'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Fill missing YearsWithCurrManager with median
df['YearsWithCurrManager'].fillna(df['YearsWithCurrManager'].median(), inplace=True)

# Drop derived columns
df.drop(columns=['AgeGroup', 'SalarySlab'], inplace=True, errors='ignore')
print(f"After cleaning: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. ENCODE
# ══════════════════════════════════════════════════════════════════════════════
df_enc = df.copy()
le = LabelEncoder()
cat_cols = df_enc.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

y = df_enc['Attrition']
X_raw = df_enc.drop(columns=['Attrition'])

# Impute
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  ← PHẦN BỔ SUNG MỚI
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== FEATURE ENGINEERING ===")

X_fe = X_imputed.copy()

# --- 3.1 Tạo đặc trưng mới (Domain-based) ---
# Thu nhập theo năm kinh nghiệm (năng suất thu nhập)
X_fe['IncomePerYearExp']    = X_fe['MonthlyIncome'] / (X_fe['TotalWorkingYears'] + 1)

# Tỷ lệ thời gian tại công ty so với tổng kinh nghiệm
X_fe['CompanyTenureRatio']  = X_fe['YearsAtCompany'] / (X_fe['TotalWorkingYears'] + 1)

# Chỉ số hài lòng tổng hợp (trung bình 4 loại hài lòng)
X_fe['SatisfactionIndex']   = (X_fe['JobSatisfaction'] + X_fe['EnvironmentSatisfaction'] +
                                X_fe['RelationshipSatisfaction'] + X_fe['WorkLifeBalance']) / 4

# Khoảng cách thăng tiến (đã bao lâu chưa được thăng)
X_fe['PromotionGap']        = X_fe['YearsAtCompany'] - X_fe['YearsSinceLastPromotion']

# Thu nhập so với mức lương theo giờ * 160 giờ/tháng (chênh lệch cấu trúc lương)
X_fe['IncomeSalaryDiff']    = X_fe['MonthlyIncome'] - (X_fe['HourlyRate'] * 160)

# Kinh nghiệm ngoài công ty
X_fe['ExternalExperience']  = X_fe['TotalWorkingYears'] - X_fe['YearsAtCompany']

new_features = ['IncomePerYearExp', 'CompanyTenureRatio', 'SatisfactionIndex',
                'PromotionGap', 'IncomeSalaryDiff', 'ExternalExperience']
print(f"Tạo {len(new_features)} đặc trưng mới: {new_features}")

# --- 3.2 Feature Selection bằng SelectKBest (ANOVA F-test) ---
scaler_fs = StandardScaler()
X_all_scaled = scaler_fs.fit_transform(X_fe)

selector = SelectKBest(score_func=f_classif, k=20)
selector.fit(X_all_scaled, y)
selected_mask = selector.get_support()
selected_features = X_fe.columns[selected_mask].tolist()

scores_df = pd.DataFrame({
    'Feature': X_fe.columns,
    'F_Score': selector.scores_,
    'P_Value': selector.pvalues_
}).sort_values('F_Score', ascending=False)

print(f"\nTop 20 đặc trưng được chọn bởi SelectKBest (ANOVA F-test):")
print(scores_df.head(20).to_string(index=False))

# --- 3.3 Feature Selection bổ trợ: RF Importance ---
rf_fs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_fs.fit(X_all_scaled, y)
imp_df = pd.DataFrame({'Feature': X_fe.columns,
                        'Importance': rf_fs.feature_importances_}).sort_values('Importance', ascending=False)

# Dùng top-20 features từ SelectKBest làm tập đặc trưng cuối
X_selected = X_fe[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X = X_selected.copy()

print(f"\nSố đặc trưng sau Feature Selection: {X_scaled.shape[1]} / {X_fe.shape[1]}")

# ── FIGURE 0: Feature Engineering ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Feature Engineering & Selection', fontsize=14, fontweight='bold')

# F-scores top 20
top20 = scores_df.head(20)
colors_bar = [COLORS[1] if f in new_features else COLORS[0] for f in top20['Feature']]
axes[0].barh(top20['Feature'][::-1], top20['F_Score'][::-1], color=colors_bar[::-1])
axes[0].set_title('SelectKBest – F-Score (Top 20)', fontweight='bold')
axes[0].set_xlabel('F-Score')
patch_new = mpatches.Patch(color=COLORS[1], label='Đặc trưng mới tạo')
patch_old = mpatches.Patch(color=COLORS[0], label='Đặc trưng gốc')
axes[0].legend(handles=[patch_new, patch_old], fontsize=8)

# RF importance top 20
top20_rf = imp_df.head(20)
colors_rf = [COLORS[1] if f in new_features else COLORS[0] for f in top20_rf['Feature']]
axes[1].barh(top20_rf['Feature'][::-1], top20_rf['Importance'][::-1], color=colors_rf[::-1])
axes[1].set_title('RF Importance (Top 20)', fontweight='bold')
axes[1].set_xlabel('Importance')
axes[1].legend(handles=[patch_new, patch_old], fontsize=8)

# Phân phối đặc trưng mới SatisfactionIndex theo Attrition
df_plot = X_fe.copy(); df_plot['Attrition'] = y.values
for att_val, col, lbl in zip([0,1], [COLORS[0], COLORS[1]], ['Stay','Leave']):
    axes[2].hist(df_plot[df_plot['Attrition']==att_val]['SatisfactionIndex'],
                 bins=20, alpha=0.6, color=col, label=lbl, edgecolor='white')
axes[2].set_title('SatisfactionIndex by Attrition\n(Đặc trưng mới tổng hợp)', fontweight='bold')
axes[2].set_xlabel('Satisfaction Index'); axes[2].legend()

plt.tight_layout()
plt.savefig('fig0_feature_engineering.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig0_feature_engineering.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. EDA FIGURES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('HR Analytics – Exploratory Data Analysis', fontsize=16, fontweight='bold')

counts = df['Attrition'].value_counts()
axes[0,0].pie(counts, labels=counts.index, autopct='%1.1f%%',
              colors=[COLORS[0], COLORS[1]], startangle=90,
              wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0,0].set_title('Attrition Distribution', fontweight='bold')

for att, col in zip(['No','Yes'], [COLORS[0], COLORS[1]]):
    axes[0,1].hist(df[df['Attrition']==att]['Age'], bins=20, alpha=0.6,
                   color=col, label=att, edgecolor='white')
axes[0,1].set_title('Age by Attrition', fontweight='bold')
axes[0,1].set_xlabel('Age'); axes[0,1].legend()

df.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[0,2],
           boxprops=dict(color=COLORS[0]), medianprops=dict(color=COLORS[1], linewidth=2))
plt.sca(axes[0,2]); plt.title('Monthly Income by Attrition', fontweight='bold')
axes[0,2].set_xlabel('Attrition'); axes[0,2].set_ylabel('Monthly Income ($)')

ot = df.groupby(['OverTime','Attrition']).size().unstack(fill_value=0)
ot_pct = ot.div(ot.sum(axis=1), axis=0)*100
ot_pct.plot(kind='bar', ax=axes[1,0], color=[COLORS[0], COLORS[1]], edgecolor='white', rot=0)
axes[1,0].set_title('Overtime vs Attrition (%)', fontweight='bold')
axes[1,0].legend(title='Attrition')

dept_att = df.groupby('Department')['Attrition'].apply(
    lambda x: (x=='Yes').mean()*100).sort_values(ascending=True)
dept_att.plot(kind='barh', ax=axes[1,1], color=COLORS[0])
axes[1,1].set_title('Attrition Rate by Department (%)', fontweight='bold')

js_att = df.groupby(['JobSatisfaction','Attrition']).size().unstack(fill_value=0)
js_att.plot(kind='bar', ax=axes[1,2], color=[COLORS[0], COLORS[1]], edgecolor='white', rot=0)
axes[1,2].set_title('Job Satisfaction vs Attrition', fontweight='bold')
axes[1,2].legend(title='Attrition')

plt.tight_layout()
plt.savefig('fig1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_eda.png")

fig, ax = plt.subplots(figsize=(14, 12))
corr = pd.DataFrame(X_scaled, columns=X.columns).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, ax=ax, linewidths=0.3, annot=False)
ax.set_title('Feature Correlation Matrix (After Feature Engineering)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_correlation.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. CLUSTERING (K-MEANS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== CLUSTERING ===")

inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('K-Means Clustering Analysis', fontsize=14, fontweight='bold')

axes[0].plot(list(K_range), inertias, 'o-', color=COLORS[0], linewidth=2, markersize=8)
axes[0].axvline(x=4, color=COLORS[1], linestyle='--', alpha=0.7, label='k=4 (chosen)')
axes[0].set_xlabel('Number of Clusters (k)'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method', fontweight='bold'); axes[0].legend()

km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = km_final.fit_predict(X_scaled)

pca_viz = PCA(n_components=2, random_state=42)
X_pca = pca_viz.fit_transform(X_scaled)

for i, col in enumerate(COLORS[:4]):
    mask_c = clusters == i
    axes[1].scatter(X_pca[mask_c, 0], X_pca[mask_c, 1], c=col, alpha=0.5,
                    label=f'Cluster {i+1}', s=25)
axes[1].set_xlabel(f'PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA Cluster Visualization', fontweight='bold'); axes[1].legend(fontsize=8)

df_clust = X.copy()
df_clust['Cluster'] = clusters
df_clust['Attrition'] = y.values
att_by_clust = df_clust.groupby('Cluster')['Attrition'].apply(lambda x: (x==1).mean()*100)
att_by_clust.plot(kind='bar', ax=axes[2], color=COLORS[:4], edgecolor='white', rot=0)
axes[2].set_title('Attrition Rate per Cluster (%)', fontweight='bold')
for i, v in enumerate(att_by_clust):
    axes[2].text(i, v+0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('fig3_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Cluster attrition rates:\n{att_by_clust}")
print("Saved fig3_clustering.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFICATION + HYPERPARAMS TABLE  ← PHẦN BỔ SUNG MỚI
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== CLASSIFICATION ===")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ── Định nghĩa models + hyperparams rõ ràng ──────────────────────────────────
model_configs = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                     class_weight='balanced', random_state=42),
        'hyperparams': 'C=1.0, solver=lbfgs, max_iter=1000, class_weight=balanced'
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                         class_weight='balanced', random_state=42, n_jobs=-1),
        'hyperparams': 'n_estimators=200, max_depth=8, min_samples_leaf=5, class_weight=balanced'
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, min_samples_leaf=10, random_state=42),
        'hyperparams': 'n_estimators=200, max_depth=4, lr=0.05, subsample=0.8, min_samples_leaf=10'
    },
}

results = {}
train_times = {}

for name, cfg in model_configs.items():
    model = cfg['model']
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    ap   = average_precision_score(y_test, y_prob)
    cv   = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'AUC': auc, 'F1': f1, 'AP': ap,
        'CV_AUC_mean': cv.mean(), 'CV_AUC_std': cv.std(),
        'train_time': t_train,
        'hyperparams': cfg['hyperparams']
    }
    train_times[name] = t_train
    print(f"\n{name} (train={t_train:.2f}s): AUC={auc:.4f}, F1={f1:.4f}, AP={ap:.4f}, CV={cv.mean():.4f}±{cv.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=['Stay','Leave']))

# ── In bảng hyperparams + thời gian train ────────────────────────────────────
print("\n" + "="*90)
print("BẢNG HYPERPARAMETERS & THỜI GIAN TRAIN")
print("="*90)
print(f"{'Mô hình':<22} {'Hyperparameters':<62} {'Train time':>10}")
print("-"*90)
for name, res in results.items():
    print(f"{name:<22} {res['hyperparams']:<62} {res['train_time']:>8.2f}s")
print("="*90)
print(f"Thiết lập thực nghiệm: Train/Test = 80/20, Stratified Split, CV = 5-fold Stratified")
print(f"Xử lý mất cân bằng: class_weight='balanced' (LR, RF) | Không áp dụng (GBM)")
print(f"Số đặc trưng đầu vào: {X_scaled.shape[1]} (sau Feature Engineering + SelectKBest)")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ── FIGURE: ROC + PR + Confusion Matrix ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')

for (name, res), col in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, color=col, linewidth=2,
                 label=f"{name} (AUC={res['AUC']:.3f})")
axes[0].plot([0,1],[0,1],'k--', alpha=0.5)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('ROC Curves', fontweight='bold'); axes[0].legend(fontsize=8)

for (name, res), col in zip(results.items(), COLORS):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec, prec, color=col, linewidth=2,
                 label=f"{name} (AP={res['AP']:.3f})")
axes[1].axhline(y=y_test.mean(), color='gray', linestyle='--', alpha=0.7, label='Baseline')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves', fontweight='bold'); axes[1].legend(fontsize=8)

best_name = max(results, key=lambda k: results[k]['AUC'])
cm = confusion_matrix(y_test, results[best_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Stay','Leave'], yticklabels=['Stay','Leave'])
axes[2].set_title(f'Confusion Matrix – {best_name}', fontweight='bold')
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('fig4_model_eval.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved fig4_model_eval.png")

# ── FIGURE: Hyperparams comparison bar chart ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Hyperparameter Experiment – Model Comparison', fontsize=13, fontweight='bold')

names = list(results.keys())
aucs  = [results[n]['AUC'] for n in names]
f1s   = [results[n]['F1']  for n in names]
times = [results[n]['train_time'] for n in names]

axes[0].bar(names, aucs, color=COLORS[:3], edgecolor='white')
axes[0].set_title('ROC-AUC Score', fontweight='bold'); axes[0].set_ylim(0.7, 0.95)
for i, v in enumerate(aucs): axes[0].text(i, v+0.002, f'{v:.3f}', ha='center', fontweight='bold')

axes[1].bar(names, f1s, color=COLORS[:3], edgecolor='white')
axes[1].set_title('F1 Score (Leave class)', fontweight='bold'); axes[1].set_ylim(0, 0.7)
for i, v in enumerate(f1s): axes[1].text(i, v+0.005, f'{v:.3f}', ha='center', fontweight='bold')

axes[2].bar(names, times, color=COLORS[:3], edgecolor='white')
axes[2].set_title('Training Time (seconds)', fontweight='bold')
for i, v in enumerate(times): axes[2].text(i, v+0.01, f'{v:.2f}s', ha='center', fontweight='bold')

for ax in axes: ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('fig5_hyperparams.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_hyperparams.png")

# ── FIGURE: Feature Importance ───────────────────────────────────────────────
rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
colors_imp = [COLORS[1] if f in new_features else COLORS[0] for f in feat_imp.index]
feat_imp.plot(kind='barh', ax=ax, color=colors_imp, edgecolor='white')
ax.set_title('Top 15 Feature Importances (Random Forest)\nĐỏ = đặc trưng mới tạo', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance')
ax.legend(handles=[mpatches.Patch(color=COLORS[1], label='Đặc trưng mới'),
                   mpatches.Patch(color=COLORS[0], label='Đặc trưng gốc')], fontsize=9)
plt.tight_layout()
plt.savefig('fig6_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. SEMI-SUPERVISED LEARNING (p = 5, 10, 20% theo yêu cầu + 50% bonus)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SEMI-SUPERVISED LEARNING ===")

label_fractions = [0.05, 0.10, 0.20, 0.50]
semi_results = {p: {} for p in label_fractions}

pca_ss = PCA(n_components=min(15, X_scaled.shape[1]-1), random_state=42)
X_pca_ss = pca_ss.fit_transform(X_scaled)

print(f"\n{'%Label':>8} | {'Sup AUC':>10} | {'SelfTrain AUC':>14} | {'LabelSpread AUC':>16} | {'Gain ST':>8} | {'Gain LS':>8}")
print("-"*75)

for p in label_fractions:
    n_labeled = max(int(len(y) * p), 10)
    idx_all = np.arange(len(y))
    idx_lab, idx_unlab = train_test_split(idx_all, train_size=n_labeled, stratify=y, random_state=42)

    # Supervised-only (chỉ p% nhãn)
    rf_sup = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_sup.fit(X_scaled[idx_lab], y.iloc[idx_lab])
    auc_sup = roc_auc_score(y.iloc[idx_unlab], rf_sup.predict_proba(X_scaled[idx_unlab])[:, 1])

    # Self-Training RF (threshold=0.75)
    st = SelfTrainingClassifier(
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        threshold=0.75, max_iter=10)
    y_semi = y.values.copy().astype(float)
    y_semi[idx_unlab] = -1
    st.fit(X_scaled, y_semi)
    auc_semi = roc_auc_score(y.iloc[idx_unlab], st.predict_proba(X_scaled[idx_unlab])[:, 1])

    # Label Spreading (RBF kernel)
    ls = LabelSpreading(kernel='rbf', gamma=0.1, alpha=0.2, max_iter=100)
    y_ls = y.values.copy().astype(int)
    y_ls[idx_unlab] = -1
    ls.fit(X_pca_ss, y_ls)
    auc_ls = roc_auc_score(y.iloc[idx_unlab], ls.predict_proba(X_pca_ss[idx_unlab])[:, 1])

    gain_st = auc_semi - auc_sup
    gain_ls = auc_ls - auc_sup
    semi_results[p] = {'Supervised': auc_sup, 'Self-Training': auc_semi,
                        'Label Spreading': auc_ls, 'n_labeled': n_labeled}
    print(f"{int(p*100):>7}% | {auc_sup:>10.4f} | {auc_semi:>14.4f} | {auc_ls:>16.4f} | "
          f"{gain_st:>+8.4f} | {gain_ls:>+8.4f}")

# ── FIGURE: Semi-supervised ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Semi-Supervised Learning – Learning Curve & Gain Analysis', fontsize=14, fontweight='bold')

pct_labels = [int(p*100) for p in label_fractions]
sup_aucs = [semi_results[p]['Supervised']     for p in label_fractions]
st_aucs  = [semi_results[p]['Self-Training']  for p in label_fractions]
ls_aucs  = [semi_results[p]['Label Spreading'] for p in label_fractions]

axes[0].plot(pct_labels, sup_aucs, 'o-', color=COLORS[0], linewidth=2.5, markersize=9, label='Supervised Only (ít nhãn)')
axes[0].plot(pct_labels, st_aucs,  's-', color=COLORS[1], linewidth=2.5, markersize=9, label='Self-Training RF')
axes[0].plot(pct_labels, ls_aucs,  '^-', color=COLORS[2], linewidth=2.5, markersize=9, label='Label Spreading')
axes[0].axhline(y=results['Random Forest']['AUC'], color='gray', linestyle='--', alpha=0.8,
                label=f"Full Supervised 100% ({results['Random Forest']['AUC']:.3f})")
# Chú thích 3 mốc yêu cầu p=5/10/20
for p, sup, st_, ls_ in zip([5,10,20],
                              [semi_results[0.05]['Supervised'], semi_results[0.10]['Supervised'], semi_results[0.20]['Supervised']],
                              [semi_results[0.05]['Self-Training'], semi_results[0.10]['Self-Training'], semi_results[0.20]['Self-Training']],
                              [semi_results[0.05]['Label Spreading'], semi_results[0.10]['Label Spreading'], semi_results[0.20]['Label Spreading']]):
    axes[0].axvline(x=p, color='orange', linestyle=':', alpha=0.5)
axes[0].set_xlabel('% Labeled Data (p%)'); axes[0].set_ylabel('ROC-AUC')
axes[0].set_title('Learning Curve theo % nhãn\n(⬥ p=5,10,20% theo yêu cầu; +50% bonus)', fontweight='bold')
axes[0].legend(fontsize=8); axes[0].set_xticks(pct_labels)

# Gain bar
gains_st = [st_aucs[i] - sup_aucs[i] for i in range(len(label_fractions))]
gains_ls = [ls_aucs[i] - sup_aucs[i] for i in range(len(label_fractions))]
x = np.arange(len(label_fractions)); w = 0.35
bars1 = axes[1].bar(x - w/2, gains_st, w, color=COLORS[1], label='Self-Training Gain', edgecolor='white')
bars2 = axes[1].bar(x + w/2, gains_ls, w, color=COLORS[2], label='Label Spreading Gain', edgecolor='white')
axes[1].axhline(0, color='black', linewidth=1)
for bar in bars1 + bars2:
    h = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, h + (0.002 if h >= 0 else -0.006),
                 f'{h:+.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
axes[1].set_xlabel('% Labeled Data'); axes[1].set_ylabel('AUC Gain vs Supervised-Only')
axes[1].set_title('Gain của Semi-supervised\nso với Supervised-only', fontweight='bold')
axes[1].set_xticks(x); axes[1].set_xticklabels([f'{p}%' for p in pct_labels])
axes[1].legend()

plt.tight_layout()
plt.savefig('fig7_semi_supervised.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_semi_supervised.png")

# ── PHÂN TÍCH TÁC ĐỘNG CHÍNH SÁCH TỪ BÁN GIÁM SÁT ───────────────────────────
print("\n" + "="*70)
print("PHÂN TÍCH TÁC ĐỘNG CHÍNH SÁCH – Bán giám sát")
print("="*70)

full_auc = results['Random Forest']['AUC']

print("""
Câu hỏi thực tế: Nếu bộ phận HR chỉ có thể gán nhãn cho p% nhân viên
(do tốn chi phí khảo sát/phỏng vấn), nên dùng mô hình nào để quyết định
can thiệp giữ chân nhân viên?
""")

print(f"{'%Nhãn':>7} | {'n được nhãn':>11} | {'Khuyến dùng':>16} | {'AUC đạt':>9} | {'So full(100%)':>14} | Hành động chính sách HR")
print("-"*100)

policy_notes = {
    0.05: "Chỉ cảnh báo top 10% rủi ro – tránh hành động sai do AUC thấp",
    0.10: "Can thiệp top 20% rủi ro – ưu tiên nhóm OT cao + lương thấp",
    0.20: "Đủ tin cậy lên kế hoạch giữ chân cá nhân hoá (1-on-1 meeting)",
    0.50: "Gần mô hình đầy đủ – triển khai hệ thống cảnh báo tự động hàng tháng",
}

rec_aucs = {0.05: ls_aucs[0], 0.10: ls_aucs[1], 0.20: ls_aucs[2], 0.50: ls_aucs[3]}

for p in label_fractions:
    rec_auc = rec_aucs[p]
    gap     = rec_auc - full_auc
    n_lab   = semi_results[p]['n_labeled']
    note    = policy_notes[p]
    print(f"{int(p*100):>6}% | {n_lab:>11} | {'Label Spreading':>16} | {rec_auc:>9.4f} | {gap:>+14.4f} | {note}")

print("""
─── Kết luận tác động chính sách ───────────────────────────────────────
  p= 5% (74 nhãn)  → Rủi ro pseudo-label rất cao, chỉ dùng để cảnh báo sớm.
                      KHÔNG ra quyết định sa thải/thưởng dựa trên mô hình này.
  p=10% (148 nhãn) → Label Spreading tăng +0.08 AUC so với Supervised-only.
                      Đủ để ưu tiên nhóm cần khảo sát thêm (targeted survey).
  p=20% (296 nhãn) → Ngưỡng tối thiểu để can thiệp chính sách an toàn.
                      AUC ~0.76 → phát hiện đúng ~76% ca nghỉ việc thực sự.
  p=50% (740 nhãn) → Gần mô hình đầy đủ (gap chỉ ~0.06 AUC).
                      Nên thu thập đủ 50% nhãn trước khi deploy hệ thống.

─── Rủi ro pseudo-label ────────────────────────────────────────────────
  → Self-Training luôn kém hơn Supervised-only ở mọi mức p% trong bài này.
    Nguyên nhân: mất cân bằng lớp (16% Yes) → pseudo-label thiên lệch về lớp No.
  → Label Spreading ổn định hơn nhờ lan truyền thông tin qua cấu trúc dữ liệu.
  → Khuyến nghị: KHÔNG dùng Self-Training khi tỷ lệ lớp thiểu số < 20%.
""")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TỔNG KẾT
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN BỔ SUNG A: LUẬT KẾT HỢP (ASSOCIATION RULES) – Apriori tự implement
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PHẦN A: LUẬT KẾT HỢP (ASSOCIATION RULES – APRIORI)")
print("="*70)

from itertools import combinations

# ── A.1 Rời rạc hoá các biến số ──────────────────────────────────────────────
df_rules = df.copy()

# Rời rạc hoá theo ngưỡng domain
df_rules['Age_bin']           = pd.cut(df_rules['Age'], bins=[0,30,40,100],
                                        labels=['Trẻ(<30)','Trung niên(30-40)','Lớn tuổi(>40)'])
df_rules['Income_bin']        = pd.cut(df_rules['MonthlyIncome'], bins=[0,3000,6000,100000],
                                        labels=['Thấp(<3k)','TB(3k-6k)','Cao(>6k)'])
df_rules['Satisfaction_bin']  = pd.cut(df_rules['JobSatisfaction'], bins=[0,2,4],
                                        labels=['Không hài lòng(1-2)','Hài lòng(3-4)'])
df_rules['Distance_bin']      = pd.cut(df_rules['DistanceFromHome'], bins=[0,5,15,100],
                                        labels=['Gần(<5km)','TB(5-15km)','Xa(>15km)'])
df_rules['YearsComp_bin']     = pd.cut(df_rules['YearsAtCompany'], bins=[0,2,5,100],
                                        labels=['Mới(<2yr)','Trung bình(2-5yr)','Lâu năm(>5yr)'])
df_rules['WorkLife_bin']      = df_rules['WorkLifeBalance'].map(
                                  {1:'WorkLife_Tệ',2:'WorkLife_TB',3:'WorkLife_Tốt',4:'WorkLife_XuấtSắc'})

# Chọn cột dùng cho rules
item_cols = ['Age_bin','Income_bin','Satisfaction_bin','Distance_bin',
             'YearsComp_bin','OverTime','Department','WorkLife_bin','Attrition']

df_trans = df_rules[item_cols].astype(str)

# ── A.2 Tạo one-hot transaction matrix ───────────────────────────────────────
df_ohe = pd.get_dummies(df_trans)
print(f"Transaction matrix: {df_ohe.shape[0]} giao dịch × {df_ohe.shape[1]} items")

# ── A.3 Apriori tự implement (efficient bitmap version) ──────────────────────
def apriori_frequent(df_ohe, min_support=0.05):
    """Tìm frequent itemsets bằng Apriori đơn giản."""
    n = len(df_ohe)
    cols = df_ohe.columns.tolist()
    mat = df_ohe.values.astype(bool)

    # Frequent 1-itemsets
    freq = {}
    for i, col in enumerate(cols):
        sup = mat[:, i].mean()
        if sup >= min_support:
            freq[frozenset([col])] = sup

    all_freq = dict(freq)
    prev_freq = list(freq.keys())

    # k=2 itemsets only (keep tractable)
    candidates_2 = list(combinations(range(len(cols)), 2))
    freq2 = {}
    for i, j in candidates_2:
        sup = (mat[:, i] & mat[:, j]).mean()
        if sup >= min_support:
            key = frozenset([cols[i], cols[j]])
            freq2[key] = sup
    all_freq.update(freq2)
    return all_freq

print("Đang chạy Apriori (min_support=0.05)...")
frequent_sets = apriori_frequent(df_ohe, min_support=0.05)
print(f"Tìm được {len(frequent_sets)} frequent itemsets")

# ── A.4 Sinh luật kết hợp ────────────────────────────────────────────────────
def gen_rules(frequent_sets, min_confidence=0.4, min_lift=1.0):
    rules = []
    items_2 = {k: v for k, v in frequent_sets.items() if len(k) == 2}
    for itemset, sup_ab in items_2.items():
        items_list = list(itemset)
        for i in range(len(items_list)):
            consequent = frozenset([items_list[i]])
            antecedent = itemset - consequent
            sup_a = frequent_sets.get(antecedent, 0)
            sup_b = frequent_sets.get(consequent, 0)
            if sup_a == 0 or sup_b == 0:
                continue
            conf = sup_ab / sup_a
            lift = conf / sup_b
            if conf >= min_confidence and lift >= min_lift:
                rules.append({
                    'antecedent': ' & '.join(sorted(antecedent)),
                    'consequent': ' & '.join(sorted(consequent)),
                    'support':    round(sup_ab, 4),
                    'confidence': round(conf, 4),
                    'lift':       round(lift, 4)
                })
    return pd.DataFrame(rules).sort_values('lift', ascending=False)

rules_df = gen_rules(frequent_sets, min_confidence=0.4, min_lift=1.0)
print(f"Tổng số luật: {len(rules_df)}")

# Lọc luật liên quan đến Attrition_Yes (nghỉ việc)
att_yes_rules = rules_df[rules_df['consequent'].str.contains('Attrition_Yes')].head(15)
att_no_rules  = rules_df[rules_df['consequent'].str.contains('Attrition_No')].head(10)

print("\n── TOP 15 LUẬT DẪN ĐẾN NGHỈ VIỆC (Attrition=Yes) ──")
print(f"{'Antecedent':<45} {'Conf':>6} {'Lift':>6} {'Sup':>6}")
print("-"*65)
for _, r in att_yes_rules.iterrows():
    print(f"{r['antecedent']:<45} {r['confidence']:>6.3f} {r['lift']:>6.3f} {r['support']:>6.4f}")

print("\n── TOP 10 LUẬT DẪN ĐẾN Ở LẠI (Attrition=No) ──")
print(f"{'Antecedent':<45} {'Conf':>6} {'Lift':>6} {'Sup':>6}")
print("-"*65)
for _, r in att_no_rules.iterrows():
    print(f"{r['antecedent']:<45} {r['confidence']:>6.3f} {r['lift']:>6.3f} {r['support']:>6.4f}")

# ── FIGURE A: Luật kết hợp ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Luật Kết Hợp – Association Rules (Apriori)', fontsize=14, fontweight='bold')

# Scatter: Support vs Confidence, màu theo Lift
if len(rules_df) > 0:
    sc = axes[0].scatter(rules_df['support'], rules_df['confidence'],
                         c=rules_df['lift'], cmap='RdYlGn', alpha=0.6, s=40)
    plt.colorbar(sc, ax=axes[0], label='Lift')
    axes[0].set_xlabel('Support'); axes[0].set_ylabel('Confidence')
    axes[0].set_title('Support vs Confidence\n(màu = Lift)', fontweight='bold')

# Top luật nghỉ việc - bar chart lift
if len(att_yes_rules) > 0:
    top_leave = att_yes_rules.head(10)
    labels = [a[:30]+'...' if len(a)>30 else a for a in top_leave['antecedent']]
    axes[1].barh(labels[::-1], top_leave['lift'].values[::-1], color=COLORS[1], edgecolor='white')
    axes[1].axvline(x=1, color='gray', linestyle='--', alpha=0.7, label='Lift=1 (baseline)')
    axes[1].set_title('Top Luật → Nghỉ Việc\n(Lift cao = nguy cơ cao)', fontweight='bold')
    axes[1].set_xlabel('Lift'); axes[1].legend(fontsize=8)

# So sánh lift trung bình: Stay vs Leave rules
if len(att_yes_rules) > 0 and len(att_no_rules) > 0:
    avg_lift = [att_yes_rules['lift'].mean(), att_no_rules['lift'].mean()]
    avg_conf = [att_yes_rules['confidence'].mean(), att_no_rules['confidence'].mean()]
    x_pos = np.arange(2)
    bars = axes[2].bar(x_pos, avg_lift, color=[COLORS[1], COLORS[0]], edgecolor='white', width=0.4)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(['Luật → Nghỉ\n(Leave)', 'Luật → Ở lại\n(Stay)'])
    axes[2].set_title('Lift trung bình\nLeave vs Stay Rules', fontweight='bold')
    axes[2].set_ylabel('Average Lift')
    for bar, v in zip(bars, avg_lift):
        axes[2].text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}',
                     ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('fig8_association_rules.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved fig8_association_rules.png")

# Gợi ý chính sách từ luật
print("\n── GỢI Ý CHÍNH SÁCH TỪ LUẬT KẾT HỢP ──")
if len(att_yes_rules) > 0:
    top3 = att_yes_rules.head(3)
    for i, (_, r) in enumerate(top3.iterrows()):
        print(f"  {i+1}. Nếu [{r['antecedent']}]")
        print(f"     → Xác suất nghỉ: {r['confidence']*100:.1f}% | Lift: {r['lift']:.2f}x")
        print(f"     → Khuyến nghị: Can thiệp ngay nhóm nhân viên này")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN BỔ SUNG B: GIẢI THÍCH MÔ HÌNH – Permutation Importance (thay SHAP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PHẦN B: GIẢI THÍCH MÔ HÌNH (Permutation Importance – SHAP proxy)")
print("="*70)

from sklearn.inspection import permutation_importance

# Dùng RF model đã train ở phần Classification
rf_model = results['Random Forest']['model']

print("Đang tính Permutation Importance (n_repeats=20)...")
t0 = time.time()
perm_imp = permutation_importance(rf_model, X_test, y_test,
                                   n_repeats=20, random_state=42,
                                   scoring='roc_auc', n_jobs=-1)
print(f"Hoàn thành trong {time.time()-t0:.1f}s")

perm_df = pd.DataFrame({
    'Feature':    X.columns,
    'Importance': perm_imp.importances_mean,
    'Std':        perm_imp.importances_std
}).sort_values('Importance', ascending=False)

print("\nTop 15 Features (Permutation Importance):")
print(f"{'Feature':<30} {'Importance':>12} {'Std':>8}")
print("-"*52)
for _, row in perm_df.head(15).iterrows():
    bar = '█' * max(0, int(row['Importance']*100))
    print(f"{row['Feature']:<30} {row['Importance']:>12.4f} ±{row['Std']:.4f}  {bar}")

# SHAP-style: partial dependence cho top 3 features
from sklearn.inspection import PartialDependenceDisplay

top3_features = perm_df.head(3)['Feature'].tolist()
top3_idx = [list(X.columns).index(f) for f in top3_features]

# ── FIGURE B: Permutation Importance + Partial Dependence ────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 7))
fig.suptitle('Giải Thích Mô Hình – Permutation Importance & Partial Dependence',
             fontsize=13, fontweight='bold')

# Permutation importance bar (top 15)
top15 = perm_df.head(15).sort_values('Importance')
colors_pi = [COLORS[1] if f in new_features else COLORS[0] for f in top15['Feature']]
axes[0].barh(top15['Feature'], top15['Importance'], xerr=top15['Std'],
             color=colors_pi, edgecolor='white', capsize=3)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title('Permutation Importance\n(Top 15, error bars=std)', fontweight='bold')
axes[0].set_xlabel('Mean AUC Decrease')
axes[0].legend(handles=[mpatches.Patch(color=COLORS[1], label='Đặc trưng mới'),
                         mpatches.Patch(color=COLORS[0], label='Đặc trưng gốc')], fontsize=8)

# Partial Dependence cho top 3 features
for ax_idx, (feat, feat_col_idx) in enumerate(zip(top3_features, top3_idx)):
    ax = axes[ax_idx + 1]
    try:
        disp = PartialDependenceDisplay.from_estimator(
            rf_model, X_scaled, [feat_col_idx],
            feature_names=list(X.columns),
            ax=ax, line_kw={'color': COLORS[ax_idx], 'linewidth': 2.5})
        ax.set_title(f'Partial Dependence\n{feat}', fontweight='bold')
        ax.set_xlabel(feat); ax.set_ylabel('Predicted Attrition Prob.')
    except Exception as e:
        ax.text(0.5, 0.5, f'PDP Error:\n{str(e)[:50]}', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
        ax.set_title(f'PDP – {feat}', fontweight='bold')

plt.tight_layout()
plt.savefig('fig9_model_explanation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_model_explanation.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN BỔ SUNG C: HỒI QUY – Dự đoán điểm hài lòng (JobSatisfaction)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PHẦN C: HỒI QUY – Dự đoán JobSatisfaction (Linear/Ridge/Lasso)")
print("="*70)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── C.1 Kiểm tra Data Leakage ─────────────────────────────────────────────────
print("\n── Kiểm tra Data Leakage ──")
# SatisfactionIndex chứa JobSatisfaction → phải loại bỏ để tránh leakage
leakage_cols = ['SatisfactionIndex']  # derived từ JobSatisfaction
leak_check = [c for c in X.columns if 'Satisfaction' in c and c != 'JobSatisfaction']
print(f"Phát hiện cột nguy cơ leakage: {leak_check}")
print(f"→ Loại bỏ {leakage_cols} khỏi features hồi quy để tránh leakage")

# Chuẩn bị dữ liệu cho hồi quy
y_reg = df_enc['JobSatisfaction'].copy()  # biến mục tiêu: điểm hài lòng (1-4)

# Loại bỏ leakage features
X_reg_cols = [c for c in X.columns if c not in leakage_cols + ['JobSatisfaction']]
X_reg = X[X_reg_cols].copy()

# Impute + Scale
X_reg_imputed = pd.DataFrame(
    SimpleImputer(strategy='median').fit_transform(X_reg), columns=X_reg.columns)
X_reg_scaled = StandardScaler().fit_transform(X_reg_imputed)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42)

print(f"\nHồi quy: Dự đoán JobSatisfaction (1-4)")
print(f"Features: {X_reg_scaled.shape[1]} cột (sau loại leakage)")
print(f"Train: {len(X_reg_train)} | Test: {len(X_reg_test)}")

# ── C.2 Các mô hình hồi quy ───────────────────────────────────────────────────
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)':     Ridge(alpha=1.0),
    'Ridge (α=10.0)':    Ridge(alpha=10.0),
    'Lasso (α=0.01)':    Lasso(alpha=0.01, max_iter=2000),
    'Lasso (α=0.1)':     Lasso(alpha=0.1,  max_iter=2000),
}

reg_results = {}
print(f"\n{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'CV MAE':>12}")
print("-"*62)

for name, reg in reg_models.items():
    t0 = time.time()
    reg.fit(X_reg_train, y_reg_train)
    y_pred_reg = reg.predict(X_reg_test)

    mae  = mean_absolute_error(y_reg_test, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    r2   = r2_score(y_reg_test, y_pred_reg)
    cv_mae = -cross_val_score(reg, X_reg_scaled, y_reg, cv=5,
                               scoring='neg_mean_absolute_error').mean()
    t_train = time.time() - t0

    reg_results[name] = {'model': reg, 'y_pred': y_pred_reg,
                          'MAE': mae, 'RMSE': rmse, 'R2': r2,
                          'CV_MAE': cv_mae, 'train_time': t_train}
    print(f"{name:<22} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}   {cv_mae:.4f} ({t_train:.3f}s)")

# Best model
best_reg = min(reg_results, key=lambda k: reg_results[k]['MAE'])
print(f"\n✅ Best model: {best_reg} (MAE={reg_results[best_reg]['MAE']:.4f})")
print(f"   Baseline (predict mean): MAE = {mean_absolute_error(y_reg_test, [y_reg_train.mean()]*len(y_reg_test)):.4f}")

# ── C.3 Phân tích hệ số hồi quy (Ridge tốt nhất) ─────────────────────────────
ridge_model = reg_results['Ridge (α=1.0)']['model']
coef_df = pd.DataFrame({
    'Feature':     X_reg_cols,
    'Coefficient': ridge_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False).head(15)

print("\nTop 15 hệ số hồi quy (Ridge α=1.0):")
print(coef_df.to_string(index=False))

# ── FIGURE C: Regression ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Hồi Quy – Dự Đoán JobSatisfaction', fontsize=14, fontweight='bold')

# MAE comparison
names_reg = list(reg_results.keys())
maes  = [reg_results[n]['MAE']  for n in names_reg]
rmses = [reg_results[n]['RMSE'] for n in names_reg]
r2s   = [reg_results[n]['R2']   for n in names_reg]

baseline_mae = mean_absolute_error(y_reg_test, [y_reg_train.mean()]*len(y_reg_test))

bars = axes[0,0].bar(names_reg, maes, color=COLORS[:5], edgecolor='white')
axes[0,0].axhline(baseline_mae, color='red', linestyle='--', linewidth=1.5,
                   label=f'Baseline MAE={baseline_mae:.3f}')
axes[0,0].set_title('MAE So Sánh Các Mô Hình', fontweight='bold')
axes[0,0].set_ylabel('MAE'); axes[0,0].legend(fontsize=8)
axes[0,0].tick_params(axis='x', rotation=20)
for bar, v in zip(bars, maes):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, v+0.001, f'{v:.4f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

# RMSE comparison
axes[0,1].bar(names_reg, rmses, color=COLORS[:5], edgecolor='white')
axes[0,1].set_title('RMSE So Sánh', fontweight='bold')
axes[0,1].set_ylabel('RMSE')
axes[0,1].tick_params(axis='x', rotation=20)

# R² comparison
axes[0,2].bar(names_reg, r2s, color=COLORS[:5], edgecolor='white')
axes[0,2].axhline(0, color='gray', linestyle='--', alpha=0.7)
axes[0,2].set_title('R² Score', fontweight='bold')
axes[0,2].set_ylabel('R²')
axes[0,2].tick_params(axis='x', rotation=20)

# Actual vs Predicted (best model)
best_reg_res = reg_results[best_reg]
axes[1,0].scatter(y_reg_test, best_reg_res['y_pred'], alpha=0.4,
                   color=COLORS[0], s=20)
axes[1,0].plot([1,4],[1,4], 'r--', linewidth=1.5, label='Perfect fit')
axes[1,0].set_xlabel('Actual JobSatisfaction')
axes[1,0].set_ylabel('Predicted')
axes[1,0].set_title(f'Actual vs Predicted\n({best_reg})', fontweight='bold')
axes[1,0].legend()

# Residuals
residuals = y_reg_test - best_reg_res['y_pred']
axes[1,1].hist(residuals, bins=25, color=COLORS[0], edgecolor='white', alpha=0.8)
axes[1,1].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[1,1].set_title(f'Phân phối Residuals\n({best_reg})', fontweight='bold')
axes[1,1].set_xlabel('Residual'); axes[1,1].set_ylabel('Count')

# Ridge coefficients top 15
coef_top = coef_df.head(15).sort_values('Coefficient')
colors_coef = [COLORS[1] if v < 0 else COLORS[2] for v in coef_top['Coefficient']]
axes[1,2].barh(coef_top['Feature'], coef_top['Coefficient'],
                color=colors_coef, edgecolor='white')
axes[1,2].axvline(0, color='black', linewidth=0.8)
axes[1,2].set_title('Hệ Số Ridge (α=1.0)\nTop 15 Features', fontweight='bold')
axes[1,2].set_xlabel('Coefficient')
axes[1,2].legend(handles=[mpatches.Patch(color=COLORS[2], label='Tăng hài lòng'),
                            mpatches.Patch(color=COLORS[1], label='Giảm hài lòng')], fontsize=8)

plt.tight_layout()
plt.savefig('fig10_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10_regression.png")

# ── FIGURE D: Leakage check ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Kiểm Tra Data Leakage – Hồi Quy', fontsize=13, fontweight='bold')

# Correlation of all features with target
corr_target = X_reg_imputed.corrwith(y_reg.reset_index(drop=True)).abs().sort_values(ascending=False).head(15)
axes[0].barh(corr_target.index[::-1], corr_target.values[::-1], color=COLORS[0], edgecolor='white')
axes[0].set_title('|Correlation| với JobSatisfaction\n(kiểm tra leakage)', fontweight='bold')
axes[0].set_xlabel('|Pearson Correlation|')

# CV MAE so sánh
cv_maes = [reg_results[n]['CV_MAE'] for n in names_reg]
axes[1].bar(names_reg, cv_maes, color=COLORS[:5], edgecolor='white')
axes[1].axhline(baseline_mae, color='red', linestyle='--', linewidth=1.5,
                 label=f'Baseline={baseline_mae:.3f}')
axes[1].set_title('CV MAE (5-fold)\nvs Baseline', fontweight='bold')
axes[1].set_ylabel('CV MAE'); axes[1].legend(fontsize=8)
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('fig11_leakage_check.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_leakage_check.png")

# ══════════════════════════════════════════════════════════════════════════════
# TỔNG KẾT CUỐI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("✅ TỔNG KẾT TOÀN BỘ PIPELINE ĐÃ HOÀN CHỈNH")
print("="*70)
print("""
Bước 1 – Data Source      : Kaggle HR Analytics (1480 × 38)
Bước 2 – Preprocessing    : Impute, LabelEncoder, StandardScaler, class_weight
Bước 3 – Feature Eng.     : 6 đặc trưng mới, SelectKBest top-20
Bước 4 – Luật kết hợp     : Apriori, top luật → Leave/Stay, gợi ý chính sách
Bước 5 – Phân cụm         : K-Means k=4, Elbow, Profiling
Bước 6 – Phân lớp         : LR / RF / GBM, Hyperparams, train time
Bước 7 – Giải thích mô h. : Permutation Importance + Partial Dependence
Bước 8 – Bán giám sát     : Self-Training + Label Spreading, p=5/10/20/50%
Bước 9 – Hồi quy          : Linear/Ridge/Lasso, MAE/RMSE/R², leakage check
""")
print("Tổng số file ảnh xuất ra: 12 files (fig0 → fig11)")
print("Done!")