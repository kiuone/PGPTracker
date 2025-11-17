import scipy.stats as ss
import numpy as np

# Dados brutos extraídos da fixture 'df_stats_long' / 'df_wide_N_D_stats'
# Feature: PGPT_2 (O caso que falhou)
# S1, S2, S3 (Group A) -> [1.0, 10.0, 2.0]
# S4, S5, S6 (Group B) -> [9.0, 3.0, 11.0]

group_a = [1.0, 10.0, 2.0]
group_b = [9.0, 3.0, 11.0]

print("--- DADOS ---")
print(f"Grupo A: {group_a}")
print(f"Grupo B: {group_b}")
print("-" * 20)

# 1. Verificação do Kruskal-Wallis
# O Kruskal-Wallis usa rankings.
# Rankings conjuntos: 
# 1.0 (1), 2.0 (2), 3.0 (3), 9.0 (4), 10.0 (5), 11.0 (6)
# Ranks A: 1, 5, 2 (Soma = 8)
# Ranks B: 4, 3, 6 (Soma = 13)
stat, pval = ss.kruskal(group_a, group_b)

print(f"\n[Kruskal-Wallis] Scipy Reference:")
print(f"  Statistic esperado: {stat}")
print(f"  P-value esperado:   {pval}")

# Comparação com os valores da disputa:
print(f"\n  Valor do Código (Actual): 0.2752335...")
print(f"  Valor do Teste Antigo:    0.48839...")

if abs(pval - 0.2752335) < 0.0001:
    print("\n>> VEREDITO: O CÓDIGO está certo. O valor antigo do teste estava errado.")
elif abs(pval - 0.48839) < 0.0001:
    print("\n>> VEREDITO: O CÓDIGO está errado. O valor antigo do teste estava certo.")
else:
    print("\n>> VEREDITO: AMBOS estão errados.")

# 2. Verificação do Mann-Whitney U
# Teste bicaudal (two-sided)
stat_mwu, pval_mwu = ss.mannwhitneyu(group_a, group_b, alternative='two-sided')

print(f"\n[Mann-Whitney U] Scipy Reference:")
print(f"  P-value esperado:   {pval_mwu}")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import scipy.stats as ss
from skbio.diversity import alpha_diversity
from scipy.spatial.distance import pdist, squareform

print("=== AUDITORIA MATEMÁTICA DOS TESTES ===\n")

# --- 1. DIVERSITY (test_diversity.py) ---
print("--- 1. Diversity (Shannon & Bray-Curtis) ---")
# Dados do teste: S1=[10, 10, 0], S3=[1, 1, 1]
s1_counts = np.array([10, 10, 0])
s3_counts = np.array([1, 1, 1])

# Shannon Base 2 (O que seu projeto exige)
# Formula: -sum(p * log2(p))
p_s1 = s1_counts[s1_counts > 0] / s1_counts.sum()
shannon_s1 = -np.sum(p_s1 * np.log2(p_s1))

p_s3 = s3_counts / s3_counts.sum()
shannon_s3 = -np.sum(p_s3 * np.log2(p_s3))

print(f"Shannon S1 (Esperado 1.0): {shannon_s1:.5f}")
print(f"Shannon S3 (Esperado 1.58496): {shannon_s3:.5f}")

# Bray-Curtis S1 vs S2 ([0, 5, 5])
# Formula: sum(|u - v|) / sum(u + v)
s2_counts = np.array([0, 5, 5])
u_v_diff = np.abs(s1_counts - s2_counts).sum()
u_v_sum = (s1_counts + s2_counts).sum()
bc_dist = u_v_diff / u_v_sum
print(f"Bray-Curtis S1 vs S2 (Esperado 0.66667): {bc_dist:.5f}")
if abs(bc_dist - 0.666666) > 1e-5:
    print(">> ALERTA: Bray-Curtis suspeito!")

print("-" * 30)

# --- 2. STATISTICS (test_statistics.py) ---
print("\n--- 2. Statistics (Kruskal & Mann-Whitney) ---")
# Dados exatos da fixture que falhou
group_a = [1.0, 10.0, 2.0]
group_b = [9.0, 3.0, 11.0]
stat_kw, p_kw = ss.kruskal(group_a, group_b)
stat_mwu, p_mwu = ss.mannwhitneyu(group_a, group_b, alternative='two-sided')

print(f"Kruskal-Wallis P-value REAL: {p_kw:.7f}")
print(f"  -> Valor antigo (falho): 0.48839")
print(f"  -> Valor novo (código):  0.27523")

print(f"Mann-Whitney U P-value REAL: {p_mwu:.7f}")
print(f"  -> Valor antigo (falho): 0.8")
print(f"  -> Valor novo (código):  0.4")

print("-" * 30)

# --- 3. ORDINATION (test_ordination.py) ---
print("\n--- 3. Ordination (PCA & t-SNE) ---")
# Dados da fixture df_clr_wide
# S1: [0.5, -0.1, -0.4]
# S2: [-0.2, 0.3, -0.1]
# S3: [0.1, -0.3, 0.2]
X = np.array([
    [0.5, -0.1, -0.4],
    [-0.2, 0.3, -0.1],
    [0.1, -0.3, 0.2]
])

# PCA Check
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
s1_pc1 = X_pca[0, 0]
print(f"PCA S1_PC1 REAL (sklearn): {s1_pc1:.6f}")
print(f"  -> Valor no teste: 0.441112")

# t-SNE Check (O mais problemático)
# t-SNE é muito sensível a versões. Vamos ver o que o sklearn puro diz.
try:
    tsne = TSNE(n_components=2, perplexity=2.0, metric='euclidean', init='pca', learning_rate='auto', random_state=42)
    X_tsne = tsne.fit_transform(X)
    s1_tsne1 = X_tsne[0, 0]
    print(f"t-SNE S1_Dim1 REAL (sklearn): {s1_tsne1:.6f}")
    print(f"  -> Valor alucinado antigo: -74.2462")
    print(f"  -> Valor do log de erro:   87.0057")
except Exception as e:
    print(f"Erro ao rodar t-SNE localmente: {e}")

print("-" * 30)

# --- 4. MACHINE LEARNING (test_machine_leaning.py) ---
print("\n--- 4. Machine Learning (Lasso & RF logic) ---")
# Dados NOVOS propostos (Sinal forte)
# A e C são redundantes e preditivos. B é ruído puro.
X_ml = np.array([
    [5.0, 0.01, 4.9],   # S1 (Grupo A / Valor 20)
    [5.1, -0.02, 5.0],  # S2
    [5.0, 0.0, 5.1],    # S3
    [-5.0, -0.01, -4.9],# S4 (Grupo B / Valor -20)
    [-5.1, 0.02, -5.0], # S5
    [-5.0, 0.01, -5.1]  # S6
])
y_reg = np.array([20.1, 20.3, 19.9, -19.8, -20.2, -20.0])
y_class = np.array([0, 0, 0, 1, 1, 1])

# Lasso Check
# O Lasso deve zerar uma das features colineares (A ou C) e zerar B.
lasso = LassoCV(cv=3, random_state=42).fit(X_ml, y_reg)
print(f"Lasso Coefs: A={lasso.coef_[0]:.2f}, B={lasso.coef_[1]:.2f}, C={lasso.coef_[2]:.2f}")
if abs(lasso.coef_[1]) > 0.1:
    print(">> ALERTA: Lasso não zerou o ruído (B)!")
else:
    print(">> OK: Lasso zerou o ruído.")

# RF Check
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1).fit(X_ml, y_class)
imps = rf.feature_importances_
print(f"RF Importances: A={imps[0]:.2f}, B={imps[1]:.2f}, C={imps[2]:.2f}")
if imps[1] > 0.2:
    print(">> ALERTA: RF deu muita importância para o ruído (B)!")
else:
    print(f">> OK: Ruído B ({imps[1]:.2f}) é baixo.")

print("\n=== FIM DA AUDITORIA ===")