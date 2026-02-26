import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo

print("Iniciando a execução do script...\n")

print("--- Parte 1: Carregando os Dados ---")

iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

df = X.copy()
df['target'] = y.iloc[:, 0]

print("\nPrimeiras 5 linhas do dataset:")
print(df.head())

print("\n--- Parte 2: Entendendo a Estrutura ---")
print("\nInformações Gerais (df.info):")
df.info()

print("\nEstatísticas Descritivas (df.describe):")
print(df.describe())

print("\n--- Parte 3: Limpeza de Dados ---")
print("\nValores Nulos por coluna:")
print(df.isnull().sum())

num_duplicados = df.duplicated().sum()
print(f"\nNúmero de registros duplicados: {num_duplicados}")

print("\nDistribuição das classes (target):")
print(df["target"].value_counts())

print("\n--- Parte 4: Visualizações (Feche as janelas dos gráficos para continuar) ---")

df.drop('target', axis=1).hist(figsize=(10, 8), color='skyblue', edgecolor='black')
plt.suptitle('Histogramas das Variáveis', fontsize=16)
plt.show()

plt.figure(figsize=(8, 6))
matriz_correlacao = df.drop('target', axis=1).corr()
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap de Correlação', fontsize=16)
plt.show()

sns.pairplot(df, hue="target", palette='Dark2')
plt.suptitle('Pairplot - Relação entre variáveis', y=1.02)
plt.show()

print("\n--- Parte 5: Modelo Simples ---")

X_model = df.drop('target', axis=1)
y_model = df['target']

X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.30, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {acuracia * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))