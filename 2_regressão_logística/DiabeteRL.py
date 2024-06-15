import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Carregar os dados
data = pd.read_csv('diabeteDados.csv', delimiter=';', header=0)

# Selecionar apenas as features contínuas (excluindo 'Pregnancies' e 'Outcome')
continuous_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[continuous_features]
y = data['Outcome']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA para reduzir a dimensionalidade para 1 feature
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Implementar a regressão logística
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Fazer previsões
y_pred = lr.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calcular Parâmetros de Precisão, Recall e F1-Score
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Plotar os dados de treinamento
plt.figure(figsize=(8, 6))
plt.scatter(X_train[y_train == 0][:, 0], [0] * len(X_train[y_train == 0]), color='b', label='Não Diabetes')
plt.scatter(X_train[y_train == 1][:, 0], [0.1] * len(X_train[y_train == 1]), color='r', label='Diabetes')  # Adicione um pequeno deslocamento para visualização
plt.title('Dados de Treinamento após Redução de Dimensionalidade (PCA)')
plt.xlabel('Principal Componente 1')
plt.yticks([], [])
plt.legend()
plt.show()




