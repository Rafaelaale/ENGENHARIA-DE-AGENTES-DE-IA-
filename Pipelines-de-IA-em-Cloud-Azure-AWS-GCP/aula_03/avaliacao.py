from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def avaliar_modelo(modelo, X_teste, y_teste):
    previsoes = modelo.predict(X_teste)
    return {
        'accuracy': float(accuracy_score(y_teste, previsoes)),
        'precision': float(precision_score(y_teste, previsoes, zero_division=0)),
        'recall': float(recall_score(y_teste, previsoes, zero_division=0)),
        'f1': float(f1_score(y_teste, previsoes, zero_division=0)),
        'matriz_confusao': confusion_matrix(y_teste, previsoes).tolist()
    }
