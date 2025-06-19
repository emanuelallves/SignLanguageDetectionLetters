import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

    filtered_data = []
    filtered_labels = []

    for x, y in zip(data_dict['data'], data_dict['labels']):
        if isinstance(x, (list, np.ndarray)) and len(x) == 42:
            filtered_data.append(x)
            filtered_labels.append(y)

    data = np.asarray(filtered_data)
    labels = np.asarray(filtered_labels)

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=labels)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_pred, y_test)

print(f"Accuracy: {score:.2f}")

print("Saving model...")

if not os.path.exists('./model'):
    os.makedirs('model')

with open('./model/model.pickle', 'wb') as f:
    pickle.dump({'model': model}, f)