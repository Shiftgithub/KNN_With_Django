import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import io
import base64

from django.shortcuts import render

def knn_predict(request):
    if request.method == 'POST':
        # Retrieve user input from the form
        age = int(request.POST.get('age'))
        gender = request.POST.get('gender')
        bp = request.POST.get('bp')
        chol = request.POST.get('chol')
        na_to_k = float(request.POST.get('na_to_k'))

        # Create a DataFrame from the dataset
        df = pd.read_csv('app/dataset/drug200.csv')

        # Convert categorical variables to numerical values using LabelEncoder
        label_encoders = {}
        for col in ['Gender', 'BP', 'Cholesterol', 'Drug']:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

        # Split the dataset into features (X_train) and target (y_train)
        X_train = df.drop(columns=["Drug"])
        y_train = df['Drug']

        # Create a KNN classifier with k=3 (you can adjust k as needed)
        knn = KNeighborsClassifier(n_neighbors=3)

        # Fit the classifier with the training data
        knn.fit(X_train, y_train)

        # Convert categorical variables in custom input to numerical values
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        bp_encoded = label_encoders['BP'].transform([bp])[0]
        chol_encoded = label_encoders['Cholesterol'].transform([chol])[0]

        # Custom input data for prediction
        custom_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'BP': [bp_encoded],
            'Cholesterol': [chol_encoded],
            'Na_to_K': [na_to_k]
        })

        # Make a prediction using the KNN classifier
        predicted_class = knn.predict(custom_input)

        # Inverse transform the numerical prediction back to the original class label
        predicted_class_name = label_encoders['Drug'].inverse_transform(predicted_class)[0]

        # Prepare the PairPlot
        df_pairplot = pd.concat([df, custom_input], ignore_index=True)

        # Create a PairPlot to visualize the relationships between features
        sns_plot = sns.pairplot(df_pairplot, hue='Drug', markers=["o", "s", "D", "X", "P"])

        # Save the PairPlot to a buffer
        buffer = io.BytesIO()
        sns_plot.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()

        # Pass the plot data and other information to the template for rendering
        context = {
            'custom_input': custom_input.to_dict(orient='records')[0],
            'predicted_class': predicted_class_name,
            'plot_data': plot_data,
        }

        return render(request, 'knn_predict.html', context)

    return render(request, 'knn_predict.html')
