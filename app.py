from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    
    try:
        # Check if model files exist, if not train a new one
        if not (os.path.exists('model.pkl') and os.path.exists('scaler.pkl')):
            print("Model files not found, training new model...")
            train_model()
        
        # Load the model and scaler
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        print("✅ Model and scaler loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Scaler type: {type(scaler)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Training new model...")
        train_model()
        load_model()  # Try loading again

def train_model():
    """Train the crop recommendation model"""
    global model, scaler
    
    try:
        # Load the dataset
        try:
            df = pd.read_csv('Crop_dataset.csv')
            print("Using existing Crop_dataset.csv")
        except FileNotFoundError:
            # If dataset not found, create a sample dataset for demonstration
            print("Dataset not found, creating sample data...")
            df = create_sample_dataset()
        
        # Prepare features and target
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = df[features]
        y = df['label']
        
        print(f"Training data shape: {X.shape}")
        print(f"Features: {features}")
        print(f"Target classes: {y.unique()}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train the model with better hyperparameters
        model = RandomForestClassifier(
            n_estimators=200,  # More trees for better diversity
            max_depth=15,       # Control tree depth
            min_samples_split=5, # Minimum samples to split
            min_samples_leaf=2,  # Minimum samples in leaf
            max_features='sqrt', # Use sqrt of features for each split
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Evaluate model performance
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(scaler.transform(X_test), y_test)
        
        print("✅ Model trained and saved successfully!")
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature importance:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        raise e

def create_sample_dataset():
    """Create a sample dataset for demonstration purposes"""
    np.random.seed(42)
    n_samples = 2000  # Increased sample size for better diversity
    
    # Generate more diverse sample data with realistic ranges
    data = {
        'N': np.random.randint(0, 140, n_samples),
        'P': np.random.randint(5, 145, n_samples),
        'K': np.random.randint(5, 205, n_samples),
        'temperature': np.random.uniform(8.8, 43.7, n_samples),
        'humidity': np.random.uniform(14.0, 99.9, n_samples),
        'ph': np.random.uniform(3.5, 10.0, n_samples),
        'rainfall': np.random.uniform(20.0, 298.0, n_samples)
    }
    
    # More sophisticated rule-based labels for better diversity
    labels = []
    for i in range(n_samples):
        N, P, K = data['N'][i], data['P'][i], data['K'][i]
        temp, hum, ph, rain = data['temperature'][i], data['humidity'][i], data['ph'][i], data['rainfall'][i]
        
        # Rice: High temperature, high humidity, moderate rainfall
        if temp > 25 and hum > 70 and rain > 100 and N > 50:
            labels.append('rice')
        
        # Wheat: Cool temperature, moderate humidity, moderate pH
        elif temp < 25 and hum < 80 and 5.5 < ph < 7.5 and P > 40:
            labels.append('wheat')
        
        # Cotton: High temperature, moderate humidity, high N and P
        elif temp > 30 and 40 < hum < 80 and N > 80 and P > 60:
            labels.append('cotton')
        
        # Maize: Moderate temperature, moderate humidity, balanced nutrients
        elif 20 < temp < 30 and 50 < hum < 70 and 40 < N < 80 and 30 < P < 70:
            labels.append('maize')
        
        # Sugarcane: High temperature, high humidity, high rainfall
        elif temp > 28 and hum > 75 and rain > 150 and K > 100:
            labels.append('sugarcane')
        
        # Coffee: Moderate temperature, high humidity, acidic pH
        elif 18 < temp < 25 and hum > 80 and ph < 6.5 and rain > 120:
            labels.append('coffee')
        
        # Tea: Cool temperature, high humidity, acidic pH
        elif temp < 22 and hum > 85 and ph < 6.0 and rain > 100:
            labels.append('tea')
        
        # Potato: Cool temperature, moderate humidity, neutral pH
        elif temp < 20 and 60 < hum < 80 and 5.5 < ph < 7.0 and P > 50:
            labels.append('potato')
        
        # Tomato: Moderate temperature, moderate humidity, balanced nutrients
        elif 20 < temp < 28 and 60 < hum < 75 and 6.0 < ph < 7.0 and N > 60:
            labels.append('tomato')
        
        # Pepper: High temperature, moderate humidity, high N
        elif temp > 25 and 50 < hum < 70 and N > 70 and P > 50:
            labels.append('pepper')
        
        # Default to maize for remaining cases
        else:
            labels.append('maize')
    
    data['label'] = labels
    
    # Print dataset statistics
    print(f"Sample dataset created with {n_samples} samples")
    print(f"Crop distribution: {pd.Series(labels).value_counts().to_dict()}")
    
    return pd.DataFrame(data)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/test_model')
def test_model():
    """Test endpoint to verify model diversity"""
    global model, scaler
    
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Test different input combinations
    test_inputs = [
        {'N': 20, 'P': 20, 'K': 20, 'temperature': 20, 'humidity': 55, 'ph': 6, 'rainfall': 66},
        {'N': 100, 'P': 100, 'K': 100, 'temperature': 35, 'humidity': 80, 'ph': 7, 'rainfall': 200},
        {'N': 50, 'P': 50, 'K': 50, 'temperature': 15, 'humidity': 70, 'ph': 5.5, 'rainfall': 80},
        {'N': 80, 'P': 60, 'K': 120, 'temperature': 28, 'humidity': 75, 'ph': 6.5, 'rainfall': 150},
        {'N': 30, 'P': 40, 'K': 30, 'temperature': 22, 'humidity': 85, 'ph': 5.0, 'rainfall': 120}
    ]
    
    results = []
    for i, test_input in enumerate(test_inputs):
        try:
            features = np.array([[
                float(test_input['N']),
                float(test_input['P']),
                float(test_input['K']),
                float(test_input['temperature']),
                float(test_input['humidity']),
                float(test_input['ph']),
                float(test_input['rainfall'])
            ]])
            
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            results.append({
                'test_case': i + 1,
                'input': test_input,
                'prediction': prediction,
                'probabilities': dict(zip(model.classes_, [round(p * 100, 2) for p in probabilities]))
            })
            
        except Exception as e:
            results.append({
                'test_case': i + 1,
                'input': test_input,
                'error': str(e)
            })
    
    return jsonify({
        'model_info': {
            'classes': list(model.classes_),
            'n_estimators': model.n_estimators,
            'feature_names': ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        },
        'test_results': results
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop recommendation"""
    global model, scaler
    
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            print("Model or scaler not loaded, attempting to load...")
            load_model()
            if model is None or scaler is None:
                return jsonify({
                    'success': False,
                    'error': 'Model not properly loaded. Please restart the application.'
                }), 500
        
        # Get input data
        data = request.get_json()
        
        # Validate input data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract features
        features = np.array([[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]])
        
        print(f"Input features: {features}")
        print(f"Scaler type: {type(scaler)}")
        
        # Scale features
        features_scaled = scaler.transform(features)
        print(f"Scaled features: {features_scaled}")
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        max_probability = np.max(probabilities) * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'crop': model.classes_[idx],
                'confidence': round(probabilities[idx] * 100, 2)
            })
        
        print(f"Prediction: {prediction}, Confidence: {max_probability:.2f}%")
        print(f"Top 3 predictions: {top_predictions}")
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': round(max_probability, 2),
            'top_predictions': top_predictions,
            'message': f'Recommended crop: {prediction} (Confidence: {round(max_probability, 2)}%)',
            'all_probabilities': dict(zip(model.classes_, [round(p * 100, 2) for p in probabilities]))
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    print("🚀 Starting Crop Recommendation Web Application...")
    print("📊 Loading machine learning model...")
    
    # Load the model when starting the app
    load_model()
    
    if model is not None and scaler is not None:
        print("✅ Application ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Application cannot start.")
        exit(1)
