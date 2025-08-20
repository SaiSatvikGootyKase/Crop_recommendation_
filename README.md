# 🌱 Crop Recommendation Web Application

A modern, AI-powered web application that provides intelligent crop recommendations based on soil and environmental parameters. Built with Flask, Machine Learning, and modern web technologies.

## ✨ Features

- **AI-Powered Recommendations**: Uses Random Forest algorithm for accurate crop predictions
- **Modern Web Interface**: Beautiful, responsive design that works on all devices
- **Real-time Analysis**: Instant crop recommendations with confidence scores
- **Comprehensive Input**: Covers all essential soil and climate parameters
- **User-Friendly**: Intuitive form design with helpful input validation
- **Professional UI**: Modern design with smooth animations and transitions

## 🚀 Live Demo

The application provides a web interface where users can:
1. Input soil parameters (N, P, K levels)
2. Enter environmental conditions (temperature, humidity, pH, rainfall)
3. Receive instant AI-powered crop recommendations
4. View confidence scores for each recommendation

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for API endpoints
- **Scikit-learn**: Machine learning library
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Pickle**: Model serialization

### Frontend
- **HTML5 & CSS3**: Modern web standards
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icon library
- **Vanilla JavaScript**: Interactive functionality

### Machine Learning
- **Random Forest Classifier**: Ensemble learning algorithm
- **Standard Scaler**: Feature normalization
- **Model Persistence**: Saved models for quick loading

## 📋 Prerequisites

Before running this application, make sure you have:

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SaiSatvikGootyKase/Crop_recommendation.git
cd Crop_recommendation
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

The application will be available at: `http://localhost:5000`

## 📊 Input Parameters

The system accepts the following soil and environmental parameters:

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| **Nitrogen (N)** | 0-140 | kg/ha | Essential for leaf growth |
| **Phosphorus (P)** | 5-145 | kg/ha | Important for root development |
| **Potassium (K)** | 5-205 | kg/ha | Disease resistance & water regulation |
| **Temperature** | 8.8-43.7 | °C | Affects plant growth rates |
| **Humidity** | 14.0-99.9 | % | Water availability & disease pressure |
| **pH Level** | 3.5-10.0 | - | Nutrient availability in soil |
| **Rainfall** | 20.0-298.0 | mm | Water supply for crops |

## 🔧 How It Works

### 1. Data Input
Users input their soil and climate parameters through an intuitive web form.

### 2. Model Processing
The Random Forest algorithm analyzes the input data against trained patterns.

### 3. Recommendation Generation
The system provides crop recommendations with confidence scores.

### 4. Result Display
Beautiful, informative results are displayed to the user.

## 📁 Project Structure

```
crop/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── index.html       # Main page with crop recommendation form
│   └── about.html       # About page with system information
├── model.pkl            # Trained ML model (generated on first run)
├── scaler.pkl           # Feature scaler (generated on first run)
└── Crop_dataset.csv     # Training dataset (if available)
```

## 🎯 Machine Learning Model

### Algorithm: Random Forest Classifier
- **Type**: Ensemble learning method
- **Advantages**: High accuracy, handles outliers, feature importance
- **Training**: Uses historical crop data with soil parameters
- **Output**: Crop recommendations with confidence scores

### Model Features
- Automatic training on first run
- Fallback to sample data if dataset unavailable
- Real-time predictions
- Confidence scoring for recommendations

## 🌐 API Endpoints

### GET `/`
- **Description**: Main application page
- **Response**: HTML page with crop recommendation form

### POST `/predict`
- **Description**: Get crop recommendation
- **Request Body**: JSON with soil parameters
- **Response**: JSON with crop prediction and confidence

### GET `/about`
- **Description**: About page with system information
- **Response**: HTML page with detailed information

## 🎨 Customization

### Styling
- Modify CSS variables in `templates/index.html`
- Update color scheme in `:root` section
- Customize animations and transitions

### Features
- Add new input parameters in `app.py`
- Modify ML model in `train_model()` function
- Extend API endpoints as needed

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Waitress (Windows)
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Custom port number (default: 5000)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Sai Satvik Gooty Kase**
- GitHub: [@SaiSatvikGootyKase](https://github.com/SaiSatvikGootyKase)
- Project: [Crop Recommendation System](https://github.com/SaiSatvikGootyKase/Crop_recommendation.git)

## 🙏 Acknowledgments

- Scikit-learn team for the machine learning library
- Bootstrap team for the responsive UI framework
- Font Awesome for the icon library
- Flask community for the web framework

## 📞 Support

If you have any questions or need help:
1. Check the [Issues](https://github.com/SaiSatvikGootyKase/Crop_recommendation/issues) page
2. Create a new issue for bugs or feature requests
3. Star the repository if you find it helpful!

---

**Happy Farming with AI! 🌾🤖**
