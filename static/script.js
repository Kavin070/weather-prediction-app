// Weather condition emojis and colors
const weatherEmojis = {
    'Sunny': '☀️',
    'Rainy': '🌧️',
    'Cloudy': '☁️',
    'Clear': '🌤️',
    'Snowy': '❄️'
};

const weatherColors = {
    'Sunny': 'weather-sunny',
    'Rainy': 'weather-rainy',
    'Cloudy': 'weather-cloudy',
    'Clear': 'weather-clear',
    'Snowy': 'weather-snowy'
};

// DOM elements
const form = document.getElementById('weatherForm');
const resultsSection = document.getElementById('resultsSection');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const fetchWeatherBtn = document.getElementById('fetchWeatherBtn');
const cityInput = document.getElementById('cityInput');
const liveWeatherContent = document.getElementById('liveWeatherContent');
const useRealDataBtn = document.getElementById('useRealDataBtn');

// Global variable to store current weather data
let currentWeatherData = null;

// Live weather functionality
fetchWeatherBtn.addEventListener('click', fetchLiveWeather);
cityInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        fetchLiveWeather();
    }
});

async function fetchLiveWeather() {
    const city = cityInput.value.trim();
    if (!city) {
        showWeatherError('Please enter a city name');
        return;
    }

    // Show loading state
    liveWeatherContent.innerHTML = `
        <div class="weather-loading">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Loading weather data for ${city}...</p>
        </div>
    `;
    useRealDataBtn.style.display = 'none';

    try {
        const response = await fetch(`/get-live-weather?city=${encodeURIComponent(city)}`);
        const data = await response.json();

        if (data.success) {
            currentWeatherData = data;
            displayLiveWeather(data);
            useRealDataBtn.style.display = 'block';
        } else {
            showWeatherError(data.error || 'Failed to fetch weather data');
        }
    } catch (error) {
        console.error('Weather fetch error:', error);
        showWeatherError('Network error: Unable to fetch weather data. Please check your connection and try again.');
    }
}

function displayLiveWeather(data) {
    const weatherIcon = getWeatherIcon(data.icon);
    const currentDate = getCurrentDateString();
    
    liveWeatherContent.innerHTML = `
        <div class="weather-display">
            <h3>${weatherIcon} ${data.current_weather || data.temperature}°C</h3>
            <div class="location">${data.city}, ${data.country}</div>
            <div class="date">${currentDate}</div>
            <div class="current-weather">${data.temperature}°C</div>
            <div class="description">${data.description}</div>
        </div>
        <div class="weather-params">
            <div class="param-item">
                <div class="param-label">Temperature</div>
                <div class="param-value">${data.temperature}°C</div>
            </div>
            <div class="param-item">
                <div class="param-label">Humidity</div>
                <div class="param-value">${data.humidity}%</div>
            </div>
            <div class="param-item">
                <div class="param-label">Pressure</div>
                <div class="param-value">${data.pressure} hPa</div>
            </div>
            <div class="param-item">
                <div class="param-label">Wind Speed</div>
                <div class="param-value">${data.wind_speed} km/h</div>
            </div>
            <div class="param-item">
                <div class="param-label">Cloud Cover</div>
                <div class="param-value">${data.cloud_cover}%</div>
            </div>
            <div class="param-item">
                <div class="param-label">Feels Like</div>
                <div class="param-value">${data.feels_like}°C</div>
            </div>
        </div>
    `;
}

function showWeatherError(message) {
    liveWeatherContent.innerHTML = `
        <div class="weather-error">
            <i class="fas fa-exclamation-triangle"></i>
            <p>${message}</p>
        </div>
    `;
    useRealDataBtn.style.display = 'none';
}

function getCurrentDateString() {
    const now = new Date();
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        timeZone: 'Europe/Amsterdam' // Netherlands timezone
    };
    return now.toLocaleDateString('en-US', options);
}

function getWeatherIcon(iconCode) {
    const iconMap = {
        '01d': '☀️', '01n': '🌙',
        '02d': '🌤️', '02n': '🌙',
        '03d': '☁️', '03n': '☁️',
        '04d': '☁️', '04n': '☁️',
        '09d': '🌧️', '09n': '🌧️',
        '10d': '🌦️', '10n': '🌦️',
        '11d': '⛈️', '11n': '⛈️',
        '13d': '❄️', '13n': '❄️',
        '50d': '🌫️', '50n': '🌫️'
    };
    return iconMap[iconCode] || '🌤️';
}

// Use real data button functionality
useRealDataBtn.addEventListener('click', () => {
    if (currentWeatherData) {
        document.getElementById('temperature').value = currentWeatherData.temperature;
        document.getElementById('humidity').value = currentWeatherData.humidity;
        document.getElementById('pressure').value = currentWeatherData.pressure;
        document.getElementById('wind_speed').value = currentWeatherData.wind_speed;
        document.getElementById('cloud_cover').value = currentWeatherData.cloud_cover;
        
        // Scroll to the input section
        document.querySelector('.input-section').scrollIntoView({ behavior: 'smooth' });
        
        // Add visual feedback
        const inputs = document.querySelectorAll('.input-section input');
        inputs.forEach(input => {
            input.style.borderColor = '#28a745';
            setTimeout(() => {
                input.style.borderColor = '#e1e5e9';
            }, 2000);
        });
    }
});

// Form submission handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading
    loading.style.display = 'flex';
    resultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    
    // Get form data
    const formData = {
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        pressure: parseFloat(document.getElementById('pressure').value),
        wind_speed: parseFloat(document.getElementById('wind_speed').value),
        cloud_cover: parseFloat(document.getElementById('cloud_cover').value)
    };
    
    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result.prediction, result.probabilities);
        } else {
            showError(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        loading.style.display = 'none';
    }
});

// Display prediction results
function displayResults(prediction, probabilities) {
    // Update main prediction
    document.getElementById('weatherEmoji').textContent = weatherEmojis[prediction] || '🌤️';
    document.getElementById('predictionText').textContent = prediction;
    
    // Update prediction card color
    const mainPrediction = document.querySelector('.main-prediction');
    mainPrediction.className = `main-prediction ${weatherColors[prediction] || 'weather-clear'}`;
    
    // Create probability bars
    const probabilityBars = document.getElementById('probabilityBars');
    probabilityBars.innerHTML = '';
    
    // Sort probabilities by value (descending)
    const sortedProbs = Object.entries(probabilities)
        .sort(([,a], [,b]) => b - a);
    
    sortedProbs.forEach(([condition, probability]) => {
        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar';
        
        barContainer.innerHTML = `
            <div class="probability-label">
                <span>${weatherEmojis[condition] || '🌤️'} ${condition}</span>
                <span>${probability}%</span>
            </div>
            <div class="probability-fill" style="width: ${probability}%"></div>
        `;
        
        probabilityBars.appendChild(barContainer);
    });
    
    // Show results with animation
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth' });
}

// Add input validation and UX improvements
document.querySelectorAll('input[type="number"]').forEach(input => {
    input.addEventListener('input', function() {
        // Remove invalid styling
        this.style.borderColor = '#e1e5e9';
        
        // Add some basic validation feedback
        const value = parseFloat(this.value);
        const id = this.id;
        
        let isValid = true;
        
        if (id === 'humidity' || id === 'cloud_cover') {
            if (value < 0 || value > 100) isValid = false;
        } else if (id === 'wind_speed') {
            if (value < 0) isValid = false;
        } else if (id === 'temperature') {
            if (value < -50 || value > 60) isValid = false;
        } else if (id === 'pressure') {
            if (value < 900 || value > 1100) isValid = false;
        }
        
        if (!isValid && this.value !== '') {
            this.style.borderColor = '#ff6b6b';
        }
    });
});

// Check if model is ready on page load and fetch initial weather
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/model-info');
        const info = await response.json();
        
        if (!info.trained) {
            showError('Model is not trained yet. Please wait a moment and refresh the page.');
        } else {
            // Fetch weather for default city (Chennai based on user location)
            fetchLiveWeather();
        }
    } catch (error) {
        showError('Unable to connect to the prediction server.');
        console.error('Model info error:', error);
    }
});