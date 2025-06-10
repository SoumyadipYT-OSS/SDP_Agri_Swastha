// Main JavaScript file

// Firebase Config
const firebaseConfig = {
    apiKey: "AIzaSyAk35bQx3R1keYgWCV3dPrbV4ejOYlVjKU",
    authDomain: "agri-swasthya.firebaseapp.com",
    databaseURL: "https://agri-swasthya-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "agri-swasthya",
    storageBucket: "agri-swasthya.firebasestorage.app",
    messagingSenderId: "987282539692",
    appId: "1:987282539692:web:addc431ee50022c7e00b09"
};

// Initialize Firebase and get auth instance
let auth;
let database;

try {
    firebase.initializeApp(firebaseConfig);
    auth = firebase.auth();
    database = firebase.database();
    console.log('Firebase initialized successfully');
} catch (error) {
    console.error('Error initializing Firebase:', error);
}

// Login Function
function login() {
    console.log('Login function called');
    const email = document.getElementById("email").value;
    const password = document.getElementById("password").value;
    
    // Input validation
    if (!email || !password) {
        alert("Please enter both email and password");
        return;
    }

    if (!auth) {
        alert("Authentication system is not initialized. Please try again in a few moments.");
        return;
    }

    // Show loading state
    const loginBtn = document.querySelector('.login-btn');
    loginBtn.textContent = 'Logging in...';
    loginBtn.disabled = true;

    auth.signInWithEmailAndPassword(email, password)
        .then((userCredential) => {
            console.log("Login successful");
            window.location.href = "dashboard.html";
        })
        .catch((error) => {
            console.error("Login error:", error);
            alert("Login failed: " + error.message);
            loginBtn.textContent = 'Login';
            loginBtn.disabled = false;
        });
}

// Logout Function
function logout() {
    auth.signOut()
        .then(() => {
            console.log("Logout successful");
            window.location.href = "index.html";
        })
        .catch((error) => {
            console.error("Logout error:", error);
            alert("Logout failed: " + error.message);
        });
}

// Authentication state observer
auth?.onAuthStateChanged((user) => {
    if (user) {
        // User is signed in
        console.log('User is signed in:', user.email);
        // If we're on the login page, redirect to dashboard
        if (window.location.pathname.endsWith('index.html') || window.location.pathname === '/') {
            window.location.href = 'dashboard.html';
        }
    } else {
        // User is signed out
        console.log('No user is signed in');
        // If we're on the dashboard, redirect to login
        if (window.location.pathname.endsWith('dashboard.html')) {
            window.location.href = 'index.html';
        }
    }
});

// Check authentication state
auth.onAuthStateChanged((user) => {
    if (window.location.pathname.includes("dashboard.html")) {
        if (!user) {
            // If not logged in, redirect to login page
            window.location.href = "index.html";
        } else {
            // Start listening to sensor data
            setupRealtimeListener();
        }
    } else if (window.location.pathname.includes("index.html")) {
        if (user) {
            // If already logged in, redirect to dashboard
            window.location.href = "dashboard.html";
        }
    }
});

// Function to submit new reading
function submitReading() {
    // Get all input values and convert to numbers
    const reading = {
        N: Number(document.getElementById('input-n').value),
        P: Number(document.getElementById('input-p').value),
        K: Number(document.getElementById('input-k').value),
        EC: Number(document.getElementById('input-ec').value),
        Fe: Number(document.getElementById('input-fe').value), // Make sure Fe is capitalized
        timestamp: Date.now()
    };
    
    console.log('Submitting reading:', reading); // Debug log

    // Validate inputs
    const validation = validateReading(reading);
    if (!validation.isValid) {
        alert(validation.message);
        return;
    }

    // Show loading state
    const submitBtn = document.querySelector('.submit-btn');
    submitBtn.disabled = true;
    submitBtn.classList.add('loading');
    submitBtn.textContent = 'Submitting...';

    // Save to Firebase
    Promise.all([
        database.ref('sensor_data').set(reading),
        database.ref('sensor_readings').push(reading)
    ])
    .then(() => {
        clearForm();
        alert('Reading submitted successfully!');
    })
    .catch(error => {
        console.error('Error saving reading:', error);
        alert('Failed to save reading: ' + error.message);
    })
    .finally(() => {
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
        submitBtn.textContent = 'Submit Reading';
    });
}

// Validate reading values
function validateReading(reading) {
    // Define valid ranges for each parameter
    const ranges = {
        N: { min: 0, max: 1000 },   // Nitrogen (mg/kg)
        P: { min: 0, max: 500 },    // Phosphorus (mg/kg)
        K: { min: 0, max: 1000 },   // Potassium (mg/kg)
        EC: { min: 0, max: 10 },    // Electrical Conductivity (dS/m)
        Fe: { min: 0, max: 1000 }   // Iron (mg/kg)
    };

    // Check if any value is NaN
    for (const [key, value] of Object.entries(reading)) {
        if (key === 'timestamp') continue;
        if (isNaN(value)) {
            return {
                isValid: false,
                message: `Please enter a valid number for ${key}`
            };
        }
    }

    // Check if values are within valid ranges
    for (const [key, value] of Object.entries(reading)) {
        if (key === 'timestamp') continue;
        const range = ranges[key];
        if (value < range.min || value > range.max) {
            return {
                isValid: false,
                message: `${key} value must be between ${range.min} and ${range.max}`
            };
        }
    }

    return { isValid: true };
}

// Setup realtime listener for sensor data
function setupRealtimeListener() {
    const sensorRef = database.ref('sensor_data');
    
    sensorRef.on('value', (snapshot) => {
        const data = snapshot.val();
        if (data) {
            updateDashboard(data);
        }
    }, (error) => {
        console.error("Error reading sensor data:", error);
        alert("Failed to load sensor data: " + error.message);
    });
}

// Update dashboard with sensor data
function updateDashboard(data) {
    console.log('Received data:', data); // Debug log
    
    const elements = ['n', 'p', 'k', 'ec', 'fe'];
    const units = {
        n: 'mg/kg',
        p: 'mg/kg',
        k: 'mg/kg',
        ec: 'dS/m',
        fe: 'mg/kg'
    };

    elements.forEach(element => {
        // Try all possible case variations
        const value = data[element.toUpperCase()] || 
                     data[element.toLowerCase()] || 
                     (element === 'fe' ? data['Fe'] : null);
                     
        console.log(`${element} value:`, value); // Debug log
        
        if (value !== undefined && value !== null) {
            const el = document.getElementById(element);
            if (el) {
                el.textContent = `${value} ${units[element]}`;
            }
        }
    });
}

// Function to simulate sending sensor data (for testing)
function sendSensorData(data = null) {
    if (!data) {
        // Sample data if none provided
        data = {
            N: Math.floor(Math.random() * 200 + 100),
            P: Math.floor(Math.random() * 100 + 50),
            K: Math.floor(Math.random() * 300 + 150),
            EC: (Math.random() * 2 + 0.5).toFixed(2),
            Fe: Math.floor(Math.random() * 50 + 20)
        };
    }

    database.ref('sensor_data').set(data)
        .then(() => {
            console.log("Data sent successfully:", data);
        })
        .catch((error) => {
            console.error("Error sending data:", error);
        });
}

// Function to clear the form
function clearForm() {
    ['n', 'p', 'k', 'ec', 'fe'].forEach(id => {
        document.getElementById(`input-${id}`).value = '';
    });
}

// Add ONNX Runtime to the head of the document and ensure it's loaded
let selectedModel = 'new_knn_classifier';
let session = null;

// Function to load ONNX Runtime
function loadONNXRuntime() {
    return new Promise((resolve, reject) => {
        if (window.ort) {
            console.log('ONNX Runtime already loaded');
            resolve();
            return;
        }

        const onnxScript = document.createElement('script');
        onnxScript.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js';
        onnxScript.onload = () => {
            console.log('ONNX Runtime loaded successfully');
            resolve();
        };
        onnxScript.onerror = () => {
            reject(new Error('Failed to load ONNX Runtime'));
        };
        document.head.appendChild(onnxScript);
    });
}

// Function to update selected model
function updateSelectedModel() {
    selectedModel = document.getElementById('model-select').value;
    // Clear previous results
    document.getElementById('result-text').textContent = 'Select a model and click Analyze to get results';
    document.getElementById('result-details').innerHTML = '';
}

// Function to load ONNX model
async function loadModel(modelName) {
    try {
        console.log('Loading model:', modelName);
        const modelPath = `../Models/trained_models/${modelName}.onnx`;
        console.log('Model path:', modelPath);
        
        const response = await fetch(modelPath);
        if (!response.ok) {
            throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
        }
        
        const modelArrayBuffer = await response.arrayBuffer();
        console.log('Model buffer loaded, size:', modelArrayBuffer.byteLength);
        
        console.log('Creating inference session...');
        session = await ort.InferenceSession.create(modelArrayBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        // Log model information
        console.log('Model loaded successfully:', modelName);
        console.log('Model input names:', session.inputNames);
        console.log('Model output names:', session.outputNames);
        
        session.modelPath = modelName; // Store the model name for reference
        return true;
    } catch (error) {
        console.error('Error loading model:', error);
        return false;
    }
}

// Function to check if models are available
async function checkModelAvailability() {
    const modelSelect = document.getElementById('model-select');
    if (!modelSelect) return;

    for (const option of modelSelect.options) {
        try {
            const response = await fetch(`../Models/trained_models/${option.value}.onnx`, { method: 'HEAD' });
            if (!response.ok) {
                option.textContent += ' (Not Available)';
                option.disabled = true;
            }
        } catch (error) {
            console.error(`Error checking model ${option.value}:`, error);
            option.textContent += ' (Error)';
            option.disabled = true;
        }
    }
}

// Call this when the page loads
if (document.getElementById('model-select')) {
    checkModelAvailability();
}

// Function to analyze soil using the selected model
async function analyzeSoil() {
    const resultText = document.getElementById('result-text');
    const resultDetails = document.getElementById('result-details');
    const analyzeBtn = document.querySelector('.analyze-btn');
    
    // Make sure ONNX Runtime is loaded
    try {
        await loadONNXRuntime();
    } catch (error) {
        console.error('Failed to load ONNX Runtime:', error);
        alert('Failed to initialize the analysis system. Please refresh the page and try again.');
        return;
    }
    
    // Disable button and show loading state
    analyzeBtn.disabled = true;
    analyzeBtn.classList.add('loading');
    analyzeBtn.textContent = 'Analyzing...';
    resultDetails.innerHTML = '';
    resultText.textContent = 'Processing...';
    
    // Get current sensor values
    const input = {
        N: parseFloat(document.getElementById('n').textContent),
        P: parseFloat(document.getElementById('p').textContent),
        K: parseFloat(document.getElementById('k').textContent),
        EC: parseFloat(document.getElementById('ec').textContent),
        Fe: parseFloat(document.getElementById('fe').textContent)
    };

    // Validate input values
    for (const [key, value] of Object.entries(input)) {
        if (isNaN(value)) {
            alert(`Please ensure valid ${key} reading is available`);
            return;
        }
    }

    try {
        // Step 1: Load Model (if needed)
        // Models are loaded lazily to save bandwidth. We only load a new model
        // when the user switches to a different one
        if (!session || session.modelPath !== selectedModel) {
            resultText.textContent = 'Loading model...';
            const loaded = await loadModel(selectedModel);
            if (!loaded) {
                throw new Error('Failed to load model');
            }
        }

        // Step 2: Prepare Input Tensor
        // The model expects a 2D tensor with shape [1, 5] where:
        // - 1 represents a single sample
        // - 5 represents the features: [N, P, K, EC, Fe]
        // We use float32 as it's the standard for ML models
        const inputTensor = new ort.Tensor(
            'float32',
            Float32Array.from([input.N, input.P, input.K, input.EC, input.Fe]),
            [1, 5]
        );

        // Step 3: Run Model Inference
        resultText.textContent = 'Analyzing...';
        // Log input values for debugging and validation
        console.log('Running inference with input:', [input.N, input.P, input.K, input.EC, input.Fe]);
        
        // Prepare the input feeds object
        // 'float_input' is the name of the input tensor defined during model conversion
        // This must match the model's expected input name
        const feeds = { 'float_input': inputTensor };
        console.log('Running model with feeds:', feeds);
        
        const outputMap = await session.run(feeds);
        console.log('Model output:', outputMap);
        
        // Get the first output tensor (models typically output as 'output' or numerically)
        const outputTensor = outputMap[Object.keys(outputMap)[0]];
        // Log the raw output tensor for debugging
        console.log('Output tensor:', outputTensor);
        
        // Convert tensor data to a JavaScript array for processing
        const output = Array.from(outputTensor.data);
        console.log('Output data:', output);
        
        // Process results based on model type
        let result = '';
        
        // ONNX models may output BigInt values, convert them to regular numbers
        // This is necessary because some ONNX operations output 64-bit integers
        const outputNumbers = output.map(val => typeof val === 'bigint' ? Number(val) : val);
        
        // Handle classification model outputs
        if (selectedModel.includes('classifier')) {
            // Find the index of the highest probability (argmax)
            // This gives us the predicted class index (0=Low, 1=Medium, 2=High)
            const prediction = outputNumbers.indexOf(Math.max(...outputNumbers));
            
            // Define class labels for human-readable output
            const classes = ['Low Fertility', 'Medium Fertility', 'High Fertility'];
            
            // Convert raw probabilities to percentages with 2 decimal places
            // The model outputs probabilities between 0 and 1, multiply by 100 for percentage
            const probabilities = outputNumbers.map(p => (p * 100).toFixed(2) + '%');
            
            // Format the final result string
            result = `Soil Fertility Class: ${classes[prediction]}`;
            
            // Log predictions for debugging and monitoring
            console.log('Predicted class:', classes[prediction]);
            console.log('Class probabilities:', probabilities);
        }

        // Display Results Section
        // Step 1: Update the main result text with the prediction
        resultText.textContent = result;

        // Get the highest probability index and value
        const maxIndex = outputNumbers.indexOf(Math.max(...outputNumbers));
        const maxProb = (outputNumbers[maxIndex] * 100).toFixed(1);

        // Step 2: Create detailed results view
        resultDetails.innerHTML = `
            <div class="analysis-results">
                <!-- Main Prediction Card -->
                <div class="result-card main-prediction">
                    <h3>🎯 Prediction Result</h3>
                    <div class="prediction-value">
                        ${['Low', 'Medium', 'High'][maxIndex]} Fertility
                        <span class="confidence">${maxProb}% confidence</span>
                    </div>
                </div>

                <!-- Input Parameters Card -->
                <div class="result-card parameters">
                    <h3>📊 Soil Parameters</h3>
                    <div class="parameter-grid">
                        ${[
                            {name: 'Nitrogen (N)', value: input.N, unit: 'mg/kg', optimal: '350-380'},
                            {name: 'Phosphorus (P)', value: input.P, unit: 'mg/kg', optimal: '11-13'},
                            {name: 'Potassium (K)', value: input.K, unit: 'mg/kg', optimal: '800-850'},
                            {name: 'EC', value: input.EC, unit: 'dS/m', optimal: '0.85-0.95'},
                            {name: 'Iron (Fe)', value: input.Fe, unit: 'mg/kg', optimal: '8-10'}
                        ].map(param => `
                            <div class="parameter">
                                <span class="param-label">${param.name}</span>
                                <span class="param-value">${param.value} ${param.unit}</span>
                                <span class="param-optimal">Optimal: ${param.optimal} ${param.unit}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- Probability Distribution Card -->
                <div class="result-card probabilities">
                    <h3>📈 Fertility Probabilities</h3>
                    <div class="probability-bars">
                        ${['Low', 'Medium', 'High'].map((label, i) => `
                            <div class="prob-row">
                                <span class="prob-label">${label}</span>
                                <div class="prob-bar-container">
                                    <div class="prob-bar" style="width: ${(outputNumbers[i] * 100).toFixed(1)}%">
                                        <span class="prob-value">${(outputNumbers[i] * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <!-- Recommendations Card -->
                <div class="result-card recommendations">
                    <h3>💡 Recommendations</h3>
                    <div class="recommendations-content">
                        ${(() => {
                            const recommendations = [];
                            if (input.N < 350) recommendations.push('Consider increasing Nitrogen levels to reach optimal range (350-380 mg/kg)');
                            if (input.P < 11) recommendations.push('Phosphorus levels are low, aim for 11-13 mg/kg');
                            if (input.K < 800) recommendations.push('Increase Potassium content to optimal range (800-850 mg/kg)');
                            if (input.EC < 0.85) recommendations.push('Adjust EC levels to reach optimal range (0.85-0.95 dS/m)');
                            if (input.Fe < 8) recommendations.push('Iron content is low, target 8-10 mg/kg');
                            
                            return recommendations.length > 0 
                                ? recommendations.map(r => `<p>• ${r}</p>`).join('')
                                : '<p>All parameters are within or close to optimal ranges.</p>';
                        })()}
                    </div>
                </div>

                <!-- Model Info Card -->
                <div class="result-card model-info">
                    <h3>ℹ️ Analysis Details</h3>
                    <p>Model: ${selectedModel.replace(/_/g, ' ').replace(/classifier/i, 'Classifier')}</p>
                    <p>Analysis Time: ${new Date().toLocaleTimeString()}</p>
                    <p>Based on combined original and synthetic training data</p>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('Analysis error:', error);
        resultText.textContent = 'Error during analysis';
        
        // Prepare detailed error information
        const errorDetails = [];
        if (error.message) errorDetails.push(`Error Message: ${error.message}`);
        if (session) errorDetails.push(`Model: ${selectedModel}`);
        if (input) errorDetails.push(`Input Values: N=${input.N}, P=${input.P}, K=${input.K}, EC=${input.EC}, Fe=${input.Fe}`);
        
        resultDetails.innerHTML = `
            <p class="error-message">An error occurred during soil analysis.</p>
            <p class="error-message">Please try again or select a different model.</p>
            <div class="error-details">
                <strong>Technical Details:</strong><br>
                ${errorDetails.join('<br>')}
                <br><br>
                <strong>Stack Trace:</strong><br>
                ${error.stack || error}
            </div>
        `;
    } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('loading');
        analyzeBtn.textContent = 'Analyze Soil';
    }
}

// Existing code continues below...
// ...existing code...


