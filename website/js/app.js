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


