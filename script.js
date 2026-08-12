const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const outputDiv = document.getElementById('output');

let isDrawing = false;
let timeoutId = null;
let model = null;

// Initialize TensorFlow.js and load model
async function init() {
   outputDiv.innerText = "Initializing WebGL...";
   await tf.setBackend('webgl');
   outputDiv.innerText = "Loading model...";
   try {
       model = await tf.loadLayersModel('tfjs_model/model.json');
       outputDiv.innerText = "Draw a digit!";
   } catch (e) {
       console.error("Failed to load model:", e);
       outputDiv.innerText = "Error loading model!";
   }
}
init();

// 1. Initialize Canvas Settings
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white"; // White digit ink like MNIST
ctx.lineWidth = 18;        // Thick stroke matches handwritten style
ctx.lineCap = "round";
ctx.lineJoin = "round";

// 2. Attach Event Listeners for Drawing
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

// Touch Support for Mobile Browsers
canvas.addEventListener('touchstart', (e) => { startDrawing(e.touches[0]); e.preventDefault(); });
canvas.addEventListener('touchmove', (e) => { draw(e.touches[0]); e.preventDefault(); });
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
   isDrawing = true;
   ctx.beginPath();
   const rect = canvas.getBoundingClientRect();
   // Scale coordinate to canvas logical size
   const scaleX = canvas.width / rect.width;
   const scaleY = canvas.height / rect.height;
   ctx.moveTo((e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY);
}

function draw(e) {
   if (!isDrawing) return;
   const rect = canvas.getBoundingClientRect();
   const scaleX = canvas.width / rect.width;
   const scaleY = canvas.height / rect.height;
   ctx.lineTo((e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY);
   ctx.stroke();

   // Debounce the prediction so we don't spam the server
   clearTimeout(timeoutId);
   timeoutId = setTimeout(predict, 200);
}

function stopDrawing() {
   if(isDrawing) {
       isDrawing = false;
       predict();
   }
}

function clearCanvas() {
   ctx.fillStyle = "black";
   ctx.fillRect(0, 0, canvas.width, canvas.height);
   outputDiv.innerText = "Draw a digit!";
}

// 4. Client-side TensorFlow.js Prediction
async function predict() {
   if (!model) return;

   outputDiv.innerText = "Predicting...";
   // Yield to the browser so the "Predicting..." text renders before heavy compute
   await new Promise(resolve => setTimeout(resolve, 0));

   try {
       // Run inference inside tf.tidy to automatically clean up tensors
       const result = tf.tidy(() => {
           // Grab pixels from canvas (single channel)
           let tensor = tf.browser.fromPixels(canvas, 1);
           
           // Resize to 28x28
           tensor = tf.image.resizeBilinear(tensor, [28, 28]);
           
           // Normalize to [0.0, 1.0]
           tensor = tensor.toFloat().div(255.0);
           
           // Expand dims to [1, 28, 28, 1]
           tensor = tensor.expandDims(0);
           
           // Predict
           const predictions = model.predict(tensor);
           
           // Extract the argmax (predicted digit) and confidence
           const predictedDigit = predictions.argMax(1).dataSync()[0];
           const confidence = predictions.max(1).dataSync()[0];
           
           return { digit: predictedDigit, confidence: (confidence * 100).toFixed(2) };
       });

       outputDiv.innerText = `Prediction: ${result.digit} (${result.confidence}% confidence)`;
   } catch(e) {
       console.error("Failed to predict:", e);
       outputDiv.innerText = "Error during prediction.";
   }
}

// 5. Mobile Hamburger Navigation Logic
const hamburgerBtn = document.getElementById('hamburger-btn');
const sidebar = document.getElementById('sidebar');
const sidebarCloseBtn = document.getElementById('sidebar-close');
const sidebarOverlay = document.getElementById('sidebar-overlay');

function openSidebar() {
    if (sidebar) sidebar.classList.add('active');
    if (sidebarOverlay) sidebarOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeSidebar() {
    if (sidebar) sidebar.classList.remove('active');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');
    document.body.style.overflow = '';
}

if (hamburgerBtn) {
    hamburgerBtn.addEventListener('click', openSidebar);
}

if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener('click', closeSidebar);
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeSidebar);
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar && sidebar.classList.contains('active')) {
        closeSidebar();
    }
});