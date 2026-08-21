const canvas = document.getElementById('canvas');
const outputDiv = document.getElementById('output');

let ctx = null;
let isDrawing = false;
let timeoutId = null;
let model = null;

if (canvas) {
    ctx = canvas.getContext('2d');

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "white"; 
    ctx.lineWidth = 18;        
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    canvas.addEventListener('touchstart', (e) => { startDrawing(e.touches[0]); e.preventDefault(); }, {passive: false});
    canvas.addEventListener('touchmove', (e) => { draw(e.touches[0]); e.preventDefault(); }, {passive: false});
    canvas.addEventListener('touchend', stopDrawing);

    init();
}

async function init() {
    outputDiv.innerText = "Initializing WebGL...";
    await tf.setBackend('webgl');
    outputDiv.innerText = "Loading model...";
    try {
        // Notice relative path to local model folder
        model = await tf.loadLayersModel('model/model.json');
        outputDiv.innerText = "Ready. Draw a digit!";
        outputDiv.style.color = "var(--text-main)";
    } catch (e) {
        console.error("Failed to load model:", e);
        outputDiv.innerText = "Error loading model!";
        outputDiv.style.color = "#ff5555";
    }
}

function startDrawing(e) {
    isDrawing = true;
    ctx.beginPath();
    const rect = canvas.getBoundingClientRect();
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

    clearTimeout(timeoutId);
    timeoutId = setTimeout(predict, 200);
}

function stopDrawing() {
    if(isDrawing) {
        isDrawing = false;
        predict();
    }
}

window.clearCanvas = function() {
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (outputDiv) {
        outputDiv.innerText = "Ready. Draw a digit!";
        outputDiv.style.color = "var(--text-main)";
    }
}

async function predict() {
    if (!model || !canvas) return;

    outputDiv.innerText = "Predicting...";
    outputDiv.style.color = "var(--text-muted)";
    await new Promise(resolve => setTimeout(resolve, 0));

    try {
        const result = tf.tidy(() => {
            let tensor = tf.browser.fromPixels(canvas, 1);
            tensor = tf.image.resizeBilinear(tensor, [28, 28]);
            tensor = tensor.toFloat().div(255.0);
            tensor = tensor.expandDims(0);
            
            const predictions = model.predict(tensor);
            const predictedDigit = predictions.argMax(1).dataSync()[0];
            const confidence = predictions.max(1).dataSync()[0];
            
            return { digit: predictedDigit, confidence: (confidence * 100).toFixed(1) };
        });

        outputDiv.innerText = `Prediction: ${result.digit} (${result.confidence}%)`;
        outputDiv.style.color = "var(--accent)";
    } catch(e) {
        console.error("Failed to predict:", e);
        outputDiv.innerText = "Error during prediction.";
        outputDiv.style.color = "#ff5555";
    }
}
