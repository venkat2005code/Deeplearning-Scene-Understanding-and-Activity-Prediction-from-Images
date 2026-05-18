// DOM Elements
const dropZone = document.getElementById("dropZone");
const imageInput = document.getElementById("imageInput");
const previewSection = document.getElementById("previewSection");
const imagePreview = document.getElementById("imagePreview");
const analyzeBtn = document.getElementById("analyzeBtn");
const changeImageBtn = document.getElementById("changeImageBtn");
const loadingState = document.getElementById("loadingState");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");
const resultsPanel = document.getElementById("resultsPanel");
const welcomePanel = document.getElementById("welcomePanel");
const scanningOverlay = document.getElementById("scanningOverlay");
const modelStatus = document.getElementById("modelStatus");

// Object icons mapping
const objectIcons = {
    Car: "🚗",
    Person: "🚶",
    Bike: "🏍️",
    Boat: "🚤",
    Backpack: "🎒",
    Umbrella: "☂️",
    Bag: "👜",
};

let selectedFile = null;
let places365Available = false;

// ============ Initialize Status on Page Load ============
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch("/api/status");
        const status = await response.json();
        
        places365Available = status.places365_available;
        
        // Update status indicator
        if (modelStatus) {
            const statusIndicator = modelStatus.querySelector(".status-indicator");
            const statusText = modelStatus.querySelector(".status-text");
            
            if (statusIndicator && statusText) {
                if (places365Available) {
                    statusIndicator.style.backgroundColor = "#10b981"; // green
                    statusText.textContent = "Using Places365 + ImageNet";
                    statusIndicator.title = "Places365-trained model active";
                } else {
                    statusIndicator.style.backgroundColor = "#f59e0b"; // amber
                    statusText.textContent = "Using ImageNet";
                    statusIndicator.title = "Places365 model not loaded";
                }
            }
        }
    } catch (error) {
        console.log("Could not fetch model status:", error);
        if (modelStatus) {
            const statusText = modelStatus.querySelector(".status-text");
            if (statusText) {
                statusText.textContent = "Status unavailable";
            }
        }
    }
});

// ============ Drag & Drop Handlers ============
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

dropZone.addEventListener("click", () => {
    imageInput.click();
});

imageInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    clearError();
    
    if (!file.type.startsWith("image/")) {
        showError("Please upload a valid image file.");
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        previewSection.classList.remove("hidden");
        welcomePanel.classList.add("hidden");
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// ============ Analyze Button ============
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
        showError("Please choose an image first.");
        return;
    }

    clearError();
    analyzeBtn.disabled = true;
    loadingState.classList.remove("hidden");
    resultsPanel.classList.add("hidden");
    scanningOverlay.classList.remove("hidden");

    const formData = new FormData();
    formData.append("image", selectedFile);

    // Safety timeout - hide loading after 30 seconds max
    const timeoutId = setTimeout(() => {
        if (!resultsPanel.classList.contains("hidden")) return;
        scanningOverlay.classList.add("hidden");
        loadingState.classList.add("hidden");
        showError("Analysis took too long. Please try again.");
        analyzeBtn.disabled = false;
    }, 30000);

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        // Wait for scanning animation to finish
        setTimeout(() => {
            try {
                scanningOverlay.classList.add("hidden");
                loadingState.classList.add("hidden");
                displayResults(data);
                resultsPanel.classList.remove("hidden");
                clearTimeout(timeoutId);
            } catch (displayError) {
                console.error("Error displaying results:", displayError);
                showError("Error displaying results: " + displayError.message);
                scanningOverlay.classList.add("hidden");
                loadingState.classList.add("hidden");
            }
        }, 2000);
    } catch (error) {
        clearTimeout(timeoutId);
        scanningOverlay.classList.add("hidden");
        loadingState.classList.add("hidden");
        showError(error.message || "An unexpected error occurred.");
    } finally {
        analyzeBtn.disabled = false;
    }
}
);

changeImageBtn.addEventListener("click", () => {
    resetUI();
    imageInput.click();
});

// ============ Display Results with Animations ============
function displayResults(data) {
    if (!data) {
        showError("No data received from server.");
        return;
    }

    try {
        welcomePanel.classList.add("hidden");

        // Scene with glow animation
        const sceneValue = document.getElementById("sceneValue");
        if (sceneValue) {
            typeWriter(sceneValue, data.scene || "Unknown", 50);
        }

        // Objects with tags
        const objectsContainer = document.getElementById("objectsContainer");
        if (objectsContainer) {
            objectsContainer.innerHTML = "";
            
            if (data.objects && Array.isArray(data.objects) && data.objects.length > 0) {
                data.objects.forEach((objName, index) => {
                    setTimeout(() => {
                        const tag = document.createElement("div");
                        tag.className = "object-tag";
                        tag.innerHTML = `<span>${objectIcons[objName] || "📦"}</span> ${objName} present`;
                        objectsContainer.appendChild(tag);
                    }, index * 150);
                });
            } else {
                objectsContainer.innerHTML = '<p class="placeholder">No major objects detected</p>';
            }
        }

        // Activity
        const activityValue = document.getElementById("activityValue");
        if (activityValue) {
            typeWriter(activityValue, data.activity || "Unknown", 30);
        }

        // Risk Level with color coding
        const riskIndicator = document.getElementById("riskIndicator");
        const riskValue = document.getElementById("riskValue");
        if (riskIndicator && riskValue) {
            const riskLevel = (data.risk_level || "Unknown").toLowerCase();
            
            riskIndicator.className = `risk-indicator ${riskLevel}`;
            riskValue.textContent = data.risk_level || "Unknown";
        }

        // Event
        const eventValue = document.getElementById("eventValue");
        if (eventValue) {
            typeWriter(eventValue, data.event || "Unknown", 30);
        }

        // Model Used
        const modelUsed = document.getElementById("modelUsed");
        if (modelUsed) {
            const modelText = data.model_used || "Standard Model";
            typeWriter(modelUsed, modelText, 20);
        }

        // Processed image
        if (data.processed_image) {
            const processedImageContainer = document.getElementById("processedImageContainer");
            const processedImage = document.getElementById("processedImage");
            if (processedImageContainer && processedImage) {
                const processedImagePath = `${data.processed_image}?t=${Date.now()}`;
                processedImage.src = processedImagePath;
                imagePreview.src = processedImagePath;
                processedImageContainer.style.display = "block";
            }
        }
    } catch (error) {
        console.error("Error in displayResults:", error);
        throw error;
    }
}

// ============ Typing Animation ============
function typeWriter(element, text, speed = 50) {
    if (!element) return;
    
    element.textContent = "";
    let index = 0;
    text = String(text || "");

    function type() {
        if (index < text.length) {
            element.textContent += text.charAt(index);
            index++;
            setTimeout(type, speed);
        }
    }

    if (text.length === 0) {
        element.textContent = "—";
    } else {
        type();
    }
}

// ============ Error Handling ============
function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove("hidden");
}

function clearError() {
    errorText.textContent = "";
    errorMessage.classList.add("hidden");
}

// ============ Reset UI ============
function resetUI() {
    previewSection.classList.add("hidden");
    resultsPanel.classList.add("hidden");
    welcomePanel.classList.remove("hidden");
    loadingState.classList.add("hidden");
    scanningOverlay.classList.add("hidden");
    imagePreview.src = "";
    selectedFile = null;
    imageInput.value = "";
    clearError();
    analyzeBtn.disabled = false;
}

// ============ Prevent default drag behavior on document ============
document.addEventListener("dragover", (e) => {
    e.preventDefault();
});

document.addEventListener("drop", (e) => {
    e.preventDefault();
});
