const form = document.querySelector("#uploadForm");
const input = document.querySelector("#imageInput");
const imageUrlInput = document.querySelector("#imageUrlInput");
const dropZone = document.querySelector("#dropZone");
const predictButton = document.querySelector("#predictButton");
const resetButton = document.querySelector("#resetButton");
const statusText = document.querySelector("#statusText");
const previewImage = document.querySelector("#previewImage");
const emptyPreview = document.querySelector("#emptyPreview");
const resultLabel = document.querySelector("#resultLabel");
const confidenceValue = document.querySelector("#confidenceValue");
const confidenceBar = document.querySelector("#confidenceBar");
const sortingTip = document.querySelector("#sortingTip");
const topPredictions = document.querySelector("#topPredictions");

let isSubmitting = false;

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.style.color = isError ? "#b23b2f" : "#68736d";
}

function showPreview(source) {
  if (!source) {
    previewImage.removeAttribute("src");
    previewImage.style.display = "none";
    emptyPreview.style.display = "grid";
    return;
  }

  previewImage.src = typeof source === "string" ? source : URL.createObjectURL(source);
  previewImage.style.display = "block";
  emptyPreview.style.display = "none";
}

function renderResult(data) {
  resultLabel.textContent = data.label;
  confidenceValue.textContent = `${data.confidence}%`;
  confidenceBar.style.width = `${Math.max(0, Math.min(100, data.confidence))}%`;
  sortingTip.textContent = data.tip;
  topPredictions.innerHTML = "";

  data.top_predictions.forEach((item) => {
    const row = document.createElement("div");
    row.className = "top-item";
    row.innerHTML = `<span>${item.label}</span><span>${item.confidence}%</span>`;
    topPredictions.appendChild(row);
  });
}

function resetResult() {
  input.value = "";
  imageUrlInput.value = "";
  showPreview(null);
  resultLabel.textContent = "En attente d'analyse";
  confidenceValue.textContent = "0%";
  confidenceBar.style.width = "0%";
  sortingTip.textContent = "Le conseil de tri apparaitra ici apres prediction.";
  topPredictions.innerHTML = "";
  setStatus("");
}

input.addEventListener("change", () => {
  if (input.files[0]) {
    imageUrlInput.value = "";
    dropZone.classList.add("has-file");
  }
  showPreview(input.files[0]);
  setStatus(input.files[0] ? `${input.files[0].name} selectionnee. Cliquez sur Analyser l'image.` : "");
});

imageUrlInput.addEventListener("input", () => {
  if (imageUrlInput.value.trim()) {
    input.value = "";
    showPreview(null);
    setStatus("URL prete pour analyse.");
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) return;

  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  input.files = dataTransfer.files;
  imageUrlInput.value = "";
  dropZone.classList.add("has-file");
  showPreview(file);
  setStatus(`${file.name} selectionnee. Cliquez sur Analyser l'image.`);
});

async function submitPrediction(event) {
  event.preventDefault();
  event.stopPropagation();

  if (isSubmitting) {
    return;
  }

  const imageUrl = imageUrlInput.value.trim();

  if (!input.files[0] && !imageUrl) {
    setStatus("Ajoutez une image ou collez une URL avant l'analyse.", true);
    return;
  }

  const formData = new FormData();
  if (input.files[0]) {
    formData.append("image", input.files[0]);
  } else {
    formData.append("image_url", imageUrl);
  }

  isSubmitting = true;
  predictButton.disabled = true;
  setStatus("Analyse en cours...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Erreur pendant l'analyse.");
    }

    renderResult(data);
    showPreview(data.image_url);
    setStatus("Analyse terminee.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    isSubmitting = false;
    predictButton.disabled = false;
  }
}

form.addEventListener("submit", submitPrediction);
predictButton.addEventListener("click", submitPrediction);

resetButton.addEventListener("click", resetResult);
