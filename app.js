const FEATURE_ORDER = [
  "Income",
  "Age",
  "Experience",
  "Married",
  "House_Ownership",
  "Car_Ownership",
  "Profession",
  "CITY",
  "STATE",
  "CURRENT_JOB_YRS",
  "CURRENT_HOUSE_YRS",
];

const NUMERIC_FIELDS = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"];

const predictionText = document.getElementById("predictionText");
const riskScore = document.getElementById("riskScore");
const riskBar = document.getElementById("riskBar");
const reasonText = document.getElementById("reasonText");
const modelStatus = document.getElementById("modelStatus");
const form = document.getElementById("loanForm");

let modelSession = null;
let categoryMapping = {};

async function loadModelAssets() {
  try {
    const mappingResponse = await fetch("feature_mapping.json");
    if (!mappingResponse.ok) {
      throw new Error("feature_mapping.json not found. Run the notebook to generate it.");
    }
    categoryMapping = await mappingResponse.json();

    modelSession = await ort.InferenceSession.create("loan_risk_model.onnx");
    modelStatus.textContent = "Model ready";
    modelStatus.style.borderColor = "rgba(52, 211, 153, 0.35)";
    modelStatus.style.background = "rgba(52, 211, 153, 0.12)";
    modelStatus.style.color = "#34d399";
  } catch (error) {
    modelStatus.textContent = "Model missing";
    modelStatus.style.borderColor = "rgba(248, 113, 113, 0.35)";
    modelStatus.style.background = "rgba(248, 113, 113, 0.12)";
    modelStatus.style.color = "#f87171";
    reasonText.textContent = error.message;
    console.error(error);
  }
}

function encodeInputFromForm(formData) {
  const values = [];

  FEATURE_ORDER.forEach((fieldName) => {
    const rawValue = formData.get(fieldName);

    if (NUMERIC_FIELDS.includes(fieldName)) {
      values.push(Number(rawValue || 0));
      return;
    }

    const map = categoryMapping[fieldName] || {};
    const encoded = map[String(rawValue)] ?? 0;
    values.push(Number(encoded));
  });

  return new ort.Tensor("float32", new Float32Array(values), [1, values.length]);
}

function updateRiskVisual(score) {
  const clamped = Math.max(0, Math.min(100, score));
  const label = clamped >= 70 ? "High Risk" : clamped >= 40 ? "Medium Risk" : "Low Risk";

  riskScore.textContent = Math.round(clamped);
  riskBar.style.width = `${clamped}%`;

  const ring = document.querySelector(".ring");
  ring.style.background = `conic-gradient(
    ${clamped >= 70 ? "#f87171" : clamped >= 40 ? "#fbbf24" : "#34d399"} 0deg,
    ${clamped >= 70 ? "#f87171" : clamped >= 40 ? "#fbbf24" : "#34d399"} ${clamped * 3.6}deg,
    rgba(148, 163, 184, 0.12) ${clamped * 3.6}deg 360deg
  )`;

  predictionText.textContent = label;
  predictionText.style.color = clamped >= 70 ? "#f87171" : clamped >= 40 ? "#fbbf24" : "#34d399";

  if (clamped >= 70) {
    reasonText.textContent = "This customer shows a high chance of default and may need stricter underwriting checks.";
  } else if (clamped >= 40) {
    reasonText.textContent = "This is a moderate-risk profile. Additional review may reduce approval risk.";
  } else {
    reasonText.textContent = "This customer appears financially stable and is within a healthy risk range.";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!modelSession) {
    reasonText.textContent = "The ONNX model is not available yet. Run the training notebook first to export the model.";
    return;
  }

  const inputTensor = encodeInputFromForm(new FormData(form));
  const outputs = await modelSession.run({ input: inputTensor });
  const output = outputs[Object.keys(outputs)[0]];
  const probability = Number(output.data[1]);
  const riskPercent = probability * 100;

  updateRiskVisual(riskPercent);
});

loadModelAssets();
