import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Absolute Path Resolution for Vercel Serverless Environments
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # points to /api
PARENT_DIR = os.path.dirname(CURRENT_DIR)               # points to project root

# Candidate model filenames
MODEL_FILENAMES = ["model_logistic.pkl", "model_logistic_regression.pkl"]

MODEL_PATH = None
for filename in MODEL_FILENAMES:
    root_path = os.path.join(PARENT_DIR, filename)
    api_path = os.path.join(CURRENT_DIR, filename)
    
    if os.path.exists(root_path):
        MODEL_PATH = root_path
        break
    elif os.path.exists(api_path):
        MODEL_PATH = api_path
        break

# Load model pickle
model = None
if MODEL_PATH:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TalentPulse AI | Workforce Analytics Engine</title>
    <!-- Google Fonts, FontAwesome, jsPDF -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>

    <style>
        :root {
            --bg-dark: #0a0d14;
            --card-glass: rgba(18, 24, 38, 0.8);
            --card-border: rgba(245, 158, 11, 0.15);
            --amber-glow: #f59e0b;
            --amber-bright: #fbbf24;
            --amber-dim: rgba(245, 158, 11, 0.15);
            --emerald-glow: #10b981;
            --rose-glow: #f43f5e;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --input-bg: rgba(10, 13, 20, 0.7);
            --input-border: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        .ambient-orb {
            position: fixed;
            width: 600px;
            height: 600px;
            border-radius: 50%;
            filter: blur(140px);
            z-index: -1;
            opacity: 0.2;
            pointer-events: none;
        }
        .orb-amber { top: -200px; right: -100px; background: var(--amber-glow); }
        .orb-gold { bottom: -200px; left: -100px; background: #d97706; }

        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 6%;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10, 13, 20, 0.85);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.3rem;
            font-weight: 800;
        }

        .brand-logo {
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, var(--amber-glow), #d97706);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0a0d14;
            font-size: 1.2rem;
            box-shadow: 0 0 20px rgba(245, 158, 11, 0.35);
        }

        .nav-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: var(--amber-glow);
            background: var(--amber-dim);
            border: 1px solid rgba(245, 158, 11, 0.3);
            padding: 6px 14px;
            border-radius: 30px;
        }

        .workspace-container {
            max-width: 1400px;
            margin: 35px auto;
            padding: 0 4%;
            width: 100%;
            flex-grow: 1;
        }

        .header-section {
            text-align: center;
            margin-bottom: 38px;
        }

        .title-badge {
            display: inline-block;
            padding: 6px 18px;
            background: var(--amber-dim);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 30px;
            color: var(--amber-bright);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
        }

        .main-title {
            font-size: 2.7rem;
            font-weight: 800;
            letter-spacing: -1px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #ffffff 0%, #fde68a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .main-subtitle {
            color: var(--text-sub);
            font-size: 1.05rem;
            max-width: 650px;
            margin: 0 auto;
        }

        .grid-layout {
            display: grid;
            grid-template-columns: 1.85fr 1.15fr;
            gap: 32px;
            align-items: start;
        }

        .sticky-sidebar {
            position: sticky;
            top: 110px;
            z-index: 10;
        }

        .glass-card {
            background: var(--card-glass);
            border-radius: 28px;
            border: 1px solid var(--card-border);
            padding: 34px;
            backdrop-filter: blur(24px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.6);
            margin-bottom: 24px;
        }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-title i {
            color: var(--amber-glow);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 0.82rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
        }

        .live-val {
            font-family: 'JetBrains Mono', monospace;
            color: var(--amber-bright);
            font-weight: 700;
        }

        input, select {
            width: 100%;
            padding: 13px 15px;
            background-color: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            color: var(--text-main);
            font-size: 0.92rem;
            font-weight: 500;
            transition: all 0.25s ease;
        }

        select {
            appearance: none;
            cursor: pointer;
            padding-right: 36px;
        }

        .select-wrapper {
            position: relative;
        }

        .select-wrapper::after {
            content: '\f107';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            position: absolute;
            right: 14px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-sub);
            pointer-events: none;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--amber-glow);
            box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.15);
            background-color: rgba(10, 13, 20, 0.9);
        }

        .btn-predict {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, var(--amber-glow) 0%, #d97706 100%);
            color: #0a0d14;
            font-weight: 800;
            font-size: 1.05rem;
            border: none;
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-top: 26px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            box-shadow: 0 10px 30px rgba(245, 158, 11, 0.25);
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 40px rgba(245, 158, 11, 0.4);
            color: #ffffff;
        }

        .valuation-card {
            background: linear-gradient(180deg, rgba(245, 158, 11, 0.12) 0%, rgba(10, 13, 20, 0.05) 100%);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 20px;
            padding: 26px;
            text-align: center;
        }

        .val-tag {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-sub);
            font-weight: 700;
            margin-bottom: 6px;
        }

        .val-price {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
            margin: 8px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .feature-badge-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 18px;
        }

        .f-badge {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 10px;
            text-align: center;
        }

        .f-badge-title {
            font-size: 0.68rem;
            color: var(--text-sub);
            text-transform: uppercase;
        }

        .f-badge-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--amber-bright);
            margin-top: 4px;
        }

        .btn-report {
            width: 100%;
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-main);
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.88rem;
            cursor: pointer;
            margin-top: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: all 0.25s ease;
        }

        .btn-report:hover {
            background: var(--amber-dim);
            border-color: var(--amber-glow);
            color: var(--amber-bright);
        }

        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(10, 13, 20, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            text-align: center;
            padding: 24px;
            color: var(--text-sub);
            font-size: 0.85rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            margin-top: auto;
        }

        @media (max-width: 1024px) {
            .grid-layout { grid-template-columns: 1fr; }
            .form-grid { grid-template-columns: 1fr; }
            .sticky-sidebar { position: static; }
        }
    </style>
</head>
<body>

<div class="ambient-orb orb-amber"></div>
<div class="ambient-orb orb-gold"></div>

<nav class="navbar">
    <div class="brand">
        <div class="brand-logo">
            <i class="fa-solid fa-user-gear"></i>
        </div>
        <span>TalentPulse<span style="color: var(--amber-glow);">.ai</span></span>
    </div>
    <div class="nav-tag">
        <i class="fa-solid fa-microchip"></i> Logistic Regression Engine
    </div>
</nav>

<div class="workspace-container">
    <div class="header-section">
        <span class="title-badge">HR Executive Intelligence</span>
        <h1 class="main-title">Workforce Attrition Predictor</h1>
        <p class="main-subtitle">Analyze demographics, tenure, tier status, and bench history to forecast employee attrition risks in real-time.</p>
    </div>

    <div class="grid-layout">
        
        <!-- Left Column: Inputs -->
        <div class="glass-card">
            <div class="section-header">
                <span class="section-title"><i class="fa-solid fa-sliders"></i> Employee Attributes</span>
                <span style="font-size: 0.8rem; color: var(--text-sub);">8 Data Points Configured</span>
            </div>

            <form id="predictionForm">
                <div class="form-grid">
                    
                    <div class="form-group">
                        <label>Education Tier</label>
                        <div class="select-wrapper">
                            <select name="Education" required>
                                <option value="0" selected>Bachelor's</option>
                                <option value="1">Master's</option>
                                <option value="2">PHD</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Joining Year <span class="live-val" id="joinYearVal">2021</span></label>
                        <input type="number" id="joiningYearInput" name="JoiningYear" value="2021" min="2000" max="2026" required oninput="document.getElementById('joinYearVal').textContent = this.value; updateTenure(this.value)">
                    </div>

                    <div class="form-group">
                        <label>City Hub</label>
                        <div class="select-wrapper">
                            <select name="City" required>
                                <option value="0" selected>Bangalore</option>
                                <option value="1">Pune</option>
                                <option value="2">New Delhi</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Payment Tier</label>
                        <div class="select-wrapper">
                            <select name="PaymentTier" required>
                                <option value="1">Tier 1 (Highest)</option>
                                <option value="2">Tier 2</option>
                                <option value="3" selected>Tier 3 (Standard)</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Age <span class="live-val" id="ageVal">28</span></label>
                        <input type="number" name="Age" value="28" min="18" max="75" required oninput="document.getElementById('ageVal').textContent = this.value">
                    </div>

                    <div class="form-group">
                        <label>Gender Orientation</label>
                        <div class="select-wrapper">
                            <select name="Gender" required>
                                <option value="1">Male</option>
                                <option value="0">Female</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Ever Benched?</label>
                        <div class="select-wrapper">
                            <select name="EverBenched" required>
                                <option value="0" selected>No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Domain Experience (Yrs) <span class="live-val" id="expVal">3</span></label>
                        <input type="number" name="ExperienceInCurrentDomain" value="3" min="0" max="40" required oninput="document.getElementById('expVal').textContent = this.value">
                    </div>

                </div>

                <button type="submit" class="btn-predict" id="submitBtn">
                    <span class="btn-text">Execute Attrition Assessment</span>
                    <div class="spinner" id="btnSpinner"></div>
                </button>
            </form>
        </div>

        <!-- Right Column: Output Card -->
        <div class="sticky-sidebar" id="outputSection">
            <div class="glass-card">
                <div class="section-header">
                    <span class="section-title"><i class="fa-solid fa-chart-pie"></i> Retention Telemetry</span>
                </div>

                <div class="valuation-card" id="resultCard">
                    <div class="val-tag">Predicted Retention Status</div>
                    <div class="val-price">
                        <i class="fa-solid fa-shield-halved" id="resultIcon" style="color: var(--amber-glow);"></i>
                        <span id="resultOutput">Awaiting Input</span>
                    </div>

                    <div class="feature-badge-grid">
                        <div class="f-badge">
                            <div class="f-badge-title">Company Tenure</div>
                            <div class="f-badge-val" id="badgeTenure">5 Yrs</div>
                        </div>
                        <div class="f-badge">
                            <div class="f-badge-title">Risk Level</div>
                            <div class="f-badge-val" id="badgeRisk">Minimal</div>
                        </div>
                    </div>
                </div>

                <button type="button" class="btn-report" onclick="downloadPDFReport()">
                    <i class="fa-solid fa-file-pdf"></i> Download Executive HR Brief
                </button>
            </div>
        </div>

    </div>
</div>

<footer>
    &copy; 2026 TalentPulse AI Platform &bull; Hosted on Vercel Serverless Architecture
</footer>

<script>
    let lastResult = "Retained";

    function updateTenure(year) {
        if (year && year > 0) {
            let tenure = 2026 - Number(year);
            document.getElementById('badgeTenure').textContent = Math.max(0, tenure) + ' Yrs';
        }
    }

    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const form = e.target;
        const submitBtn = document.getElementById('submitBtn');
        const spinner = document.getElementById('btnSpinner');
        const btnText = submitBtn.querySelector('.btn-text');
        const resultOutput = document.getElementById('resultOutput');
        const resultIcon = document.getElementById('resultIcon');
        const resultCard = document.getElementById('resultCard');
        const badgeRisk = document.getElementById('badgeRisk');
        const outputSection = document.getElementById('outputSection');
        
        submitBtn.disabled = true;
        spinner.style.display = 'block';
        btnText.textContent = 'Analyzing Logistic Weights...';
        
        const formData = new FormData(form);
        
        try {
            const response = await fetch('/api/index', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server Error (${response.status}): ${errText.substring(0, 80)}`);
            }
            
            const data = await response.json();
            
            setTimeout(() => {
                if (data.status === 'success') {
                    lastResult = data.prediction;
                    resultOutput.textContent = data.prediction;
                    
                    if (data.prediction === "Retained") {
                        resultCard.style.background = "linear-gradient(180deg, rgba(16, 185, 129, 0.15) 0%, rgba(10, 13, 20, 0.05) 100%)";
                        resultCard.style.borderColor = "rgba(16, 185, 129, 0.4)";
                        resultIcon.className = "fa-solid fa-circle-check";
                        resultIcon.style.color = "var(--emerald-glow)";
                        badgeRisk.textContent = "Low Risk";
                        badgeRisk.style.color = "var(--emerald-glow)";
                    } else {
                        resultCard.style.background = "linear-gradient(180deg, rgba(244, 63, 94, 0.15) 0%, rgba(10, 13, 20, 0.05) 100%)";
                        resultCard.style.borderColor = "rgba(244, 63, 94, 0.4)";
                        resultIcon.className = "fa-solid fa-triangle-exclamation";
                        resultIcon.style.color = "var(--rose-glow)";
                        badgeRisk.textContent = "High Risk";
                        badgeRisk.style.color = "var(--rose-glow)";
                    }
                    
                    if (window.innerWidth <= 1024) {
                        outputSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    }
                } else {
                    resultOutput.textContent = 'Error: ' + data.message;
                }
                
                submitBtn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = 'Execute Attrition Assessment';
            }, 400);

        } catch (error) {
            submitBtn.disabled = false;
            spinner.style.display = 'none';
            btnText.textContent = 'Execute Attrition Assessment';
            resultOutput.textContent = error.message;
        }
    });

    function downloadPDFReport() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        doc.setFont("helvetica", "bold");
        doc.setFontSize(20);
        doc.text("TalentPulse AI - HR Risk Brief", 20, 22);
        
        doc.setFontSize(10);
        doc.setFont("helvetica", "normal");
        doc.text("Date: " + new Date().toLocaleDateString() + " | Assessment ID: #" + Math.floor(Math.random()*899999+100000), 20, 30);
        
        doc.line(20, 35, 190, 35);
        
        doc.setFontSize(14);
        doc.setFont("helvetica", "bold");
        doc.text("Forecasted Retention Status: " + lastResult, 20, 48);
        
        doc.setFontSize(11);
        doc.setFont("helvetica", "normal");
        doc.text("Tenure: " + document.getElementById('badgeTenure').textContent, 20, 60);
        doc.text("Risk Classification: " + document.getElementById('badgeRisk').textContent, 20, 68);
        
        doc.line(20, 78, 190, 78);
        doc.setFontSize(9);
        doc.text("Disclaimer: Predicted output generated via Logistic Regression classification model.", 20, 88);
        
        doc.save("HR_Retention_Report.pdf");
    }
</script>

</body>
</html>
"""

@app.route('/', methods=['GET'])
@app.route('/api/index', methods=['GET', 'POST'])
def handler():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)
        
    if model is None:
        return jsonify({'status': 'error', 'message': 'Logistic Regression model not loaded on server. Verify file location.'}), 500
        
    try:
        # Match exact 8 feature column names from pickled model
        data_dict = {
            'Education': [float(request.form['Education'])],
            'JoiningYear': [float(request.form['JoiningYear'])],
            'City': [float(request.form['City'])],
            'PaymentTier': [float(request.form['PaymentTier'])],
            'Age': [float(request.form['Age'])],
            'Gender': [float(request.form['Gender'])],
            'EverBenched': [float(request.form['EverBenched'])],
            'ExperienceInCurrentDomain': [float(request.form['ExperienceInCurrentDomain'])]
        }
        
        features_df = pd.DataFrame(data_dict)
        raw_pred = int(model.predict(features_df)[0])
        
        # Binary Classification Mapping
        label_mapping = {
            0: "Retained",
            1: "At-Risk / Attrition"
        }
        
        final_output = label_mapping.get(raw_pred, f"Class [{raw_pred}]")
        
        return jsonify({
            'status': 'success',
            'prediction': final_output
        })
        
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
