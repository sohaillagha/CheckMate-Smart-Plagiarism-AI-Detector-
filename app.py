import sys
import os
import uuid
from datetime import datetime
import io
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename

# Add 'pdf_reader' to sys.path so inner imports (like 'from pdfextraction...') work
sys.path.append(os.path.join(os.path.dirname(__file__), 'pdf_reader'))

from pdf_reader.analyzer import analyze_pdf

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_prototype'  # Change for production

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'userinput')
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple in-memory cache for report results
RESULTS_CACHE = {}
UPLOAD_HISTORY = []  # List of dicts: {report_id, filename, date, originality_score}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple hardcoded check for prototype
        if username == 'admin' and password == 'admin':
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials! Try admin/admin')
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'], uploads=UPLOAD_HISTORY)

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    # Compute stats from upload history
    total = len(UPLOAD_HISTORY)
    if total > 0:
        scores = [u['originality_score'] for u in UPLOAD_HISTORY]
        avg = sum(scores) / total * 100
        best = max(scores) * 100
    else:
        avg = 0
        best = 0

    stats = {
        'total_uploads': total,
        'avg_originality': avg,
        'best_score': best,
    }
    return render_template('profile.html', user=session['user'], stats=stats, uploads=UPLOAD_HISTORY)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('dashboard'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run Analysis
        results = analyze_pdf(filepath)
        
        # Store results in cache with a unique ID
        report_id = str(uuid.uuid4())
        RESULTS_CACHE[report_id] = {
            'results': results,
            'filename': filename
        }

        # Track upload history for the dashboard
        UPLOAD_HISTORY.insert(0, {
            'report_id': report_id,
            'filename': filename,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'originality_score': results.get('originality_score', 0.0)
        })
        
        # Pass results and report_id to result page
        return render_template('result.html', results=results, filename=filename, report_id=report_id)

    flash('Invalid file type. Please upload a PDF.')
    return redirect(url_for('dashboard'))

@app.route('/download_report/<report_id>')
def download_report(report_id):
    if 'user' not in session:
        return redirect(url_for('login'))
        
    data = RESULTS_CACHE.get(report_id)
    if not data:
        flash('Report expired or not found.')
        return redirect(url_for('dashboard'))
        
    results = data['results']
    filename = data['filename']
    
    # Generate Text Report
    buffer = io.BytesIO()
    
    def w(text):
        buffer.write((text + "\n").encode('utf-8'))
        
    w("-" * 50)
    w("CHECKMATE ANALYSIS REPORT")
    w("-" * 50)
    w(f"File: {filename}")
    w("-" * 50)
    w("")
    w("ALGORITHM SCORE SUMMARY:")
    w("-" * 25)
    w(f"[SCORE] Originality Score: {results['originality_score']*100:.0f}%")
    w(f"[INFO]  Copied Content:    {results['tfidf_score']*100:.0f}% (TF-IDF Match)")
    w(f"[INFO]  Paraphrased:       {results['sbert_score']*100:.0f}% (SBERT Match)")
    w(f"[INFO]  AI Generated:      {results['ai_prob']*100:.0f}% (AI Probability)")
    w("")
    verdict = "APPROVED" if results['originality_score'] > 0.6 else "NEEDS REVISION"
    w(f"VERDICT: {verdict}")
    w("-" * 50)
    w("DETAILED CONTENT ANALYSIS")
    w("-" * 50)
    w("")
    
    if results.get('content_analysis'):
        for item in results['content_analysis']:
            sentence = item['text'].strip()
            labels = item.get('labels', [])
            
            tag = "[NORMAL]       "
            if 'copied' in labels:
                tag = "[COPIED]       "
            elif 'paraphrased' in labels:
                tag = "[PARAPHRASED]  "
            elif 'ai' in labels:
                tag = "[AI]           "
                
            w(f"{tag} {sentence}")
            w("") # Spacer
    else:
        w("No detailed analysis available.")
        
    w("-" * 50)
    w("CheckMate - PDF Originality Analyzer")
    w("-" * 50)
    
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"report_{filename}.txt",
        mimetype='text/plain'
    )

@app.route('/view_report/<report_id>')
def view_report(report_id):
    if 'user' not in session:
        return redirect(url_for('login'))

    data = RESULTS_CACHE.get(report_id)
    if not data:
        flash('Report expired or not found.')
        return redirect(url_for('dashboard'))

    return render_template('result.html', results=data['results'], filename=data['filename'], report_id=report_id)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
