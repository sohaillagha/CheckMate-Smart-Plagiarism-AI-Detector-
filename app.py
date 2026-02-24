import sys
import os
import uuid
import json
import threading
from datetime import datetime
import io
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
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

# Global model weights (faculty-adjustable)
MODEL_WEIGHTS = {'tfidf': 0.4, 'sbert': 0.3, 'ai': 0.3}
UPLOAD_HISTORY = []  # List of dicts: {report_id, filename, date, originality_score}

# ArXiv import progress tracking
IMPORT_LOG = []
IMPORT_RUNNING = False

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
        role = request.form.get('role', 'student')
        
        # Simple hardcoded check for prototype
        if username == 'admin' and password == 'admin':
            session['user'] = username
            session['role'] = role
            if role == 'faculty':
                return redirect(url_for('faculty_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials! Try admin/admin')
            
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'], uploads=UPLOAD_HISTORY)

@app.route('/faculty_dashboard')
def faculty_dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    if session.get('role') != 'faculty':
        return redirect(url_for('dashboard'))
    return render_template('faculty_dashboard.html', user=session['user'], weights=MODEL_WEIGHTS, uploads=UPLOAD_HISTORY)

@app.route('/update_weights', methods=['POST'])
def update_weights():
    if 'user' not in session or session.get('role') != 'faculty':
        return redirect(url_for('login'))
    try:
        w_tfidf = float(request.form.get('tfidf', 40)) / 100.0
        w_sbert = float(request.form.get('sbert', 30)) / 100.0
        w_ai = float(request.form.get('ai', 30)) / 100.0
        total = round(w_tfidf + w_sbert + w_ai, 2)
        if total != 1.0:
            flash(f'Weights must sum to 1.0 (currently {total}). Please adjust.')
            return redirect(url_for('faculty_dashboard'))
        MODEL_WEIGHTS['tfidf'] = w_tfidf
        MODEL_WEIGHTS['sbert'] = w_sbert
        MODEL_WEIGHTS['ai'] = w_ai
        flash('Model weights updated successfully!')
    except ValueError:
        flash('Invalid weight values. Please enter numbers.')
    return redirect(url_for('faculty_dashboard'))

@app.route('/api/datasets')
def api_datasets():
    if 'user' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    result = {}
    for subfolder in ['tfidf', 'sbert', 'ai_detection']:
        folder_path = os.path.join(datasets_dir, subfolder)
        files = []
        if os.path.isdir(folder_path):
            for fname in sorted(os.listdir(folder_path)):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath):
                    size_bytes = os.path.getsize(fpath)
                    if size_bytes > 1024 * 1024:
                        size_str = f"{size_bytes / (1024*1024):.1f} MB"
                    elif size_bytes > 1024:
                        size_str = f"{size_bytes / 1024:.1f} KB"
                    else:
                        size_str = f"{size_bytes} B"
                    files.append({'name': fname, 'size': size_str})
        result[subfolder] = files
    return jsonify(result)

def _run_import(query, count):
    """Background worker for arXiv import with live logging."""
    global IMPORT_RUNNING
    import time
    IMPORT_RUNNING = True
    IMPORT_LOG.clear()

    try:
        from import_arxiv import search_arxiv, get_next_paper_number, sanitize_filename, TFIDF_DIR, TEMP_PDF_DIR, REFERENCE_DIR
        from pdf_reader.pdfextraction.reader import extract_pdf_text
        from pdf_reader.pdfextraction.preprocessing import clean_extracted_text
        import urllib.request
        import shutil

        IMPORT_LOG.append(f'🔍 Searching arXiv for "{query}" (max {count})...')
        papers = search_arxiv(query, count)

        if not papers:
            IMPORT_LOG.append('❌ No papers found. Try a different query.')
            IMPORT_RUNNING = False
            return

        IMPORT_LOG.append(f'📋 Found {len(papers)} papers with PDF links.\n')
        start_num = get_next_paper_number()
        IMPORT_LOG.append(f'📁 Next paper number: paper_{start_num:03d}\n')

        os.makedirs(TEMP_PDF_DIR, exist_ok=True)
        os.makedirs(TFIDF_DIR, exist_ok=True)
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        success = 0

        for i, (title, pdf_url) in enumerate(papers):
            paper_num = start_num + i
            pdf_filename = f'paper_{paper_num:03d}.pdf'
            txt_filename = f'paper_{paper_num:03d}.txt'
            pdf_path = os.path.join(TEMP_PDF_DIR, pdf_filename)
            txt_path = os.path.join(TFIDF_DIR, txt_filename)

            short_title = title[:60] + ('...' if len(title) > 60 else '')
            IMPORT_LOG.append(f'[{i+1}/{len(papers)}] {short_title}')

            # Download
            try:
                IMPORT_LOG.append('  📥 Downloading PDF...')
                urllib.request.urlretrieve(pdf_url, pdf_path)
            except Exception as e:
                IMPORT_LOG.append(f'  ❌ Download failed: {e}')
                time.sleep(3)
                continue

            # Reference copy
            safe_title = sanitize_filename(title)
            ref_name = f'paper_{paper_num:03d}_{safe_title}.pdf'
            ref_path = os.path.join(REFERENCE_DIR, ref_name)
            try:
                shutil.copy2(pdf_path, ref_path)
                IMPORT_LOG.append(f'  📂 Reference PDF: {ref_name}')
            except Exception:
                pass

            # Extract
            try:
                IMPORT_LOG.append('  📄 Extracting text...')
                full_text = extract_pdf_text(pdf_path)
                full_text = clean_extracted_text(full_text)

                if len(full_text) < 200:
                    IMPORT_LOG.append(f'  ⚠️ Too little text ({len(full_text)} chars), skipping.')
                    time.sleep(3)
                    continue

                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)

                success += 1
                IMPORT_LOG.append(f'  ✅ Saved {txt_filename} ({len(full_text):,} chars)')
            except Exception as e:
                IMPORT_LOG.append(f'  ❌ Extraction failed: {e}')

            if i < len(papers) - 1:
                IMPORT_LOG.append('  ⏳ Waiting 3s (arXiv rate limit)...')
                time.sleep(3)

        IMPORT_LOG.append(f'\n✅ Imported {success}/{len(papers)} papers.')

        # Rebuild SBERT dataset
        if success > 0:
            IMPORT_LOG.append('\n🔄 Rebuilding SBERT dataset...')
            try:
                import pandas as pd
                from nltk.tokenize import sent_tokenize
                all_sentences = []
                for fname in sorted(os.listdir(TFIDF_DIR)):
                    if fname.endswith('.txt'):
                        fpath = os.path.join(TFIDF_DIR, fname)
                        with open(fpath, 'r', encoding='utf-8') as f:
                            text = f.read()
                        sents = sent_tokenize(text)
                        all_sentences.extend(sents)
                        IMPORT_LOG.append(f'  📄 {fname}: {len(sents)} sentences')

                df = pd.DataFrame({'sentence1': all_sentences, 'sentence2': [''] * len(all_sentences)})
                df = df[df['sentence1'].str.len() > 10]
                sbert_path = os.path.join(os.path.dirname(TFIDF_DIR), 'sbert', 'sbert_pairs.csv')
                os.makedirs(os.path.dirname(sbert_path), exist_ok=True)
                df.to_csv(sbert_path, index=False)
                IMPORT_LOG.append(f'  ✅ SBERT dataset rebuilt: {len(df)} sentences saved.')
            except Exception as e:
                IMPORT_LOG.append(f'  ❌ SBERT rebuild failed: {e}')

        IMPORT_LOG.append('\n✅ All done!')
    except Exception as e:
        IMPORT_LOG.append(f'\n❌ Import error: {e}')
    finally:
        IMPORT_RUNNING = False

@app.route('/import_arxiv', methods=['POST'])
def import_arxiv_route():
    global IMPORT_RUNNING
    if 'user' not in session or session.get('role') != 'faculty':
        return jsonify({'error': 'unauthorized'}), 401

    if IMPORT_RUNNING:
        return jsonify({'error': 'Import already in progress'}), 409

    query = request.form.get('query', 'machine learning').strip()
    count = int(request.form.get('count', 5))
    count = min(count, 25)

    thread = threading.Thread(target=_run_import, args=(query, count), daemon=True)
    thread.start()
    return jsonify({'started': True})

@app.route('/import_arxiv_status')
def import_arxiv_status():
    if 'user' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    return jsonify({
        'running': IMPORT_RUNNING,
        'log': list(IMPORT_LOG)
    })

@app.route('/api/reference_papers')
def api_reference_papers():
    if 'user' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    ref_dir = os.path.join(os.path.dirname(__file__), 'reference_papers')
    files = []
    if os.path.isdir(ref_dir):
        for fname in sorted(os.listdir(ref_dir)):
            if fname.lower().endswith('.pdf'):
                fpath = os.path.join(ref_dir, fname)
                size_bytes = os.path.getsize(fpath)
                if size_bytes >= 1048576:
                    size_str = f"{size_bytes / 1048576:.1f} MB"
                elif size_bytes >= 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes} B"
                # Extract readable title from filename: paper_006_Some_Title.pdf -> Some Title
                name_part = os.path.splitext(fname)[0]
                parts = name_part.split('_', 2)
                title = parts[2].replace('_', ' ') if len(parts) > 2 else name_part.replace('_', ' ')
                files.append({'filename': fname, 'title': title, 'size': size_str})
    return jsonify(files)

@app.route('/download_reference/<filename>')
def download_reference(filename):
    if 'user' not in session:
        return redirect(url_for('login'))
    safe_name = os.path.basename(filename)
    ref_dir = os.path.join(os.path.dirname(__file__), 'reference_papers')
    filepath = os.path.join(ref_dir, safe_name)
    if os.path.isfile(filepath):
        return send_file(filepath, as_attachment=True)
    flash('File not found.')
    return redirect(url_for('faculty_dashboard'))

@app.route('/manual_insert', methods=['POST'])
def manual_insert():
    if 'user' not in session or session.get('role') != 'faculty':
        return redirect(url_for('login'))

    target = request.form.get('target', 'tfidf')
    if target not in ('tfidf', 'sbert', 'ai_detection'):
        flash('Invalid target dataset.')
        return redirect(url_for('faculty_dashboard'))

    if 'dataset_file' not in request.files:
        flash('No file uploaded.')
        return redirect(url_for('faculty_dashboard'))

    file = request.files['dataset_file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('faculty_dashboard'))

    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', target)
    os.makedirs(datasets_dir, exist_ok=True)
    fname = secure_filename(file.filename)
    dest = os.path.join(datasets_dir, fname)
    file.save(dest)
    flash(f'File "{fname}" added to {target} dataset.')
    return redirect(url_for('faculty_dashboard'))

@app.route('/delete_dataset_file', methods=['POST'])
def delete_dataset_file():
    if 'user' not in session or session.get('role') != 'faculty':
        return jsonify({'error': 'unauthorized'}), 401

    data = request.get_json()
    folder = data.get('folder', '')
    filename = data.get('filename', '')

    if folder not in ('tfidf', 'sbert', 'ai_detection'):
        return jsonify({'error': 'invalid folder'}), 400

    # Prevent path traversal
    safe_name = os.path.basename(filename)
    filepath = os.path.join(os.path.dirname(__file__), 'datasets', folder, safe_name)

    if os.path.isfile(filepath):
        os.remove(filepath)
        return jsonify({'success': True, 'message': f'Deleted {safe_name}'})
    else:
        return jsonify({'error': 'File not found'}), 404

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
        results = analyze_pdf(filepath, weights=MODEL_WEIGHTS)
        
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
            'originality_score': results.get('originality_score', 0.0),
            'weights': dict(MODEL_WEIGHTS)
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
