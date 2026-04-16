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

# Paper categories configuration
PAPER_CATEGORIES = [
    'AI/ML',
    'Computer Vision',
    'Medical & Healthcare',
    'Mathematics & Statistics',
    'Computer Engineering',
    'Electrical Engineering',
    'Mechanical Engineering',
    'Uncategorized'
]

# Category keywords for auto-categorization
CATEGORY_KEYWORDS = {
    'AI/ML': ['deep learning', 'neural network', 'machine learning', 'artificial intelligence', 
              'deep neural', 'cnn', 'rnn', 'lstm', 'transformer', 'reinforcement learning',
              'supervised', 'unsupervised', 'classification', 'regression', 'training'],
    'Computer Vision': ['computer vision', 'image', 'visual', 'vision', 'object detection',
                        'segmentation', 'opencv', 'convolutional', 'yolo', 'detection'],
    'Medical & Healthcare': ['medical', 'health', 'cancer', 'breast', 'disease', 'diagnosis',
                             'patient', 'clinical', 'mammography', 'tumor', 'screening',
                             'biomedical', 'healthcare', 'therapeutic', 'injury', 'ct'],
    'Mathematics & Statistics': ['random', 'graph', 'matrix', 'theorem', 'proof', 'probability',
                                 'statistics', 'mathematical', 'algebra', 'geometry', 'topology',
                                 'polynomial', 'chaos', 'spectrum', 'fluctuation'],
    'Computer Engineering': ['computing', 'algorithm', 'data structure', 'programming',
                            'software', 'hardware', 'processor', 'memory', 'architecture'],
    'Electrical Engineering': ['circuit', 'signal', 'power', 'semiconductor', 'device modeling',
                              'electronic', 'voltage', 'current', 'transistor', 'analog'],
    'Mechanical Engineering': ['mechanical', 'thermal', 'fluid', 'dynamics', 'robotics',
                              'manufacturing', 'materials', 'mechanics']
}

def get_categories_file():
    """Get path to paper categories JSON file"""
    ref_dir = os.path.join(os.path.dirname(__file__), 'reference_papers')
    return os.path.join(ref_dir, 'paper_categories.json')

def load_paper_categories():
    """Load paper category mappings from JSON file"""
    categories_file = get_categories_file()
    if os.path.exists(categories_file):
        try:
            with open(categories_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_paper_categories(categories_dict):
    """Save paper category mappings to JSON file"""
    categories_file = get_categories_file()
    os.makedirs(os.path.dirname(categories_file), exist_ok=True)
    with open(categories_file, 'w', encoding='utf-8') as f:
        json.dump(categories_dict, f, indent=2, ensure_ascii=False)

def auto_categorize_paper(title, filename):
    """Automatically categorize a paper based on its title and filename"""
    text = f"{title} {filename}".lower()
    
    # Check each category's keywords
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > 0:
            scores[category] = score
    
    # Return category with highest score, or 'Uncategorized'
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    return 'Uncategorized'

def get_categorized_papers():
    """Get all reference papers organized by category"""
    ref_dir = os.path.join(os.path.dirname(__file__), 'reference_papers')
    categories_map = load_paper_categories()
    papers_by_category = {cat: [] for cat in PAPER_CATEGORIES}
    
    if os.path.isdir(ref_dir):
        needs_save = False
        for fname in sorted(os.listdir(ref_dir)):
            if fname.lower().endswith('.pdf'):
                fpath = os.path.join(ref_dir, fname)
                size_bytes = os.path.getsize(fpath)
                
                # Format size
                if size_bytes >= 1048576:
                    size_str = f"{size_bytes / 1048576:.1f} MB"
                elif size_bytes >= 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes} B"
                
                # Extract title
                name_part = os.path.splitext(fname)[0]
                parts = name_part.split('_', 2)
                title = parts[2].replace('_', ' ') if len(parts) > 2 else name_part.replace('_', ' ')
                
                # Get or assign category
                if fname not in categories_map:
                    categories_map[fname] = auto_categorize_paper(title, fname)
                    needs_save = True
                
                category = categories_map[fname]
                if category not in papers_by_category:
                    category = 'Uncategorized'
                
                papers_by_category[category].append({
                    'filename': fname,
                    'title': title,
                    'size': size_str,
                    'category': category
                })
        
        # Save if any new papers were categorized
        if needs_save:
            save_paper_categories(categories_map)
    
    # Remove empty categories
    return {cat: papers for cat, papers in papers_by_category.items() if papers}

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
    return jsonify(get_categorized_papers())

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

@app.route('/api/paper_categories')
def api_paper_categories():
    if 'user' not in session:
        return jsonify({'error': 'unauthorized'}), 401
    categories_map = load_paper_categories()
    return jsonify({
        'categories': categories_map,
        'available_categories': PAPER_CATEGORIES
    })

@app.route('/api/save_paper_categories', methods=['POST'])
def api_save_paper_categories():
    if 'user' not in session or session.get('role') != 'faculty':
        return jsonify({'error': 'unauthorized'}), 401
    try:
        data = request.get_json()
        categories_map = data.get('categories', {})
        save_paper_categories(categories_map)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/auto_categorize_papers', methods=['POST'])
def api_auto_categorize_papers():
    if 'user' not in session or session.get('role') != 'faculty':
        return jsonify({'error': 'unauthorized'}), 401
    try:
        ref_dir = os.path.join(os.path.dirname(__file__), 'reference_papers')
        categories_map = {}
        
        if os.path.isdir(ref_dir):
            for fname in os.listdir(ref_dir):
                if fname.lower().endswith('.pdf'):
                    # Extract title
                    name_part = os.path.splitext(fname)[0]
                    parts = name_part.split('_', 2)
                    title = parts[2].replace('_', ' ') if len(parts) > 2 else name_part.replace('_', ' ')
                    
                    # Auto categorize
                    categories_map[fname] = auto_categorize_paper(title, fname)
        
        return jsonify({'success': True, 'categories': categories_map})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
    return render_template('profile.html', user=session['user'], role=session.get('role', 'student'), stats=stats, uploads=UPLOAD_HISTORY)

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

    from fpdf import FPDF
    import math

    # Colour palette
    NAVY       = (20, 22, 48)
    DARK_NAVY  = (12, 14, 36)
    ACCENT     = (99, 102, 241)
    GREEN      = (16, 185, 129)
    RED        = (239, 68, 68)
    ORANGE     = (245, 158, 11)
    BLUE       = (59, 130, 246)
    LIGHT_BG   = (248, 249, 252)
    CARD_BG    = (241, 243, 249)
    TEXT_DARK  = (30, 30, 50)
    TEXT_MID   = (90, 95, 110)
    TEXT_LIGHT = (160, 165, 180)
    WHITE      = (255, 255, 255)

    class ReportPDF(FPDF):
        def header(self):
            h = 44
            for i in range(h):
                r = int(NAVY[0] + (DARK_NAVY[0] - NAVY[0]) * i / h)
                g = int(NAVY[1] + (DARK_NAVY[1] - NAVY[1]) * i / h)
                b = int(NAVY[2] + (DARK_NAVY[2] - NAVY[2]) * i / h)
                self.set_fill_color(r, g, b)
                self.rect(0, i, 210, 1, 'F')
            self.set_fill_color(*ACCENT)
            self.rect(0, h, 210, 2, 'F')
            self.set_y(10)
            self.set_font('CustomFont', 'B', 22)
            self.set_text_color(*WHITE)
            self.cell(0, 10, 'CHECKMATE', align='C', new_x="LMARGIN", new_y="NEXT")
            self.set_font('CustomFont', '', 9)
            self.set_text_color(180, 185, 220)
            self.cell(0, 5, 'SMART PLAGIARISM & AI DETECTION REPORT', align='C', new_x="LMARGIN", new_y="NEXT")
            self.set_y(h + 8)

        def footer(self):
            self.set_y(-20)
            self.set_draw_color(*ACCENT)
            self.set_line_width(0.4)
            self.line(15, self.get_y(), 195, self.get_y())
            self.ln(3)
            self.set_font('CustomFont', '', 7)
            self.set_text_color(*TEXT_LIGHT)
            self.cell(0, 5, f'CheckMate  |  Plagiarism & AI Detection  |  Page {self.page_no()}/{{nb}}', align='C')

        def draw_progress_bar(self, x, y, w, h, pct, bar_color, bg_color=(230,232,240)):
            self.set_fill_color(*bg_color)
            self.rect(x, y, w, h, 'F')
            fill_w = max(w * pct, 0)
            if fill_w > 0:
                self.set_fill_color(*bar_color)
                self.rect(x, y, fill_w, h, 'F')

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.alias_nb_pages()

    font_dir = r'C:\Windows\Fonts'
    pdf.add_font('CustomFont', '', os.path.join(font_dir, 'arial.ttf'), uni=True)
    pdf.add_font('CustomFont', 'B', os.path.join(font_dir, 'arialbd.ttf'), uni=True)
    pdf.add_font('CustomFont', 'I', os.path.join(font_dir, 'ariali.ttf'), uni=True)
    pdf.add_font('CustomFont', 'BI', os.path.join(font_dir, 'arialbi.ttf'), uni=True)

    pdf.add_page()

    # ── SECTION 1: File Info Card ──
    card_y = pdf.get_y()
    pdf.set_fill_color(*CARD_BG)
    pdf.rect(12, card_y, 186, 22, 'F')
    pdf.set_fill_color(*ACCENT)
    pdf.rect(12, card_y, 2.5, 22, 'F')

    pdf.set_xy(18, card_y + 3)
    pdf.set_font('CustomFont', 'B', 9)
    pdf.set_text_color(*TEXT_MID)
    pdf.cell(18, 5, 'FILE', new_x="END")
    pdf.set_font('CustomFont', '', 9)
    pdf.set_text_color(*TEXT_DARK)
    pdf.cell(90, 5, filename, new_x="END")
    pdf.set_font('CustomFont', 'B', 9)
    pdf.set_text_color(*TEXT_MID)
    pdf.cell(18, 5, 'DATE', new_x="END")
    pdf.set_font('CustomFont', '', 9)
    pdf.set_text_color(*TEXT_DARK)
    pdf.cell(0, 5, datetime.now().strftime('%B %d, %Y  %H:%M'), new_x="LMARGIN", new_y="NEXT")

    pdf.set_xy(18, card_y + 12)
    pdf.set_font('CustomFont', 'B', 9)
    pdf.set_text_color(*TEXT_MID)
    pdf.cell(18, 5, 'USER', new_x="END")
    pdf.set_font('CustomFont', '', 9)
    pdf.set_text_color(*TEXT_DARK)
    pdf.cell(0, 5, session.get('user', 'N/A'), new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(card_y + 28)

    # ── SECTION 2: Score Cards with Progress Bars ──
    pdf.set_font('CustomFont', 'B', 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, '   Algorithm Score Summary', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    scores_data = [
        ('Originality Score',       results['originality_score'], GREEN,  'Weighted combination of all detection models'),
        ('Copied Content (TF-IDF)', results['tfidf_score'],       RED,    'Direct text matching against reference corpus'),
        ('Paraphrased (SBERT)',     results['sbert_score'],        ORANGE, 'Semantic similarity via sentence embeddings'),
        ('AI Generated',            results['ai_prob'],            BLUE,   'ML classifier for AI-generated text'),
    ]

    card_w = 186
    bar_w = 80

    for label, value, color, desc in scores_data:
        cy = pdf.get_y()
        card_h = 18

        pdf.set_fill_color(*LIGHT_BG)
        pdf.rect(12, cy, card_w, card_h, 'F')
        pdf.set_fill_color(*color)
        pdf.rect(12, cy, 3, card_h, 'F')

        pdf.set_xy(18, cy + 2)
        pdf.set_font('CustomFont', 'B', 10)
        pdf.set_text_color(*TEXT_DARK)
        pdf.cell(80, 6, label, new_x="END")

        bar_x = 108
        bar_y = cy + 5
        pdf.draw_progress_bar(bar_x, bar_y, bar_w, 6, value, color)

        pdf.set_xy(bar_x + bar_w + 3, cy + 2)
        pdf.set_font('CustomFont', 'B', 12)
        pdf.set_text_color(*color)
        pdf.cell(15, 6, f'{value*100:.0f}%', new_x="LMARGIN", new_y="NEXT")

        pdf.set_xy(18, cy + 10)
        pdf.set_font('CustomFont', '', 7)
        pdf.set_text_color(*TEXT_LIGHT)
        pdf.cell(0, 5, desc, new_x="LMARGIN", new_y="NEXT")

        pdf.set_y(cy + card_h + 3)

    pdf.ln(3)

    # ── SECTION 3: Verdict Banner ──
    verdict = "APPROVED" if results['originality_score'] > 0.6 else "NEEDS REVISION"
    v_color = GREEN if verdict == "APPROVED" else RED
    v_icon = 'PASS' if verdict == "APPROVED" else 'FAIL'
    v_msg = "This paper meets originality standards." if verdict == "APPROVED" else "This paper requires revision for originality."

    vy = pdf.get_y()
    pdf.set_fill_color(*v_color)
    pdf.rect(12, vy, card_w, 20, 'F')

    pdf.set_xy(15, vy + 2)
    pdf.set_font('CustomFont', 'B', 16)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 9, f'   [{v_icon}]   VERDICT: {verdict}', new_x="LMARGIN", new_y="NEXT")

    pdf.set_xy(28, vy + 12)
    pdf.set_font('CustomFont', '', 8)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 5, v_msg, new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(vy + 28)

    # ── SECTION 4: Detailed Content Analysis ──
    pdf.set_font('CustomFont', 'B', 12)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 8, '   Detailed Content Analysis', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # Legend bar
    ly = pdf.get_y()
    pdf.set_fill_color(*LIGHT_BG)
    pdf.rect(12, ly, card_w, 8, 'F')
    pdf.set_xy(16, ly + 1.5)
    pdf.set_font('CustomFont', 'B', 7)
    legend_items = [
        ('Copied', RED),
        ('Paraphrased', ORANGE),
        ('AI Generated', BLUE),
        ('Original', GREEN),
    ]
    for lbl, clr in legend_items:
        pdf.set_fill_color(*clr)
        pdf.rect(pdf.get_x(), ly + 2.5, 3, 3, 'F')
        pdf.set_x(pdf.get_x() + 4)
        pdf.set_text_color(*clr)
        pdf.cell(28, 5, lbl, new_x="END")

    pdf.set_y(ly + 11)

    if results.get('content_analysis'):
        for idx, item in enumerate(results['content_analysis']):
            sentence = item['text'].strip()
            labels = item.get('labels', [])

            if 'copied' in labels:
                strip_color = RED
                tag = 'COPIED'
            elif 'paraphrased' in labels:
                strip_color = ORANGE
                tag = 'PARAPHRASED'
            elif 'ai' in labels:
                strip_color = BLUE
                tag = 'AI GENERATED'
            else:
                strip_color = (200, 210, 220)
                tag = 'ORIGINAL'

            pdf.set_font('CustomFont', '', 8)
            text_w = card_w - 10
            line_height = 4.5
            num_lines = max(1, math.ceil(pdf.get_string_width(sentence) / text_w))
            block_h = max(num_lines * line_height + 6, 12)

            sy = pdf.get_y()
            if sy + block_h > 275:
                pdf.add_page()
                sy = pdf.get_y()

            # Alternating row bg
            row_bg = LIGHT_BG if idx % 2 == 0 else WHITE
            pdf.set_fill_color(*row_bg)
            pdf.rect(12, sy, card_w, block_h, 'F')

            # Color strip
            pdf.set_fill_color(*strip_color)
            pdf.rect(12, sy, 2.5, block_h, 'F')

            # Tag
            pdf.set_xy(17, sy + 1)
            pdf.set_font('CustomFont', 'B', 6.5)
            pdf.set_text_color(*strip_color)
            pdf.cell(25, 3.5, tag, new_x="LMARGIN", new_y="NEXT")

            # Sentence
            pdf.set_xy(17, sy + 5.5)
            pdf.set_font('CustomFont', '', 8)
            pdf.set_text_color(*TEXT_DARK)
            pdf.multi_cell(text_w, line_height, sentence, new_x="LMARGIN", new_y="NEXT")

            pdf.set_y(sy + block_h + 1)
    else:
        pdf.set_font('CustomFont', 'I', 10)
        pdf.set_text_color(*TEXT_LIGHT)
        pdf.cell(0, 8, 'No detailed analysis available.', new_x="LMARGIN", new_y="NEXT")

    # Output
    buffer = io.BytesIO()
    buffer.write(pdf.output())
    buffer.seek(0)

    report_filename = os.path.splitext(filename)[0]
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"report_{report_filename}.pdf",
        mimetype='application/pdf'
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

@app.route('/save_overleaf_url', methods=['POST'])
def save_overleaf_url():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'unauthorized'}), 401
    try:
        data = request.get_json()
        url = data.get('url', '')
        session['overleaf_url'] = url
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
