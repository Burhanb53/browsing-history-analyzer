from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd
from urllib.parse import urlparse
import os
from werkzeug.utils import secure_filename
from flask_session import Session
import google.generativeai as genai
from dotenv import load_dotenv
from markdown import markdown
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'json'}
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize Flask-Session
Session(app)

# Gemini AI Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Use the latest model name
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')  # Updated to latest model
        GEMINI_ENABLED = True
    except Exception as e:
        print(f"Gemini model initialization error: {str(e)}")
        GEMINI_ENABLED = False
else:
    print("Warning: Gemini API key not found. AI features will be disabled.")
    GEMINI_ENABLED = False

# Custom JSON encoder to handle numpy and pandas types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, pd.Int64Dtype)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Template filter for markdown
@app.template_filter('markdown')
def markdown_filter(text):
    return markdown(text) if text else ""

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_gemini_analysis(history_data):
    """Generate AI analysis of browsing history using Gemini"""
    if not GEMINI_ENABLED:
        return None

    try:
        df = pd.DataFrame(history_data)

        # Sample the data to reduce size (e.g., 10% or max 1000 rows)
        sample_size = min(1000, len(df) // 10)
        if len(df) > 1000:
            df = df.sample(sample_size, random_state=42)  # random_state for reproducibility

        top_domains = df['domain'].value_counts().head(5).index.tolist()
        time_range = f"{df['visit_time'].min()} to {df['visit_time'].max()}"
        
        
        prompt = f"""
You are an expert digital behavior analyst. Analyze the following user's browsing history and provide deep, insightful, and actionable observations.

### Metadata:
- **Total Visits**: {len(history_data)}
- **Time Period**: {time_range}
- **Top Domains Visited**: {', '.join(top_domains)}

### Analysis Objectives:
Please analyze the browsing history with respect to the following dimensions:

1. **Main Interests & Content Themes**  
   Identify the most common topics or types of content the user is consuming. Mention any specific websites or categories they frequently visit.

2. **Productivity & Time Management**  
   Evaluate how productive the browsing behavior appears. Identify potential time-wasting patterns or efficient habits. Highlight visits related to learning, work, or self-improvement.

3. **Daily & Weekly Patterns**  
   Analyze usage trends based on time of day and days of the week. Highlight any routines, spikes in activity, or unusual behavior patterns.

4. **Privacy & Security Observations**  
   Identify if the user is visiting potentially unsafe websites, using incognito mode, or sites that collect large amounts of personal data. Flag any privacy concerns.

5. **Personalized Recommendations**  
   Based on your analysis, provide personalized suggestions to help improve productivity, protect privacy, or enhance the user’s online experience. Recommend tools, habits, or strategies.

### Output Format:
- Use **Markdown** with clear headings and subheadings
- Keep language clear, concise, and practical
- Provide **bullet points** or **numbered lists** where helpful
- End with a brief **summary of key takeaways**

Here is the data you should analyze:
{df.to_csv(index=False)}
"""

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini analysis error: {str(e)}")
        return "Could not generate AI analysis"

def print_analysis_report(data):
    """Print detailed analysis report to console"""
    print("\n" + "="*50)
    print("BROWSING HISTORY ANALYSIS REPORT")
    print("="*50)
    
    # Basic stats
    print(f"\nTotal visits: {len(data)}")
    print(f"Time period: {data[0]['visit_time']} to {data[-1]['visit_time']}")
    
    # Top sites
    df = pd.DataFrame(data)
    top_sites = df['domain'].value_counts().head(5)
    print("\nTop 5 Sites:")
    print(top_sites.to_string())
    
    # Activity by hour
    print("\nActivity by Hour:")
    hourly = df['hour'].value_counts().sort_index()
    print(hourly.to_string())
    
    # Activity by day
    print("\nActivity by Day:")
    daily = df['day_of_week'].value_counts()
    print(daily.to_string())
    
    # Categories
    categories = defaultdict(int)
    for domain in df['domain']:
        domain_lower = domain.lower()
        if any(s in domain_lower for s in ['google', 'bing', 'yahoo']):
            categories['Search'] += 1
        elif any(s in domain_lower for s in ['facebook', 'twitter', 'instagram']):
            categories['Social'] += 1
        elif any(s in domain_lower for s in ['youtube', 'netflix']):
            categories['Media'] += 1
        else:
            categories['Other'] += 1
    
    print("\nContent Categories:")
    for cat, count in categories.items():
        print(f"{cat}: {count} ({count/len(data)*100:.1f}%)")
    
    print("\nSample Processed Entries:")
    for i in range(min(3, len(data))):
        print(f"\nEntry {i+1}:")
        print(f"Title: {data[i]['title']}")
        print(f"Domain: {data[i]['domain']}")
        print(f"Time: {data[i]['visit_time']}")
        print(f"URL: {data[i]['url'][:50]}...")
    
    if GEMINI_ENABLED:
        print("\nAI Analysis Preview:")
        ai_analysis = generate_gemini_analysis(data)
        if ai_analysis:
            print(ai_analysis[:500] + "...")  # Print first 500 chars of analysis
    
    print("\n" + "="*50 + "\n")

def load_history_data(filepath):
    """Load and validate browser history data from specified file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("History data should be a list of visits")
        if len(data) == 0:
            raise ValueError("History data is empty")
        
        return data
    except Exception as e:
        print(f"Error loading history data: {str(e)}")
        return []

def process_history_data(data):
    """Process raw history data into structured format"""
    processed = []
    
    for entry in data:
        try:
            # Convert timestamps
            visit_time = datetime.fromtimestamp(float(entry['visitTime'])/1000)
            last_visit_time = datetime.fromtimestamp(float(entry['lastVisitTime'])/1000)
            
            # Extract domain
            url = entry['url']
            domain = urlparse(url).netloc if urlparse(url).netloc else url.split('/')[2] if len(url.split('/')) > 2 else url
            
            processed.append({
                'id': entry['id'],
                'title': entry['title'],
                'url': url,
                'domain': domain,
                'visit_time': visit_time,
                'last_visit_time': last_visit_time,
                'visit_count': entry['visitCount'],
                'hour': visit_time.hour,
                'day_of_week': visit_time.strftime('%A'),
                'date': visit_time.date()
            })
        except Exception as e:
            print(f"Error processing entry {entry.get('id')}: {str(e)}")
            continue
    
    return processed

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Store the filename in session for analysis
            session['current_file'] = filepath
            return redirect(url_for('analyze'))
    
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    # Pass the filepath to the template if it exists
    filepath = session.get('current_file', None)
    if filepath:
        # Load and process the data to pass to template
        raw_data = load_history_data(filepath)
        processed_data = process_history_data(raw_data)
        df = pd.DataFrame(processed_data)
        
        return render_template('analyze.html', 
                             filepath=filepath,
                             data={
                                 'top_sites': df['domain'].value_counts().head(10).to_dict(),
                                 'total_visits': len(df),
                                 'first_visit': df['visit_time'].min().strftime('%Y-%m-%d %H:%M:%S'),
                                 'last_visit': df['visit_time'].max().strftime('%Y-%m-%d %H:%M:%S'),
                                 'ai_analysis': generate_gemini_analysis(processed_data)
                             })
    else:
        return redirect(url_for('index'))

@app.route('/api/analytics')
def get_analytics():
    filepath = request.args.get('filepath') or session.get('current_file')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({
            'status': 'error',
            'message': 'No history file available. Please upload one first.'
        })
    
    raw_data = load_history_data(filepath)
    processed_data = process_history_data(raw_data)
    
    # Print analysis to console
    print_analysis_report(processed_data)
    
    # Generate analytics for frontend
    df = pd.DataFrame(processed_data)
    top_sites = df['domain'].value_counts().head(10).to_dict()
    activity_by_hour = {hour: 0 for hour in range(24)}
    hour_counts = df['hour'].value_counts()
    for hour, count in hour_counts.items():
        activity_by_hour[hour] = count
    
    # Prepare daily visits data
    daily_visits = df.groupby('date').size()
    daily_visits_dict = {str(date): count for date, count in daily_visits.items()}
    
    return jsonify({
        'status': 'success',
        'data': {
            'top_sites': top_sites,
            'activity_by_hour': {int(k): int(v) for k, v in activity_by_hour.items()},
            'activity_by_day': df['day_of_week'].value_counts().to_dict(),
            'categories': {
                'Search': int(len(df[df['domain'].str.contains('google|bing|yahoo|duckduckgo|ask.com', case=False)])),
                'Social': int(len(df[df['domain'].str.contains('facebook|twitter|instagram|linkedin|snapchat|tiktok|pinterest|reddit', case=False)])),
                'Media': int(len(df[df['domain'].str.contains('youtube|netflix|hulu|vimeo|dailymotion|spotify|soundcloud|primevideo|hotstar', case=False)])),
                'Shopping': int(len(df[df['domain'].str.contains('amazon|flipkart|ebay|aliexpress|walmart|bestbuy|target|myntra|shopify', case=False)])),
                'Education': int(len(df[df['domain'].str.contains('wikipedia|khanacademy|coursera|edx|udemy|udacity|academia|nptel', case=False)])),
                'News': int(len(df[df['domain'].str.contains('cnn|bbc|nytimes|theguardian|reuters|ndtv|timesofindia|hindustantimes|aljazeera', case=False)])),
                'Entertainment': int(len(df[df['domain'].str.contains('imdb|rottentomatoes|fandom|buzzfeed', case=False)])),
                'Technology': int(len(df[df['domain'].str.contains('github|stackoverflow|gitlab|geeksforgeeks|w3schools|tutorialspoint|hackerrank', case=False)])),
                'Finance': int(len(df[df['domain'].str.contains('paypal|stripe|moneycontrol|bankofamerica|icicibank|hdfcbank|tradingview', case=False)])),
                'Travel': int(len(df[df['domain'].str.contains('tripadvisor|makemytrip|booking|airbnb|expedia|goibibo|trivago|agoda', case=False)])),
                'Other': int(len(df) - len(df[df['domain'].str.contains(
                    'google|bing|yahoo|duckduckgo|ask.com|'
                    'facebook|twitter|instagram|linkedin|snapchat|tiktok|pinterest|reddit|'
                    'youtube|netflix|hulu|vimeo|dailymotion|spotify|soundcloud|primevideo|hotstar|'
                    'amazon|flipkart|ebay|aliexpress|walmart|bestbuy|target|myntra|shopify|'
                    'wikipedia|khanacademy|coursera|edx|udemy|udacity|academia|nptel|'
                    'cnn|bbc|nytimes|theguardian|reuters|ndtv|timesofindia|hindustantimes|aljazeera|'
                    'imdb|rottentomatoes|fandom|buzzfeed|'
                    'github|stackoverflow|gitlab|geeksforgeeks|w3schools|tutorialspoint|hackerrank|'
                    'paypal|stripe|moneycontrol|bankofamerica|icicibank|hdfcbank|tradingview|'
                    'tripadvisor|makemytrip|booking|airbnb|expedia|goibibo|trivago|agoda',
                    case=False
                )]))
            },
            'total_visits': int(len(df)),
            'unique_domains': int(df['domain'].nunique()),
            'first_visit': df['visit_time'].min().strftime('%Y-%m-%d %H:%M:%S'),
            'last_visit': df['visit_time'].max().strftime('%Y-%m-%d %H:%M:%S'),
            'daily_visits': {k: int(v) for k, v in daily_visits_dict.items()}
        }
    })

@app.route('/api/ai_analysis')
def get_ai_analysis():
    filepath = request.args.get('filepath') or session.get('current_file')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({
            'status': 'error',
            'message': 'No history file available. Please upload one first.'
        })
    
    raw_data = load_history_data(filepath)
    processed_data = process_history_data(raw_data)
    
    return jsonify({
        'status': 'success',
        'data': {
            'ai_analysis': generate_gemini_analysis(processed_data)
        }
    })
@app.route('/api/raw-data')
def get_raw_data():
    filepath = request.args.get('filepath') or session.get('current_file')
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({
            'status': 'error',
            'message': 'No history file available. Please upload one first.'
        })
    
    raw_data = load_history_data(filepath)
    processed_data = process_history_data(raw_data)
    
    # Return only the first 100 entries for preview
    preview_data = processed_data
    
    raw_data_response = [{
        'title': entry['title'],
        'url': entry['url'],
        'domain': entry['domain'],
        'visit_time': entry['visit_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'visit_count': int(entry['visit_count'])
    } for entry in preview_data]
    
    return jsonify({
        'status': 'success',
        'data': raw_data_response
    })
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
