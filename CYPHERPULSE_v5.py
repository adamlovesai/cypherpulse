#!/usr/bin/env python3
"""
CypherPulse - Sentiment Analysis Tool
For best console output, run with: python -u CYPHERPULSE.py
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import csv
import os
import sys

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
    print("✅ matplotlib loaded successfully", flush=True)
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  WARNING: matplotlib not installed. Charts will not be generated.", flush=True)
    print("   To enable charts, run: pip install matplotlib", flush=True)

# ==================== CONFIGURATION ====================
# OPTION 1: NewsAPI.org (Free tier - 100 requests/day, no full content in free tier)
NEWSAPI_KEY = "44a8f55bbfdb49bda90699dfd2534cf3"  # Get free key at https://newsapi.org/

# OPTION 2: GNews.io (Paid tier needed for full content - ~$50/month)
# GNEWS_API_KEY = "YOUR_GNEWS_API_KEY_HERE"  # Get at https://gnews.io/

USE_GNEWS = False  # Set to True if using GNews.io paid plan

# Article Limits:
# - NewsAPI.org free tier: Can fetch many articles, but needs scraping (slower)
# - GNews.io: Max ~100 articles per request
# - Recommended: Start with 50 articles, increase as needed
# - Note: More articles = longer processing time (0.5s per scraped article)

# Global storage for analysis results
current_results = {
    'topic': '',
    'timestamp': '',
    'articles': [],
    'median_score': 0,
    'mean_score': 0,
    'total_found': 0,
    'successfully_analyzed': 0
}

# ==================== SENTIMENT ANALYZER ====================
sentiment_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """Analyzes text using VADER and returns compound score (-1 to 1)."""
    if not text or not isinstance(text, str):
        return 0.0
    return sentiment_analyzer.polarity_scores(text)['compound']

def normalize_to_percentage(compound_score):
    """Converts VADER score (-1 to 1) to percentage (0 to 100)."""
    return (compound_score + 1) / 2 * 100

# ==================== WEB SCRAPING ====================
def scrape_article_content(url):
    """
    Scrapes article content and author from URL.
    Returns (text, author) or (None, None) on failure.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Try to find main article content
        article_body = None
        
        # Try semantic HTML5 tags
        if soup.find('article'):
            article_body = soup.find('article')
        elif soup.find('main'):
            article_body = soup.find('main')
        else:
            # Try common class names
            possible_containers = soup.find_all(
                ['div'],
                class_=['post-content', 'article-body', 'entry-content', 
                       'article-content', 'post-body', 'content', 'article__body']
            )
            if possible_containers:
                article_body = possible_containers[0]
        
        # Extract text
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        else:
            # Fallback: get all paragraphs from body
            paragraphs = soup.find_all('p')
            article_text = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Try to find author
        author_name = "Unknown"
        
        # Try meta tags
        meta_author = soup.find('meta', attrs={'name': 'author'}) or soup.find('meta', property='article:author')
        if meta_author and meta_author.get('content'):
            author_name = meta_author.get('content')
        else:
            # Try common author classes
            author_elem = soup.find(class_=['author', 'byline', 'author-name', 'article-author'])
            if author_elem:
                author_name = author_elem.get_text(strip=True)
        
        if article_text and len(article_text) > 100:  # Ensure we got substantial content
            return article_text, author_name
        
        return None, None
        
    except Exception as e:
        print(f"Scraping error for {url}: {e}", flush=True)
        return None, None

# ==================== API FUNCTIONS ====================
def fetch_articles_newsapi(topic, api_key, max_articles=20):
    """Fetches articles from NewsAPI.org (free tier)."""
    print(f"📡 Fetching up to {max_articles} articles from NewsAPI.org...", flush=True)
    
    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    all_articles = []
    page = 1
    
    # NewsAPI allows max 100 results per request, paginated
    while len(all_articles) < max_articles:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': topic,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(100, max_articles - len(all_articles)),  # Max 100 per page
            'page': page,
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            articles = data.get('articles', [])
            if not articles:
                break  # No more articles
            
            # Extract URLs and metadata
            for article in articles:
                if article.get('url'):
                    all_articles.append({
                        'url': article['url'],
                        'title': article.get('title', 'No title'),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'author': article.get('author', 'Unknown')
                    })
                    
                    if len(all_articles) >= max_articles:
                        break
            
            # Check if there are more pages
            total_results = data.get('totalResults', 0)
            if len(all_articles) >= total_results or len(all_articles) >= max_articles:
                break
            
            page += 1
            
        except Exception as e:
            messagebox.showerror("API Error", f"NewsAPI error: {e}")
            break
    
    print(f"✅ Found {len(all_articles)} article URLs", flush=True)
    return all_articles

def fetch_articles_gnews(topic, api_key, max_articles=20):
    """Fetches articles from GNews.io (paid tier with full content)."""
    print(f"📡 Fetching up to {max_articles} articles from GNews.io...", flush=True)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    url = "https://gnews.io/api/v4/search"
    params = {
        'q': topic,
        'lang': 'en',
        'max': min(max_articles, 100),  # GNews has limits per request
        'from': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'apikey': api_key,
        'expand': 'content'  # Paid feature
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        articles = data.get('articles', [])
        if not articles:
            return []
        
        article_list = [
            {
                'url': article['url'],
                'title': article.get('title', 'No title'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'author': article.get('author', 'Unknown'),
                'content': article.get('content', '')  # Full content from API
            }
            for article in articles if article.get('url')
        ]
        
        print(f"✅ Found {len(article_list)} articles with content", flush=True)
        return article_list
        
    except Exception as e:
        messagebox.showerror("API Error", f"GNews error: {e}")
        return []

# ==================== CHART GENERATION FUNCTIONS ====================
def generate_charts(filepath_base):
    """Generates visualization charts for the analysis results."""
    print(f"\n{'='*70}", flush=True)
    print(f"📊 CHART GENERATION STARTING", flush=True)
    print(f"{'='*70}", flush=True)
    
    if not current_results['articles']:
        print("❌ No articles to visualize", flush=True)
        return [], ["No articles to visualize"]
    
    if not MATPLOTLIB_AVAILABLE:
        print("❌ matplotlib is not installed", flush=True)
        return [], ["matplotlib is not installed. Run: pip install matplotlib"]
    
    chart_files = []
    articles = current_results['articles']
    errors = []
    
    print(f"Processing {len(articles)} articles...", flush=True)
    
    # Set cyberpunk color scheme
    plt.style.use('dark_background')
    colors_sentiment = {'POSITIVE': '#00ff00', 'NEUTRAL': '#ffff00', 'NEGATIVE': '#ff0000'}
    bg_color = '#0a0a0a'
    text_color = '#00ff00'
    
    # Chart 1: Sentiment Distribution Pie Chart
    try:
        print("\n🎨 Creating distribution pie chart...", flush=True)
        sentiment_counts = {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0}
        for article in articles:
            sentiment_counts[article['label']] += 1
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
        labels = [f"{k}\n({v} articles)" for k, v in sentiment_counts.items() if v > 0]
        sizes = [v for v in sentiment_counts.values() if v > 0]
        colors = [colors_sentiment[k] for k in sentiment_counts.keys() if sentiment_counts[k] > 0]
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': text_color, 'fontsize': 12, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(14)
            autotext.set_weight('bold')
        
        ax.set_title(
            f'SENTIMENT DISTRIBUTION\n{current_results["topic"]}',
            color=text_color,
            fontsize=16,
            weight='bold',
            pad=20
        )
        
        pie_chart_path = f"{filepath_base}_distribution.png"
        plt.tight_layout()
        plt.savefig(pie_chart_path, facecolor=bg_color, dpi=150)
        plt.close()
        chart_files.append(pie_chart_path)
        print(f"✅ Generated: {os.path.basename(pie_chart_path)}", flush=True)
    except Exception as e:
        error_msg = f"Distribution chart error: {str(e)}"
        print(f"❌ {error_msg}", flush=True)
        errors.append(error_msg)
    
    # Chart 2: Article Scores Bar Chart
    try:
        print("🎨 Creating scores bar chart...", flush=True)
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=bg_color)
        
        scores = [article['score'] for article in articles]
        labels_short = [f"#{i+1}" for i in range(len(articles))]
        bar_colors = [colors_sentiment[article['label']] for article in articles]
        
        # If too many articles, make the chart wider
        if len(articles) > 30:
            fig.set_size_inches(max(14, len(articles) * 0.3), 8)
        
        bars = ax.bar(labels_short, scores, color=bar_colors, edgecolor=text_color, linewidth=1.5)
        
        # Add median line
        median = current_results['median_score']
        ax.axhline(y=median, color='#00ccff', linestyle='--', linewidth=2, label=f'Median: {median:.1f}%')
        
        # Add average line
        mean = current_results['mean_score']
        ax.axhline(y=mean, color='#ff00ff', linestyle=':', linewidth=2, label=f'Average: {mean:.1f}%')
        
        ax.set_xlabel('Article Rank', color=text_color, fontsize=12, weight='bold')
        ax.set_ylabel('Sentiment Score (%)', color=text_color, fontsize=12, weight='bold')
        ax.set_title(
            f'SENTIMENT SCORES BY ARTICLE\n{current_results["topic"]} ({len(articles)} articles)',
            color=text_color,
            fontsize=16,
            weight='bold',
            pad=20
        )
        
        # Rotate x-axis labels if many articles
        if len(articles) > 30:
            plt.xticks(rotation=90, fontsize=8)
        
        ax.tick_params(colors=text_color)
        ax.set_facecolor(bg_color)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['top'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        ax.spines['right'].set_color(text_color)
        
        ax.legend(facecolor=bg_color, edgecolor=text_color, fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, color=text_color)
        
        bar_chart_path = f"{filepath_base}_scores.png"
        plt.tight_layout()
        plt.savefig(bar_chart_path, facecolor=bg_color, dpi=150)
        plt.close()
        chart_files.append(bar_chart_path)
        print(f"✅ Generated: {os.path.basename(bar_chart_path)}", flush=True)
    except Exception as e:
        error_msg = f"Bar chart error: {str(e)}"
        print(f"❌ {error_msg}", flush=True)
        errors.append(error_msg)
    
    # Chart 3: Sentiment Gauge/Meter
    try:
        print("🎨 Creating sentiment gauge...", flush=True)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=bg_color, subplot_kw={'projection': 'polar'})
        
        median = current_results['median_score']
        
        # Create gauge sections
        theta = np.linspace(0, np.pi, 100)
        
        # Color zones
        ax.fill_between(theta[:33], 0, 1, color='#ff0000', alpha=0.3, label='Negative (0-40%)')
        ax.fill_between(theta[33:66], 0, 1, color='#ffff00', alpha=0.3, label='Neutral (40-60%)')
        ax.fill_between(theta[66:], 0, 1, color='#00ff00', alpha=0.3, label='Positive (60-100%)')
        
        # Needle pointing to median score
        needle_angle = np.pi * (1 - median / 100)
        ax.plot([needle_angle, needle_angle], [0, 0.9], color='#00ccff', linewidth=5, marker='o', markersize=15)
        
        # Styling
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['100%', '75%', '50%', '25%', '0%'], color=text_color, fontsize=11)
        ax.spines['polar'].set_color(text_color)
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=text_color)
        
        ax.text(
            0.5, 1.3, 
            f'OVERALL SENTIMENT GAUGE\n{current_results["topic"]}\nMedian: {median:.1f}%',
            ha='center',
            va='center',
            transform=ax.transAxes,
            color=text_color,
            fontsize=14,
            weight='bold'
        )
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=bg_color, edgecolor=text_color, fontsize=9)
        
        gauge_path = f"{filepath_base}_gauge.png"
        plt.tight_layout()
        plt.savefig(gauge_path, facecolor=bg_color, dpi=150)
        plt.close()
        chart_files.append(gauge_path)
        print(f"✅ Generated: {os.path.basename(gauge_path)}", flush=True)
    except Exception as e:
        error_msg = f"Gauge chart error: {str(e)}"
        print(f"❌ {error_msg}", flush=True)
        errors.append(error_msg)
    
    # Chart 4: Top 10 Sources Distribution
    try:
        if len(articles) >= 3:
            print("🎨 Creating sources distribution chart...", flush=True)
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
            
            # Count articles by source
            source_counts = {}
            source_avg_scores = {}
            for article in articles:
                source = article['source']
                if source not in source_counts:
                    source_counts[source] = 0
                    source_avg_scores[source] = []
                source_counts[source] += 1
                source_avg_scores[source].append(article['score'])
            
            # Get top sources
            top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_sources:
                sources = [s[0][:30] for s in top_sources]  # Truncate long names
                counts = [s[1] for s in top_sources]
                avg_scores = [np.mean(source_avg_scores[s[0]]) for s in top_sources]
                
                # Color bars by average sentiment
                bar_colors_sources = []
                for score in avg_scores:
                    if score >= 60:
                        bar_colors_sources.append('#00ff00')
                    elif score >= 40:
                        bar_colors_sources.append('#ffff00')
                    else:
                        bar_colors_sources.append('#ff0000')
                
                bars = ax.barh(sources, counts, color=bar_colors_sources, edgecolor=text_color, linewidth=1.5)
                
                # Add score labels
                for i, (bar, score) in enumerate(zip(bars, avg_scores)):
                    width = bar.get_width()
                    ax.text(width + 0.1, i, f'{score:.0f}%', 
                           va='center', color=text_color, fontsize=9, weight='bold')
                
                ax.set_xlabel('Number of Articles', color=text_color, fontsize=12, weight='bold')
                ax.set_title(
                    f'TOP SOURCES\n{current_results["topic"]}\n(Labels show avg sentiment)',
                    color=text_color,
                    fontsize=16,
                    weight='bold',
                    pad=20
                )
                
                ax.tick_params(colors=text_color)
                ax.set_facecolor(bg_color)
                ax.spines['bottom'].set_color(text_color)
                ax.spines['top'].set_color(text_color)
                ax.spines['left'].set_color(text_color)
                ax.spines['right'].set_color(text_color)
                ax.grid(True, alpha=0.2, color=text_color, axis='x')
                
                sources_path = f"{filepath_base}_sources.png"
                plt.tight_layout()
                plt.savefig(sources_path, facecolor=bg_color, dpi=150)
                plt.close()
                chart_files.append(sources_path)
                print(f"✅ Generated: {os.path.basename(sources_path)}", flush=True)
    except Exception as e:
        error_msg = f"Sources chart error: {str(e)}"
        print(f"❌ {error_msg}", flush=True)
        errors.append(error_msg)
    
    print(f"\n{'='*70}", flush=True)
    print(f"📊 CHART GENERATION COMPLETE", flush=True)
    print(f"Total charts created: {len(chart_files)}", flush=True)
    if errors:
        print(f"Errors encountered: {len(errors)}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    return chart_files, errors

# ==================== EXPORT FUNCTIONS ====================
def export_csv_only():
    """Exports only the CSV file."""
    if not current_results['articles']:
        messagebox.showwarning("No Data", "No analysis results to export. Run an analysis first.")
        return
    
    # Generate default filename
    topic_safe = "".join(c for c in current_results['topic'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_filename = f"CypherPulse_{topic_safe}_{timestamp}.csv"
    
    # Ask user where to save
    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=default_filename,
        title="Save CSV Results"
    )
    
    if not filepath:
        return  # User cancelled
    
    print(f"\n{'='*70}", flush=True)
    print(f"📂 CSV EXPORT STARTING", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Save location: {filepath}", flush=True)
    
    try:
        # Export CSV
        print("\n💾 Writing CSV file...", flush=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Rank', 'Sentiment_Label', 'Sentiment_Score_%', 'Compound_Score',
                'Title', 'Source', 'Author', 'URL'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write metadata
            writer.writerow({'Rank': '# ANALYSIS METADATA', 'Sentiment_Label': '', 'Sentiment_Score_%': '', 'Compound_Score': '', 'Title': '', 'Source': '', 'Author': '', 'URL': ''})
            writer.writerow({'Rank': f'# Topic: {current_results["topic"]}', 'Sentiment_Label': '', 'Sentiment_Score_%': '', 'Compound_Score': '', 'Title': '', 'Source': '', 'Author': '', 'URL': ''})
            writer.writerow({'Rank': f'# Analysis Date: {current_results["timestamp"]}', 'Sentiment_Label': '', 'Sentiment_Score_%': '', 'Compound_Score': '', 'Title': '', 'Source': '', 'Author': '', 'URL': ''})
            writer.writerow({'Rank': f'# Median Sentiment: {current_results["median_score"]:.2f}%', 'Sentiment_Label': '', 'Sentiment_Score_%': '', 'Compound_Score': '', 'Title': '', 'Source': '', 'Author': '', 'URL': ''})
            writer.writerow({'Rank': f'# Average Sentiment: {current_results["mean_score"]:.2f}%', 'Sentiment_Label': '', 'Sentiment_Score_%': '', 'Compound_Score': '', 'Title': '', 'Source': '', 'Author': '', 'URL': ''})
            writer.writerow({'Rank': f'# Articles Analyzed: {current_results["successfully_analyzed"]}/{current_results["total_found"]}', 'Sentiment_Label': '', 'Sentiment_Score_%': '', 'Compound_Score': '', 'Title': '', 'Source': '', 'Author': '', 'URL': ''})
            writer.writerow({'Rank': '# -----', 'Sentiment_Label': '', 'Sentiment_Score_%': '', 'Compound_Score': '', 'Title': '', 'Source': '', 'Author': '', 'URL': ''})
            
            # Write article data
            for i, article in enumerate(current_results['articles'], 1):
                writer.writerow({
                    'Rank': i,
                    'Sentiment_Label': article['label'],
                    'Sentiment_Score_%': f"{article['score']:.2f}",
                    'Compound_Score': f"{article['compound_score']:.4f}",
                    'Title': article['title'],
                    'Source': article['source'],
                    'Author': article['author'],
                    'URL': article['url']
                })
        
        print(f"✅ CSV file exported successfully!", flush=True)
        print(f"   Location: {filepath}", flush=True)
        print(f"{'='*70}\n", flush=True)
        
        messagebox.showinfo(
            "CSV Export Successful",
            f"✅ CSV file saved successfully!\n\n"
            f"📄 File: {os.path.basename(filepath)}\n"
            f"📁 Location: {os.path.dirname(filepath)}\n"
            f"📈 Articles: {len(current_results['articles'])}"
        )
        
    except Exception as e:
        print(f"\n❌ CSV EXPORT ERROR: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        messagebox.showerror("Export Error", f"Failed to export CSV:\n{str(e)}")


def export_charts_only():
    """Exports only the chart visualizations."""
    if not current_results['articles']:
        messagebox.showwarning("No Data", "No analysis results to export. Run an analysis first.")
        return
    
    if not MATPLOTLIB_AVAILABLE:
        messagebox.showerror(
            "Charts Unavailable",
            "matplotlib is not installed.\n\n"
            "To enable charts, run:\npip install matplotlib\n\n"
            "Then restart CypherPulse."
        )
        return
    
    # Ask user where to save
    folder = filedialog.askdirectory(
        title="Select Folder to Save Charts"
    )
    
    if not folder:
        return  # User cancelled
    
    print(f"\n{'='*70}", flush=True)
    print(f"📊 CHARTS EXPORT STARTING", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Save folder: {folder}", flush=True)
    
    try:
        # Generate filename base
        topic_safe = "".join(c for c in current_results['topic'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath_base = os.path.join(folder, f"CypherPulse_{topic_safe}_{timestamp}")
        
        # Generate charts
        chart_files, chart_errors = generate_charts(filepath_base)
        
        if chart_files:
            existing_charts = [f for f in chart_files if os.path.exists(f)]
            
            if existing_charts:
                chart_list = '\n'.join([f"  • {os.path.basename(f)}" for f in existing_charts])
                message = (
                    f"✅ Charts exported successfully!\n\n"
                    f"📊 Charts Generated ({len(existing_charts)}):\n{chart_list}\n\n"
                    f"📁 Location: {folder}\n"
                    f"📈 Articles: {len(current_results['articles'])}"
                )
                
                if chart_errors:
                    message += f"\n\n⚠️  Some charts had errors:\n" + "\n".join([f"  • {e}" for e in chart_errors])
                
                messagebox.showinfo("Charts Export Successful", message)
            else:
                messagebox.showwarning(
                    "Export Failed",
                    "Chart files were not created.\n\n"
                    "Check the PowerShell/console window for error details."
                )
        else:
            error_details = "\n".join([f"  • {e}" for e in chart_errors]) if chart_errors else "Unknown error"
            messagebox.showerror(
                "Charts Export Failed",
                f"Chart generation failed:\n{error_details}\n\n"
                "Check console for details."
            )
        
    except Exception as e:
        print(f"\n❌ CHARTS EXPORT ERROR: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        messagebox.showerror("Export Error", f"Failed to export charts:\n{str(e)}")

# ==================== MAIN ANALYSIS FUNCTION ====================
def analyze_topic(topic, max_articles):
    """Main function that orchestrates the sentiment analysis."""
    
    # Clear previous results
    global current_results
    current_results = {
        'topic': topic,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'articles': [],
        'median_score': 0,
        'mean_score': 0,
        'total_found': 0,
        'successfully_analyzed': 0
    }
    
    # Step 1: Fetch articles
    if USE_GNEWS:
        articles = fetch_articles_gnews(topic, GNEWS_API_KEY, max_articles)
    else:
        articles = fetch_articles_newsapi(topic, NEWSAPI_KEY, max_articles)
    
    if not articles:
        update_results(f"❌ No articles found for '{topic}'")
        enable_button()
        return
    
    current_results['total_found'] = len(articles)
    update_results(f"🔍 Found {len(articles)} articles. Starting analysis...\n")
    
    # Add a note about scraping time for large numbers
    if len(articles) > 50:
        update_results(f"⏰ Analyzing {len(articles)} articles may take several minutes...\n")
        update_results(f"⚠️  Being polite to servers (0.5s delay per article)...\n\n")
    
    results = []
    sentiment_scores = []
    successful = 0
    failed = 0
    
    # Step 2: Process each article
    for i, article in enumerate(articles):
        url = article['url']
        title = article['title']
        source = article['source']
        
        update_results(f"📄 [{i+1}/{len(articles)}] Processing: {title[:60]}...")
        
        # Get content (either from API or by scraping)
        if USE_GNEWS and article.get('content'):
            content = article['content']
            author = article.get('author', 'Unknown')
        else:
            content, author = scrape_article_content(url)
            time.sleep(0.5)  # Be polite to servers
        
        if not content:
            failed += 1
            update_results(f"   ⚠️  Could not extract content from {source}\n")
            continue
        
        # Step 3: Analyze sentiment
        compound_score = analyze_sentiment(content)
        percentage_score = normalize_to_percentage(compound_score)
        sentiment_scores.append(percentage_score)
        successful += 1
        
        # Determine sentiment label
        if percentage_score >= 60:
            label = "POSITIVE"
            emoji = "😊"
        elif percentage_score >= 40:
            label = "NEUTRAL"
            emoji = "😐"
        else:
            label = "NEGATIVE"
            emoji = "😞"
        
        result_entry = {
            'url': url,
            'title': title,
            'source': source,
            'author': author,
            'score': percentage_score,
            'label': label,
            'compound_score': compound_score
        }
        results.append(result_entry)
        
        update_results(f"   ✅ {emoji} {label} | Score: {percentage_score:.1f}% | Author: {author}\n")
    
    # Step 4: Calculate statistics
    if not sentiment_scores:
        update_results(f"\n❌ No articles could be analyzed successfully.")
        update_results(f"\n⚠️  {failed} articles failed to scrape. Try a different topic or check your internet connection.")
        enable_button()
        return
    
    median_score = np.median(sentiment_scores)
    mean_score = np.mean(sentiment_scores)
    
    # Store results globally
    current_results['articles'] = results
    current_results['median_score'] = median_score
    current_results['mean_score'] = mean_score
    current_results['successfully_analyzed'] = successful
    
    # Overall sentiment
    if median_score >= 60:
        overall_sentiment = "😊 POSITIVE"
    elif median_score >= 40:
        overall_sentiment = "😐 NEUTRAL"
    else:
        overall_sentiment = "😞 NEGATIVE"
    
    # Step 5: Generate final report
    report = "\n" + "="*70 + "\n"
    report += "🔮 CYPHERPULSE SENTIMENT ANALYSIS REPORT\n"
    report += "="*70 + "\n\n"
    report += f"📊 Topic: {topic}\n"
    report += f"📅 Time Range: Last 7 days\n"
    report += f"📰 Articles Found: {len(articles)}\n"
    report += f"✅ Successfully Analyzed: {successful}\n"
    if failed > 0:
        report += f"⚠️  Failed to Scrape: {failed}\n"
    report += f"\n{'─'*70}\n"
    report += f"📈 OVERALL SENTIMENT: {overall_sentiment}\n"
    report += f"🎯 Median Score: {median_score:.1f}%\n"
    report += f"📊 Average Score: {mean_score:.1f}%\n"
    report += f"{'─'*70}\n\n"
    
    report += "📋 DETAILED RESULTS:\n\n"
    
    for i, result in enumerate(results, 1):
        emoji = "😊" if result['label'] == "POSITIVE" else "😐" if result['label'] == "NEUTRAL" else "😞"
        report += f"{i}. {emoji} {result['label']}\n"
        report += f"   Score: {result['score']:.1f}%\n"
        report += f"   Title: {result['title']}\n"
        report += f"   Source: {result['source']}\n"
        report += f"   Author: {result['author']}\n"
        report += f"   URL: {result['url']}\n\n"
    
    report += "="*70 + "\n"
    report += "🔮 Analysis complete. Stay wired, netrunner.\n"
    report += "💾 Click 'Export CSV' or 'Export Charts' to save results.\n"
    
    update_results(report, clear=True)
    enable_button()
    enable_export_buttons()

# ==================== GUI FUNCTIONS ====================
def start_analysis():
    """Triggered when user clicks the Start Analysis button."""
    topic = topic_entry.get().strip()
    
    if not topic:
        messagebox.showwarning("Input Required", "Please enter a topic to analyze")
        return
    
    # Get number of articles
    try:
        max_articles = int(articles_spinbox.get())
        if max_articles < 1:
            messagebox.showwarning("Invalid Number", "Please enter a number greater than 0")
            return
        if max_articles > 500:
            result = messagebox.askyesno(
                "Large Analysis",
                f"Analyzing {max_articles} articles will take considerable time "
                f"({max_articles * 0.5 / 60:.1f}+ minutes with scraping delays).\n\n"
                "Continue anyway?"
            )
            if not result:
                return
    except ValueError:
        messagebox.showwarning("Invalid Number", "Please enter a valid number of articles")
        return
    
    # Check if API key is set
    if USE_GNEWS:
        if "YOUR_GNEWS_API_KEY" in globals().get('GNEWS_API_KEY', ''):
            messagebox.showerror("API Key Required", 
                               "Please set your GNews.io API key in the code")
            return
    else:
        if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE":
            messagebox.showerror("API Key Required", 
                               "Please set your NewsAPI.org API key in the code.\n\n"
                               "Get a free key at: https://newsapi.org/")
            return
    
    # Disable buttons and clear results
    analyze_button.config(state=tk.DISABLED)
    disable_export_buttons()
    results_text.config(state=tk.NORMAL)
    results_text.delete(1.0, tk.END)
    results_text.insert(tk.END, f"🔮 CYPHERPULSE INITIATED...\n")
    results_text.insert(tk.END, f"🎯 Target Topic: {topic}\n")
    results_text.insert(tk.END, f"📊 Target Articles: {max_articles}\n")
    results_text.insert(tk.END, f"⚡ Decrypting online sentiment...\n\n")
    results_text.config(state=tk.DISABLED)
    
    # Run analysis in background thread
    thread = threading.Thread(target=analyze_topic, args=(topic, max_articles))
    thread.daemon = True
    thread.start()

def update_results(message, clear=False):
    """Updates the results text box."""
    def task():
        results_text.config(state=tk.NORMAL)
        if clear:
            results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, message)
        results_text.see(tk.END)
        results_text.config(state=tk.DISABLED)
    
    root.after(0, task)

def enable_button():
    """Re-enables the analyze button."""
    def task():
        analyze_button.config(state=tk.NORMAL)
    root.after(0, task)

def enable_export_buttons():
    """Enables the export buttons after successful analysis."""
    def task():
        csv_export_button.config(state=tk.NORMAL)
        if MATPLOTLIB_AVAILABLE:
            charts_export_button.config(state=tk.NORMAL)
    root.after(0, task)

def disable_export_buttons():
    """Disables the export buttons."""
    def task():
        csv_export_button.config(state=tk.DISABLED)
        charts_export_button.config(state=tk.DISABLED)
    root.after(0, task)

# ==================== GUI SETUP ====================
root = tk.Tk()
root.title("🔮 CypherPulse - Sentiment Analyzer")
root.geometry("900x700")
root.configure(bg='#1a1a1a')

# Style
style_bg = '#1a1a1a'
style_fg = '#00ff00'
style_button_bg = '#2a2a2a'

# Main frame
main_frame = tk.Frame(root, bg=style_bg, padx=20, pady=20)
main_frame.pack(expand=True, fill=tk.BOTH)

# Header
header_label = tk.Label(
    main_frame, 
    text="🔮 CYPHERPULSE", 
    font=('Courier', 24, 'bold'),
    bg=style_bg,
    fg=style_fg
)
header_label.pack(pady=(0, 5))

subtitle_label = tk.Label(
    main_frame,
    text="Decrypt the sentiment pulse of the net",
    font=('Courier', 10),
    bg=style_bg,
    fg='#888888'
)
subtitle_label.pack(pady=(0, 20))

# Input frame
input_frame = tk.Frame(main_frame, bg=style_bg)
input_frame.pack(fill=tk.X, pady=10)

topic_label = tk.Label(
    input_frame,
    text="TARGET TOPIC:",
    font=('Courier', 10, 'bold'),
    bg=style_bg,
    fg=style_fg
)
topic_label.pack(side=tk.LEFT, padx=(0, 10))

topic_entry = tk.Entry(
    input_frame,
    font=('Courier', 11),
    bg='#2a2a2a',
    fg='#00ff00',
    insertbackground='#00ff00',
    width=35
)
topic_entry.pack(side=tk.LEFT, padx=(0, 10))
topic_entry.focus()

articles_label = tk.Label(
    input_frame,
    text="ARTICLES:",
    font=('Courier', 10, 'bold'),
    bg=style_bg,
    fg=style_fg
)
articles_label.pack(side=tk.LEFT, padx=(0, 5))

articles_spinbox = tk.Spinbox(
    input_frame,
    from_=1,
    to=500,
    width=8,
    font=('Courier', 11),
    bg='#2a2a2a',
    fg='#00ff00',
    insertbackground='#00ff00',
    buttonbackground='#2a2a2a'
)
articles_spinbox.delete(0, tk.END)
articles_spinbox.insert(0, "50")  # Default value
articles_spinbox.pack(side=tk.LEFT, padx=(0, 10))

analyze_button = tk.Button(
    input_frame,
    text="▶ ANALYZE",
    command=start_analysis,
    font=('Courier', 10, 'bold'),
    bg=style_button_bg,
    fg=style_fg,
    activebackground='#3a3a3a',
    activeforeground='#00ff00',
    cursor='hand2'
)
analyze_button.pack(side=tk.LEFT, padx=(0, 5))

csv_export_button = tk.Button(
    input_frame,
    text="💾 EXPORT CSV",
    command=export_csv_only,
    font=('Courier', 9, 'bold'),
    bg=style_button_bg,
    fg='#00ccff',
    activebackground='#3a3a3a',
    activeforeground='#00ccff',
    cursor='hand2',
    state=tk.DISABLED
)
csv_export_button.pack(side=tk.LEFT, padx=(0, 5))

charts_export_button = tk.Button(
    input_frame,
    text="📊 EXPORT CHARTS",
    command=export_charts_only,
    font=('Courier', 9, 'bold'),
    bg=style_button_bg,
    fg='#ff00ff',
    activebackground='#3a3a3a',
    activeforeground='#ff00ff',
    cursor='hand2',
    state=tk.DISABLED
)
charts_export_button.pack(side=tk.LEFT)

# Results frame
results_frame = tk.Frame(main_frame, bg=style_bg)
results_frame.pack(expand=True, fill=tk.BOTH, pady=10)

results_text = scrolledtext.ScrolledText(
    results_frame,
    wrap=tk.WORD,
    font=('Courier', 9),
    bg='#0a0a0a',
    fg='#00ff00',
    insertbackground='#00ff00',
    state=tk.DISABLED
)
results_text.pack(expand=True, fill=tk.BOTH)

# Footer
footer_text = "⚡ Powered by VADER | NewsAPI.org | 💾 CSV Export | "
if MATPLOTLIB_AVAILABLE:
    footer_text += "📊 Charts Export"
else:
    footer_text += "⚠️  Install matplotlib for charts"
footer_text += " | 📈 1-500 Articles"

footer_label = tk.Label(
    main_frame,
    text=footer_text,
    font=('Courier', 8),
    bg=style_bg,
    fg='#555555'
)
footer_label.pack(pady=(10, 0))

# Bind Enter key
topic_entry.bind('<Return>', lambda e: start_analysis())

# Show matplotlib warning if not installed
if not MATPLOTLIB_AVAILABLE:
    root.after(500, lambda: messagebox.showwarning(
        "Charts Unavailable",
        "⚠️  matplotlib is not installed.\n\n"
        "CSV export will work, but charts won't be generated.\n\n"
        "To enable chart generation, run:\n"
        "pip install matplotlib\n\n"
        "Then restart CypherPulse."
    ))

root.mainloop()