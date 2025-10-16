# 🔮 CYPHERPULSE

```
 ██████╗██╗   ██╗██████╗ ██╗  ██╗███████╗██████╗ ██████╗ ██╗   ██╗██╗     ███████╗███████╗
██╔════╝╚██╗ ██╔╝██╔══██╗██║  ██║██╔════╝██╔══██╗██╔══██╗██║   ██║██║     ██╔════╝██╔════╝
██║      ╚████╔╝ ██████╔╝███████║█████╗  ██████╔╝██████╔╝██║   ██║██║     ███████╗█████╗  
██║       ╚██╔╝  ██╔═══╝ ██╔══██║██╔══╝  ██╔══██╗██╔═══╝ ██║   ██║██║     ╚════██║██╔══╝  
╚██████╗   ██║   ██║     ██║  ██║███████╗██║  ██║██║     ╚██████╔╝███████╗███████║███████╗
 ╚═════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚══════╝╚══════╝╚══════╝
```

> **[CLASSIFIED]** Decrypt the sentiment pulse of the net. Extract emotional intelligence from the digital void.

[![Python](https://img.shields.io/badge/Python-3.6+-00ff00?style=for-the-badge&logo=python&logoColor=00ff00)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-00ff00?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-OPERATIONAL-00ff00?style=for-the-badge)]()

---

## ⚡ SYSTEM OVERVIEW

**CypherPulse** is an autonomous sentiment analysis engine designed for netrunners, data analysts, and digital investigators. It scrapes the web for recent articles, extracts content, and quantifies the emotional tone using advanced NLP algorithms.

**CAPABILITIES:**
- 🔍 **Real-time Article Discovery** - Query NewsAPI for up to 500 recent articles
- 🤖 **Intelligent Web Scraping** - Fault-tolerant content extraction from any source
- 📊 **VADER Sentiment Analysis** - Military-grade emotional intelligence scoring
- 📈 **Data Visualization** - Cyberpunk-themed charts and graphs
- 💾 **Export Modules** - CSV and PNG output for further analysis

---

## 🛠️ INSTALLATION PROTOCOL

### Prerequisites
```bash
# System Requirements
Python 3.6 or higher
pip package manager
Active internet connection
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/cypherpulse.git
cd cypherpulse
```

### Step 2: Install Dependencies
```bash
pip install requests beautifulsoup4 vaderSentiment numpy matplotlib
```

### Step 3: Obtain API Credentials

**[REQUIRED]** Get a free NewsAPI key:
1. Navigate to https://newsapi.org/
2. Register for a free account (100 requests/day)
3. Copy your API key

**[OPTIONAL]** For enhanced capabilities, subscribe to GNews.io (~$50/month):
- Full article content without scraping
- Higher rate limits
- Get key at https://gnews.io/

### Step 4: Configure API Key
Open `CYPHERPULSE_v5.py` in a text editor and replace:
```python
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY_HERE"  # <-- PASTE YOUR KEY HERE
```

---

## 🚀 LAUNCH SEQUENCE

### Quick Start
```bash
# Standard launch
python CYPHERPULSE_v5.py

# Recommended: Launch with unbuffered output for real-time console logs
python -u CYPHERPULSE_v5.py
```

### Interface Overview
```
┌─────────────────────────────────────────────────────────┐
│  🔮 CYPHERPULSE                                         │
│  Decrypt the sentiment pulse of the net                │
├─────────────────────────────────────────────────────────┤
│  TARGET TOPIC: [_____________] ARTICLES: [50] [▶ ANALYZE] │
│  [💾 EXPORT CSV] [📊 EXPORT CHARTS]                    │
├─────────────────────────────────────────────────────────┤
│  [Analysis Results Display]                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📖 OPERATION MANUAL

### Basic Analysis
1. **Enter Target Topic** - Type your search query (e.g., "Bitcoin", "AI regulation")
2. **Set Article Count** - Choose 1-500 articles (default: 50)
3. **Initiate Analysis** - Click "▶ ANALYZE" or press Enter
4. **Wait for Processing** - Watch real-time progress in the terminal
5. **Review Results** - Examine sentiment scores and statistics

### Export Options

#### CSV Export 💾
- Click **"💾 EXPORT CSV"** after analysis
- Choose save location
- Includes metadata, scores, titles, authors, and URLs
- Compatible with Excel, Google Sheets, and data analysis tools

#### Charts Export 📊
- Click **"📊 EXPORT CHARTS"** after analysis
- Select folder for chart files
- Generates 4 professional visualizations:
  - **Distribution Pie Chart** - Sentiment breakdown
  - **Scores Bar Chart** - Individual article scores
  - **Sentiment Gauge** - Overall sentiment meter
  - **Top Sources Chart** - Source analysis

### Performance Tips
```
⚡ Quick Test (20-50 articles)   → ~10-25 seconds
📊 Standard Analysis (100)        → ~50 seconds  
🔬 Deep Research (200-500)        → 2-5 minutes
```

**NOTE:** Processing time = Articles × 0.5 seconds (due to polite scraping delays)

---

## 🎨 OUTPUT SPECIFICATIONS

### Sentiment Scoring System
```
100% ═══════════════════════════════ 60%  😊 POSITIVE
 60% ═══════════════════════════════ 40%  😐 NEUTRAL
 40% ═══════════════════════════════  0%  😞 NEGATIVE
```

### Statistical Metrics
- **Median Score** - Most reliable central tendency (outlier-resistant)
- **Average Score** - Mean of all sentiment values
- **Success Rate** - % of articles successfully analyzed

### Chart Specifications
- **Resolution:** 150 DPI (publication-ready)
- **Format:** PNG with transparency
- **Color Scheme:** Dark background with neon accents
- **Dimensions:** Optimized for both screen and print

---

## 🔧 ADVANCED CONFIGURATION

### Switching to GNews.io
```python
# In CYPHERPULSE_v5.py
USE_GNEWS = True  # Enable GNews.io
GNEWS_API_KEY = "your_gnews_key_here"
```

### Adjusting Scraping Delays
```python
# Modify in scrape_article_content()
time.sleep(0.5)  # Increase for slower, more polite scraping
```

### Customizing Article Limits
```python
# Change spinbox range in GUI setup
articles_spinbox = tk.Spinbox(
    from_=1,
    to=1000,  # <-- Increase maximum
    ...
)
```

---

## 🐛 TROUBLESHOOTING

### Issue: No Charts Generated
**Solution:**
```bash
pip install matplotlib
# Then restart CypherPulse
```

### Issue: "API Key Required" Error
**Solution:** Replace `YOUR_NEWSAPI_KEY_HERE` with your actual API key

### Issue: Many Failed Scrapes
**Causes:**
- Paywalled content (WSJ, NYT Premium)
- JavaScript-heavy sites
- Anti-bot protections

**Solutions:**
- Use GNews.io for full content
- Increase article count to compensate
- Choose topics with open-access sources

### Issue: Console Output Not Showing
**Solution:** Launch with unbuffered output:
```bash
python -u CYPHERPULSE_v5.py
```

---

## 📊 TECHNICAL ARCHITECTURE

### Core Technologies
```
┌─────────────────────────────────────────┐
│  Tkinter          → GUI Framework       │
│  Requests         → HTTP Client         │
│  BeautifulSoup4   → HTML Parser         │
│  VADER            → Sentiment Engine    │
│  NumPy            → Statistics          │
│  Matplotlib       → Visualization       │
│  NewsAPI.org      → Data Source         │
└─────────────────────────────────────────┘
```

### Processing Pipeline
```
[User Input] → [API Query] → [Web Scraping] → [Sentiment Analysis]
                   ↓              ↓                    ↓
           [Article URLs]  [Content Text]    [Scores 0-100%]
                                                      ↓
                                            [Statistical Summary]
                                                      ↓
                                            [Export CSV/Charts]
```

---

## 🎯 USE CASES

### Business Intelligence
- Monitor brand sentiment across news sources
- Track competitor mentions and perception
- Identify emerging trends in your industry

### Market Research
- Gauge public opinion on products/services
- Compare sentiment across different topics
- Track sentiment changes over time

### Academic Research
- Analyze media coverage of events
- Study public discourse on social issues
- Quantify bias in news reporting

### Personal Projects
- Track sentiment on your interests
- Compare coverage across news sources
- Create data-driven blog content

---

## 🔒 PRIVACY & ETHICS

**CypherPulse operates with respect:**
- ✅ Public articles only (no authentication bypass)
- ✅ Polite scraping with delays
- ✅ Respects robots.txt
- ✅ No personal data collection
- ✅ Local processing (data never leaves your machine)

**Ethical Use Guidelines:**
- Don't overload servers (keep article counts reasonable)
- Don't use for harassment or targeting
- Respect copyright when sharing results
- Be transparent about automated analysis

---

## 📜 LICENSE

```
MIT License

Copyright (c) 2025 CypherPulse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🤝 CONTRIBUTING

Contributions are welcome! Areas for enhancement:
- 🌐 Additional API integrations
- 📊 More visualization types
- 🧠 Alternative NLP models
- 🔍 Advanced filtering options
- 🌍 Multi-language support

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Open a Pull Request

---

## 🔗 RESOURCES

- **VADER Sentiment:** [https://github.com/cjhutto/vaderSentiment](https://github.com/cjhutto/vaderSentiment)
- **NewsAPI Documentation:** [https://newsapi.org/docs](https://newsapi.org/docs)
- **Beautiful Soup Docs:** [https://www.crummy.com/software/BeautifulSoup/](https://www.crummy.com/software/BeautifulSoup/)
- **Matplotlib Gallery:** [https://matplotlib.org/stable/gallery/](https://matplotlib.org/stable/gallery/)

---

## 💬 SUPPORT

**Found a bug?** Open an issue on GitHub

**Have questions?** Check the troubleshooting section or create a discussion

**Want to share results?** Tag us with your findings!

---

```
[TRANSMISSION END]

"In the neon-lit corridors of the net, CypherPulse decodes the pulse of humanity."

Stay wired. Stay informed. Stay ahead.

    ⚡ CYPHERPULSE ⚡
```

---

**⭐ Star this repo if CypherPulse helped you decrypt the net!**