# Resume Optimization System

A powerful resume analysis and optimization system that uses Gemini AI to evaluate resumes against job descriptions. The system provides detailed matching analysis, suggestions for improvement, and structured information extraction.

## Features

- Multiple resume upload support (PDF and DOCX formats)
- AI-powered resume analysis using Google's Gemini API
- Semantic matching score calculation
- Detailed feedback and improvement suggestions
- Structured information extraction (skills, experience, education)
- User-friendly web interface using Streamlit

## Requirements

- Python 3.8+
- Gemini API key from Google AI Studio
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd resume-optimization-system
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Gemini API key:
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Replace "YOUR_GEMINI_API_KEY" in app.py with your actual API key

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload one or more resumes (PDF or DOCX format)

4. Paste the job description

5. View the analysis results:
   - Match score
   - Detailed analysis and suggestions
   - Extracted information

## File Structure

- `app.py`: Main application file containing the Streamlit interface and analysis logic
- `requirements.txt`: List of Python dependencies
- `README.md`: Project documentation

## How It Works

1. **Resume Parsing**: The system extracts text from PDF and DOCX files using PyPDF2 and python-docx libraries.

2. **Match Score Calculation**: Uses sentence-transformers to calculate semantic similarity between the resume and job description.

3. **AI Analysis**: Leverages Gemini AI to provide detailed analysis, including:
   - Match classification (High/Low)
   - Reasons for low matching
   - Actionable improvement suggestions

4. **Information Extraction**: Extracts structured information using regular expressions:
   - Skills
   - Experience
   - Education

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 