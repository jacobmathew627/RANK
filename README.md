# Resume Rank & Optimization

A powerful AI-powered resume analysis and optimization system that uses Google's Gemini AI to evaluate resumes against job descriptions, optimize content, and generate professional-looking formatted resumes.

## Core Features

1. **Intuitive User Interface**: Provides a user-friendly design for seamless navigation and interaction.

2. **Single Resume Optimization**: Offers personalized enhancement of individual resumes to align with specific job requirements.

3. **Batch Processing**: Enables simultaneous optimization of multiple resumes against a job description.

4. **Match Score Analysis**: Calculates and displays a compatibility score between resumes and job descriptions.

5. **AI-Powered Resume Optimization**: Utilizes AI to intelligently rewrite resumes for improved job matching.

6. **Multiple Download Formats**: Download optimized resumes in TXT, PDF, or HTML formats with professional formatting.

7. **Customizable Settings**: Allows users to configure API keys and adjust application preferences.

## Requirements

- Python 3.8+
- Gemini API key from Google AI Studio
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/resume-rank.git
cd resume-rank
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Gemini API key in the application settings page.

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload your resume (PDF or DOCX format)

4. Paste the job description

5. View the analysis results and optimized resume:
   - Match score with detailed breakdown
   - Keyword analysis showing matches and missing keywords
   - AI-generated suggestions for improvement
   - Optimized resume tailored to the job description

6. Download your optimized resume in your preferred format (TXT, PDF, or HTML)

## Deployment

This application can be deployed to Streamlit Cloud for free:

1. Push your code to a GitHub repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy your app with one click

## How It Works

1. **Resume Parsing**: The system extracts text from PDF and DOCX files.

2. **Match Score Calculation**: Uses a combination of semantic embeddings and keyword matching to calculate compatibility.

3. **Analysis Generation**: Leverages Gemini AI to provide detailed analysis and recommendations.

4. **Resume Optimization**: Uses AI to rewrite the resume for better job matching.

5. **Professional Formatting**: Generates professionally formatted PDFs and HTML versions of the optimized resume.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 