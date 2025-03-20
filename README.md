# Resume Rank & Optimization

A powerful AI-based tool that analyzes resumes against job descriptions, providing match scores and optimization suggestions.

## Features

- **Resume Analysis**: Calculate how well a resume matches a specific job description
- **Match Scoring**: Get detailed scoring across multiple dimensions including technical skills, soft skills, and keyword matching
- **Resume Optimization**: Receive AI-generated suggestions to improve your resume
- **Batch Processing**: Compare multiple resumes against a single job description
- **Single Resume Mode**: Deep dive into a single resume for detailed analysis

## How to Run

1. Make sure you have Python 3.8+ installed

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

4. The application will open in your web browser at http://localhost:8501

## Usage

### Single Resume Analysis

1. Upload your resume (PDF or DOCX format)
2. Paste the job description you're applying for
3. View your match score and detailed analysis
4. Get an optimized version of your resume tailored for the specific job

### Batch Processing

1. Upload multiple resumes (PDF or DOCX format)
2. Paste the job description
3. See all resumes ranked by match score
4. Click to view detailed analysis for each resume

## How It Works

The application uses a hybrid approach to calculate match scores:

- Semantic similarity (25% weight) - Vector embeddings measure overall text similarity
- Overall context understanding (25% weight) - AI analysis of career narrative alignment
- Technical skills match (15% weight) - Specific technical skills comparison
- Soft skills match (8% weight) - Communication and interpersonal skills alignment
- Experience match (12% weight) - Years and relevance of experience
- Education match (5% weight) - Education requirements fulfillment
- Keyword match (10% weight) - Domain-specific terminology overlap

## License

This project is licensed under the MIT License - see the LICENSE file for details. 