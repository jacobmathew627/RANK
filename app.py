import PyPDF2
import docx
import re
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
import os
import random
import nltk
import io
from fpdf import FPDF
import base64
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.warning(f"Warning: Could not download NLTK data. Some functionality may be limited. Error: {e}")

# Set random seed for numpy and Python's random for more consistent results
random.seed(42)
np.random.seed(42)

# Initialize session state for API key
if 'api_key' not in st.session_state:
    # Try to get API key from environment variable or use a placeholder
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")
    st.session_state.api_key_configured = False

# Initialize session state for storing previous calculations to ensure consistency
if 'previous_calculations' not in st.session_state:
    st.session_state.previous_calculations = {}

# Initialize Gemini model and embedding model
model = None
embedding_model = None

def initialize_models():
    """Initialize AI models with the configured API key"""
    global model, embedding_model
    
    try:
        # Configure Gemini API with the key from session state
        if st.session_state.api_key:
            genai.configure(api_key=st.session_state.api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            st.session_state.api_key_configured = True
            
            # Initialize Sentence Transformer model for embeddings
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            return True
    except Exception as e:
        st.error(f"Error initializing AI models: {e}")
        st.session_state.api_key_configured = False
        return False
    
    return False

# Initialize models on startup if API key is available
if st.session_state.api_key:
    initialize_models()

def upload_and_parse_resume(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        return None
    return text

def extract_intelligent_keywords(resume_text):
    """
    Extract technical skills, soft skills and experience keywords from resume text
    using pattern matching and common skill lists
    
    Args:
        resume_text (str): The parsed text from the resume
        
    Returns:
        tuple: (tech_skills, soft_skills, experience_keywords)
    """
    # Common technical skills for pattern matching
    common_tech_skills = [
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c\+\+', 'c#', 'ruby', 'php', 'swift', 'kotlin',
        'go', 'rust', 'scala', 'perl', 'r programming', 'matlab', 'fortran', 'dart', 'lua', 'haskell',
        
        # Web Development
        'html', 'css', 'react', 'angular', 'vue', 'node\.js', 'express', 'django', 'flask', 'spring',
        'asp\.net', 'laravel', 'ruby on rails', 'jquery', 'bootstrap', 'tailwind', 'webpack', 'gatsby',
        'next\.js', 'graphql', 'restful api', 'soap', 'xml', 'json', 'ajax',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'nosql', 'redis', 'cassandra',
        'mariadb', 'dynamodb', 'firebase', 'elasticsearch', 'neo4j', 'couchbase', 'ms sql',
        
        # Data Science & ML
        'machine learning', 'deep learning', 'neural networks', 'ai', 'artificial intelligence',
        'natural language processing', 'nlp', 'computer vision', 'data mining', 'tensorflow',
        'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy', 'matplotlib', 'tableau',
        'power bi', 'statistics', 'regression', 'classification', 'clustering', 'forecasting',
        
        # Cloud & DevOps
        'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'devops', 'docker', 'kubernetes',
        'jenkins', 'terraform', 'ansible', 'puppet', 'chef', 'github actions', 'gitlab ci',
        'travis ci', 'circleci', 'ci/cd', 'serverless', 'microservices', 'linux', 'unix', 'bash',
        
        # Mobile Development
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic', 'swift', 'objective-c',
        'mobile app', 'responsive design', 'pwa', 'progressive web app',
        
        # Other Technical Skills
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'agile', 'scrum', 'kanban',
        'rest api', 'soap api', 'graphql', 'oauth', 'jwt', 'api gateway', 'microservices',
        'design patterns', 'oop', 'object-oriented', 'functional programming'
    ]
    
    # Common soft skills for pattern matching
    common_soft_skills = [
        'communication', 'teamwork', 'collaboration', 'problem solving', 'critical thinking',
        'creativity', 'leadership', 'management', 'time management', 'adaptability', 'flexibility',
        'organization', 'analytical', 'interpersonal', 'attention to detail', 'multitasking',
        'decision making', 'conflict resolution', 'negotiation', 'presentation', 'public speaking',
        'customer service', 'emotional intelligence', 'empathy', 'active listening', 'mentoring',
        'coaching', 'training', 'strategic thinking', 'planning', 'research', 'writing', 'editing',
        'self-motivated', 'proactive', 'resilience', 'work ethic', 'cultural awareness', 'networking'
    ]
    
    # Common experience patterns
    experience_patterns = [
        r'\b(\d+)[\+]?\s+years?\s+(?:of\s+)?experience\b',
        r'experience\s+(?:of|with|in)?\s+(\d+)[\+]?\s+years?\b',
        r'\b(senior|junior|lead|principal|staff|experienced|expert)\b',
        r'\bmanag(?:er|ed|ing|ement)\b', 
        r'\b(director|head|chief|vp|vice president|executive)\b',
        r'\b(intern|internship|entry[ -]level|graduate)\b'
    ]
    
    # Normalize text for consistent matching
    text_lower = resume_text.lower()
    
    # Extract technical skills
    tech_skills = set()
    for skill in common_tech_skills:
        if re.search(fr'\b{skill}\b', text_lower):
            # Format skill name for better display (capitalize first letter of each word)
            formatted_skill = ' '.join(word.capitalize() for word in skill.replace('\\', '').split())
            tech_skills.add(formatted_skill)
    
    # Extract soft skills
    soft_skills = set()
    for skill in common_soft_skills:
        if re.search(fr'\b{skill}\b', text_lower):
            # Format skill name for better display (capitalize first letter of each word)
            formatted_skill = ' '.join(word.capitalize() for word in skill.split())
            soft_skills.add(formatted_skill)
    
    # Extract experience keywords
    experience_keywords = set()
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            if match.groups():
                experience_keywords.add(match.group(0))
            else:
                experience_keywords.add(match.group(0))
    
    return sorted(list(tech_skills)), sorted(list(soft_skills)), sorted(list(experience_keywords))

def calculate_match_score(resume_text, job_description):
    """
    Calculate a match score between a resume and job description using a 3-step process:
    1. BERT-based Embedding Similarity (Semantic Understanding)
    2. TF-IDF/Keyword Matching (Keyword Relevance)
    3. LLM Refinement using Gemini (Contextual Evaluation & Final Scoring)
    
    Args:
        resume_text (str): The parsed text from the resume
        job_description (str): The job description text
        
    Returns:
        tuple: (percentage_match, enhanced_scores, keywords_found, keywords_missing, resume_keywords)
    """
    # Create a unique key for this resume-job pair to check if we've calculated it before
    # Use a hash of the texts to create a compact identifier
    cache_key = hash((resume_text[:1000], job_description[:1000]))  # Use first 1000 chars for efficiency
    
    # Check if we've already calculated this exact pair
    if cache_key in st.session_state.previous_calculations:
        print("Using cached match score calculation")
        return st.session_state.previous_calculations[cache_key]
    
    # Initialize default values in case of errors
    default_score = 0.5
    enhanced_scores = {
        "technical_skills": default_score * 100,
        "soft_skills": default_score * 100,
        "experience": default_score * 100,
        "education": default_score * 100,
        "keyword_match": default_score * 100
    }
    keywords_found = []
    keywords_missing = []
    resume_keywords = []
    
    # Extract intelligent keywords from resume
    tech_skills, soft_skills, experience_keywords = extract_intelligent_keywords(resume_text)
    
    # Initialize resume_keywords with extracted skills
    resume_keywords = tech_skills + soft_skills + experience_keywords
    
    # Step 1: BERT-based Embedding Similarity
    try:
        if embedding_model is None:
            print("Warning: Embedding model not loaded. Using default similarity score.")
            similarity_score = default_score
        else:
            print("Calculating BERT embedding similarity...")
            # Ensure consistent truncation to avoid variable inputs
            max_token_length = 512
            truncated_resume = ' '.join(resume_text.split()[:max_token_length])
            truncated_job = ' '.join(job_description.split()[:max_token_length])
            
            # Calculate embeddings with fixed parameters
            resume_embedding = embedding_model.encode([truncated_resume], normalize_embeddings=True)
            job_embedding = embedding_model.encode([truncated_job], normalize_embeddings=True)
            similarity_score = np.dot(resume_embedding, job_embedding.T)[0][0]
            
            # Normalize similarity score to [0, 1]
            similarity_score = (similarity_score + 1) / 2
            print(f"BERT Embedding Similarity Score: {similarity_score:.2f}")
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        similarity_score = default_score
    
    # Step 2: TF-IDF/Keyword Matching
    try:
        # Extract tech skills, soft skills, and experience keywords from job description
        job_tech_skills, job_soft_skills, job_exp_keywords = extract_intelligent_keywords(job_description)
        
        # Count matches between job and resume
        tech_skills_matches = len(set(tech_skills).intersection(set(job_tech_skills)))
        soft_skills_matches = len(set(soft_skills).intersection(set(job_soft_skills)))
        
        # Calculate match rates
        tech_skills_rate = tech_skills_matches / len(job_tech_skills) if job_tech_skills else 0.5
        soft_skills_rate = soft_skills_matches / len(job_soft_skills) if job_soft_skills else 0.5
        
        # Combine for overall keyword match rate
        if job_tech_skills or job_soft_skills:
            keyword_match_rate = (tech_skills_rate * len(job_tech_skills) + 
                                 soft_skills_rate * len(job_soft_skills)) / (
                                 len(job_tech_skills) + len(job_soft_skills))
        else:
            keyword_match_rate = 0.5
            
        print(f"Keyword Match Rate: {keyword_match_rate:.2f}")
    except Exception as e:
        print(f"Error calculating keyword match: {e}")
        keyword_match_rate = default_score
    
    # Step 3: LLM Refinement using Gemini 
    try:
        print("Performing LLM refinement with Gemini...")
        # Use a consistent system prompt with clear structure to encourage deterministic responses
        prompt = f"""
        You are a precise resume analysis system that provides objective, factual scoring. 
        Analyze the following resume and job description for match scoring purposes.
        
        Resume:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Please provide a detailed analysis with the following components:
        
        1. Technical Skills Analysis:
           - Identify specific technical skills mentioned in the job description (like programming languages, tools, frameworks)
           - Identify specific technical skills mentioned in the resume
           - Calculate a match score (0-1) based on exact presence and semantic similarity between skills
           - Consider only actual technical skills, not general terms or concepts
           - Base your scoring on objective evidence in the texts
    
        2. Soft Skills Analysis:
           - Identify specific soft skills mentioned in the job description
           - Identify specific soft skills mentioned in the resume
           - Calculate a match score (0-1) based on exact presence and semantic similarity
           - Consider only actual soft skills, not general terms or concepts
           - Base your scoring on objective evidence in the texts
    
        3. Experience Analysis:
           - Identify specific years of experience required in the job description
           - Identify specific years of experience mentioned in the resume
           - Calculate a match score (0-1) using these clear metrics:
             * If job requires X years and resume shows X or more years: 1.0
             * If job requires X years and resume shows Y years (Y < X): Y/X
             * If no specific experience requirement: 0.8 if resume shows relevant experience, 0.5 if unclear
    
        4. Education Analysis:
           - Identify specific education requirements in the job description (degrees, fields)
           - Identify specific education mentioned in the resume
           - Calculate a match score (0-1) that reflects exact fulfillment of requirements
           - If no education requirements specified, score 1.0 if any formal education is present, 0.5 if unclear
    
        5. Keyword Match Analysis:
           - Identify important domain-specific keywords in the job description 
           - Calculate a match score (0-1) for keyword overlap with the resume
           - Consider only meaningful keywords, not common words
           - Base score on percentage of important keywords present in resume
        
        Format your response as a JSON object with these exact keys:
        {{
            "technical_skills_match": <float between 0-1>,
            "soft_skills_match": <float between 0-1>,
            "experience_match": <float between 0-1>,
            "education_match": <float between 0-1>,
            "keyword_score": <float between 0-1>,
            "technical_skills_job": [list of strings],
            "technical_skills_resume": [list of strings],
            "soft_skills_job": [list of strings],
            "soft_skills_resume": [list of strings],
            "explanation": "<brief explanation of overall match>"
        }}
        
        Important instructions for consistent scoring:
        1. Be conservative in your scoring - if something is unclear, score it lower rather than assuming a match
        2. All scores MUST be between 0.0 and 1.0
        3. Base all scoring on objective evidence from the text, not assumptions
        4. For every score component, document the exact evidence you found in the explanation
        5. If a requested skill/requirement isn't mentioned at all in the resume, its match score should be 0.0 
        6. Provide precise lists of actual skills found, not general categories
        7. Use the same scoring methodology consistently for all resume evaluations
        """
        
        # Add generation parameters to increase determinism
        generation_config = {
            "temperature": 0.1,  # Very low temperature for more deterministic outputs
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        # Initialize default analysis values
        tech_skills_match = similarity_score
        soft_skills_match = similarity_score
        experience_match = default_score
        education_match = default_score
        keyword_score = keyword_match_rate  # Use the TF-IDF score as a base
        
        # Try to get a consistent response from the LLM
        response = model.generate_content(prompt, generation_config=generation_config)
        analysis_text = response.text
        
        # Remove any markdown code block markers that might be in the response
        analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
        
        # Parse the JSON response
        analysis = json.loads(analysis_text)
        
        # Extract and validate scores with consistent logic
        tech_skills_match = float(max(0.0, min(1.0, analysis.get("technical_skills_match", default_score))))
        soft_skills_match = float(max(0.0, min(1.0, analysis.get("soft_skills_match", default_score))))
        experience_match = float(max(0.0, min(1.0, analysis.get("experience_match", default_score))))
        education_match = float(max(0.0, min(1.0, analysis.get("education_match", default_score))))
        
        # Use our TF-IDF score as a factor in the final keyword score with fixed weights
        llm_keyword_score = float(max(0.0, min(1.0, analysis.get("keyword_score", default_score))))
        keyword_score = (keyword_match_rate * 0.4) + (llm_keyword_score * 0.6)  # Weighted combination
        
        # Extract skill lists with consistent sorting
        tech_skills_job = sorted(analysis.get('technical_skills_job', []))
        tech_skills_resume = sorted(analysis.get('technical_skills_resume', []))
        soft_skills_job = sorted(analysis.get('soft_skills_job', []))
        soft_skills_resume = sorted(analysis.get('soft_skills_resume', []))
        
        # Enhance the resume_keywords list with recognized skills from LLM
        # Prioritize skills identified by the LLM as they're more likely to be relevant
        resume_keywords = sorted(list(set(resume_keywords + tech_skills_resume + soft_skills_resume)))
        
        # Debug logging
        print(f"Technical Skills in Job: {tech_skills_job}")
        print(f"Technical Skills in Resume: {tech_skills_resume}")
        print(f"Soft Skills in Job: {soft_skills_job}")
        print(f"Soft Skills in Resume: {soft_skills_resume}")
        print(f"Experience Keywords: {experience_keywords}")
        print(f"Explanation: {analysis.get('explanation', 'No explanation provided')}")
        
        # Adjust scores based on skills presence using deterministic logic
        has_tech_requirements = len(tech_skills_job) > 0
        has_soft_requirements = len(soft_skills_job) > 0
        
        # If no skills found in job, use semantic similarity as fallback
        if not has_tech_requirements:
            tech_skills_match = similarity_score
        elif len(tech_skills_resume) == 0:
            tech_skills_match = 0.0  # No technical skills in resume
            
        if not has_soft_requirements:
            soft_skills_match = similarity_score
        elif len(soft_skills_resume) == 0:
            soft_skills_match = 0.0  # No soft skills in resume
            
        # Collect keywords for UI display with deterministic sorting
        all_job_keywords = set()
        all_job_keywords.update(tech_skills_job)
        all_job_keywords.update(soft_skills_job)
        
        all_resume_keywords = set()
        all_resume_keywords.update(tech_skills_resume)
        all_resume_keywords.update(soft_skills_resume)
        
        # Find matching and missing keywords with consistent sorting
        keywords_found = sorted(list(all_job_keywords.intersection(all_resume_keywords)))
        keywords_missing = sorted(list(all_job_keywords - all_resume_keywords))
            
    except Exception as e:
        print(f"Error in AI analysis: {e}")
        # Keep the default values but factor in the TF-IDF score
        keyword_score = keyword_match_rate
    
    # Calculate final score with weighted components
    # Fixed weights for consistent scoring
    weights = {
        "semantic": 0.20,      # BERT-based embedding similarity
        "keyword": 0.20,       # TF-IDF + LLM keyword matching
        "tech_skills": 0.25,   # Technical skills
        "soft_skills": 0.10,   # Soft skills
        "experience": 0.15,    # Experience
        "education": 0.10      # Education
    }
    
    # Adjust weights based on job requirements with consistent logic
    if 'analysis' in locals():
        has_tech_requirements = len(analysis.get('technical_skills_job', [])) > 0
        has_soft_requirements = len(analysis.get('soft_skills_job', [])) > 0
        
        # If technical skills aren't emphasized, redistribute weight consistently
        if not has_tech_requirements:
            extra = weights["tech_skills"] * 0.8  # Reduce tech weight by 80%
            weights["tech_skills"] *= 0.2         # Keep 20% of original weight
            
            # Redistribute the extra weight with fixed proportions
            weights["semantic"] += extra * 0.4    # 40% to semantic
            weights["keyword"] += extra * 0.3     # 30% to keyword
            weights["experience"] += extra * 0.3  # 30% to experience
        
        # If soft skills aren't emphasized, redistribute weight consistently
        if not has_soft_requirements:
            extra = weights["soft_skills"] * 0.8  # Reduce soft weight by 80%
            weights["soft_skills"] *= 0.2         # Keep 20% of original weight
            
            # Redistribute the extra weight with fixed proportions
            weights["semantic"] += extra * 0.5    # 50% to semantic
            weights["experience"] += extra * 0.5  # 50% to experience
    
    # Normalize weights to ensure they sum to 1.0
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] /= total_weight
    
    # Calculate weighted score with consistent approach
    combined_score = (
        (weights["semantic"] * similarity_score) +
        (weights["keyword"] * keyword_score) +
        (weights["tech_skills"] * tech_skills_match) +
        (weights["soft_skills"] * soft_skills_match) +
        (weights["experience"] * experience_match) +
        (weights["education"] * education_match)
    )
    
    # Ensure the score is within valid range [0, 1]
    combined_score = max(0.0, min(1.0, combined_score))
    
    # Calculate percentage match (0-100%) and round to 2 decimal places for consistency
    percentage_match = round(combined_score * 100, 2)
    
    # Debug information
    print(f"Final Weights: {weights}")
    print(f"Component Scores:")
    print(f"  - BERT Semantic={similarity_score:.2f}")
    print(f"  - TF-IDF/Keyword={keyword_score:.2f}")
    print(f"  - Tech Skills={tech_skills_match:.2f}")
    print(f"  - Soft Skills={soft_skills_match:.2f}")
    print(f"  - Experience={experience_match:.2f}")
    print(f"  - Education={education_match:.2f}")
    print(f"Combined Score: {combined_score:.2f}")
    print(f"Percentage Match: {percentage_match:.2f}%")
    
    # Update enhanced scores for UI display (removed overall_relevance)
    enhanced_scores = {
        "technical_skills": round(tech_skills_match * 100, 1),
        "soft_skills": round(soft_skills_match * 100, 1),
        "experience": round(experience_match * 100, 1),
        "education": round(education_match * 100, 1),
        "keyword_match": round(keyword_score * 100, 1)
    }
    
    # Store the result in session state for consistency in future calculations
    result = (percentage_match, enhanced_scores, keywords_found, keywords_missing, resume_keywords)
    st.session_state.previous_calculations[cache_key] = result
    
    return result

def optimize_resume(resume_text, job_description):
    prompt = f"""
    You are an expert resume optimization AI. Your task is to REWRITE and IMPROVE the provided resume to better match the job description. 
    
    DO NOT just provide suggestions - directly implement all improvements in a fully rewritten resume.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Follow these specific instructions:
    1. Maintain the same general structure and contact information from the original resume
    2. Add relevant keywords from the job description
    3. Strengthen achievement statements with metrics where appropriate
    4. Improve formatting and organization for better readability
    5. Tailor experience descriptions to highlight relevant skills for the job
    6. Remove or downplay irrelevant information
    7. Ensure all content is factual and based only on information in the original resume
    8. Ensure Accuracy & Authenticity: Do not add fictitious information—only refine and restructure based on the original resume content.
    9. Replace all '*' bullet points with '•' for better formatting
    10. DO NOT use asterisks/stars (*) for bullet points - use proper bullet point characters (•) instead
    11. Keep section headings on their own lines (not combined with content)
    12. Format the Interests section correctly: "Interests" should be a main section heading, and if there are subsections like "UI/UX", format them as clear subsections with their related interests indented below them
    13. Pay special attention to formatting the resume exactly as specified, with proper section structure
    
    Return ONLY the complete optimized resume in a clean, professional format. Do not include explanations, 
    suggestions, or commentary - just provide the fully optimized resume text ready for use.
    """
    try:
        response = model.generate_content(prompt)
        # Replace any remaining asterisks with proper bullet points
        optimized_text = response.text.replace(' * ', ' • ').replace('* ', '• ')
        
        # Ensure sections are properly separated (especially Interests and UI/UX)
        # Replace sections that might be incorrectly merged
        section_fixes = [
            ("Interests • ", "Interests\n• "),
            ("Interests•", "Interests\n•"),
            ("UI/UX • ", "UI/UX\n• "),
            ("UI/UX•", "UI/UX\n•")
        ]
        
        for find, replace in section_fixes:
            optimized_text = optimized_text.replace(find, replace)
        
        # Ensure each section header is on its own line
        sections = ["Summary", "Education", "Experience", "Skills", "Projects", "Certifications", "Languages", "Interests", "UI/UX"]
        for section in sections:
            # Add newline before section if it's not already there
            pattern = r'([^\n]){0}'.format(section)
            optimized_text = re.sub(pattern, r'\1\n{0}'.format(section), optimized_text)
        
        return optimized_text
    except Exception as e:
        print("Error generating optimized resume:", e)
        return "Error: Unable to generate optimized resume. Please try again."

def analyze_low_matching(resume_text, job_description):
    prompt = f"""
    As an expert resume analyst, provide a comprehensive analysis of how this specific resume aligns with the job description.
    You must thoroughly understand both the resume content and the job requirements before providing feedback.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Your analysis should include:
    
    1. Resume Structure and Content Analysis:
       - Evaluate the overall format, organization, and clarity of the resume
       - Assess the strength of achievement statements (quantifiable results vs. vague descriptions)
       - Identify any content gaps or sections that need improvement
    
    2. Skills Gap Analysis:
       - Identify technical skills required in the job description that are missing or underdeveloped in the resume
       - Identify soft skills required in the job description that are missing or underdeveloped in the resume
       - Evaluate how effectively existing skills are presented and contextualized
    
    3. Experience Relevance Analysis:
       - Evaluate how well the candidate's experience aligns with job requirements
       - Identify specific experience gaps that could be addressed
       - Assess whether experience descriptions emphasize the right achievements for this role
    
    4. Education and Credentials Analysis:
       - Evaluate if the education and certifications meet the job requirements
       - Identify any missing credentials that would strengthen the application

    5. ATS Optimization Analysis:
       - Identify missing keywords that could improve ATS scoring
       - Assess formatting issues that might impair ATS readability
    
    Provide your response in the following structured format:
    
    RESUME ASSESSMENT:
    - Overall Quality: Provide a brief assessment of the resume's overall quality
    - Key Strengths: List 2-3 strongest elements of the resume relative to this job
    - Critical Gaps: List 2-3 most significant gaps or weaknesses relative to this job
    
    DETAILED REASONS:
    - Reason 1: (detailed explanation of a key alignment issue)
    - Reason 2: (detailed explanation of another key alignment issue)
    - Reason 3: (detailed explanation of another key alignment issue if applicable)
    
    ACTIONABLE RECOMMENDATIONS:
    - Recommendation 1: (specific, actionable advice for improvement)
    - Recommendation 2: (specific, actionable advice for improvement)
    - Recommendation 3: (specific, actionable advice for improvement)
    - Recommendation 4: (specific, actionable advice for improvement if applicable)
    
    TAILORED KEYWORDS TO ADD:
    - List 5-8 specific keywords or phrases from the job description that should be incorporated
    
    Your analysis must be specific to this resume and job - avoid generic feedback that could apply to any resume.
    Focus on providing insightful, personalized analysis that shows you understand this specific resume and job.
    """
    try:
        response = model.generate_content(prompt)
        # Store the full API response for display in the UI
        full_response = response.text
        
        # Extract sections for structured display
        reasons = []
        suggestions = []
        
        # Extract different sections based on headers
        assessment_section = False
        reasons_section = False
        recommendations_section = False
        keywords_section = False
        
        for line in response.text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if "RESUME ASSESSMENT:" in line.upper():
                assessment_section = True
                reasons_section = False
                recommendations_section = False
                keywords_section = False
                continue
                
            if "DETAILED REASONS:" in line.upper() or "REASONS:" in line.upper():
                assessment_section = False
                reasons_section = True
                recommendations_section = False
                keywords_section = False
                continue
            
            if "ACTIONABLE RECOMMENDATIONS:" in line.upper() or "SUGGESTIONS:" in line.upper() or "RECOMMENDATIONS:" in line.upper():
                assessment_section = False
                reasons_section = False
                recommendations_section = True
                keywords_section = False
                continue
                
            if "TAILORED KEYWORDS" in line.upper():
                assessment_section = False
                reasons_section = False
                recommendations_section = False
                keywords_section = True
                continue
            
            # Extract content from each section
            if reasons_section and line.startswith('-'):
                reasons.append(line.lstrip('- '))
            
            if recommendations_section and line.startswith('-'):
                suggestions.append(line.lstrip('- '))
        
        # Fallback content if API response is empty or parsing failed
        if not reasons:
            reasons = ["The resume lacks specific keywords from the job description.", 
                       "The experience section does not align well with the job requirements.",
                       "The resume structure may not be optimized for ATS systems."]
        if not suggestions:
            suggestions = ["Consider adding more relevant keywords from the job description.", 
                           "Expand on your experience to better match the job requirements.",
                           "Quantify your achievements with specific metrics and outcomes.",
                           "Refine formatting for better ATS compatibility."]
        
    except Exception as e:
        print("Error during API call:", e)
        full_response = "Error in generating analysis."
        reasons = ["Error in generating detailed resume analysis."]
        suggestions = ["Please try again or contact support for assistance."]
    
    return reasons, suggestions, full_response

def format_analysis(text):
    """Format analysis text for better readability in HTML, using enhanced styling.
    
    Args:
        text (str): The raw analysis text from the AI model
        
    Returns:
        str: HTML formatted text with enhanced styling
    """
    if not text:
        return ""
    
    # Replace section headers with styled headers
    for pattern in [r'(?m)^#\s+(.*?)$', r'(?m)^##\s+(.*?)$', r'(?m)^###\s+(.*?)$']:
        text = re.sub(pattern, r'<div class="section-header">\1</div>', text)
    
    # Format bullet points with enhanced styling
    text = re.sub(r'(?m)^[*•-]\s+(.*?)$', r'<div class="bullet-point"><span class="bullet"></span><span class="bullet-content">\1</span></div>', text)
    
    # Handle multi-line bullet points (maintain indentation)
    text = re.sub(r'(?m)^(\s{2,})(.*?)$', r'<div style="margin-left: 18px;">\2</div>', text)
    
    # Format paragraphs only if they are substantial (not headers or breaks)
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    for p in paragraphs:
        # Skip empty paragraphs
        if not p.strip():
            continue
            
        # If paragraph doesn't contain a header or bullet already, wrap it in paragraph tags
        if not re.search(r'<div class="section-header"|<div class="bullet-point"', p):
            # Only wrap substantial text in paragraph tags (more than just a few characters)
            if len(p.strip()) > 5 and not p.strip().startswith('<div'):
                p = f'<p>{p}</p>'
        
        formatted_paragraphs.append(p)
    
    return ''.join(formatted_paragraphs)

def load_css():
    # Load external CSS file
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def batch_processing_ui():
    # Add app header with proper styling for batch processing
    st.markdown("""
    <div class="batch-header">
        <div class="batch-title">Batch Resume Processing</div>
        <div class="batch-description">Upload multiple resumes to compare them against a job description and rank the best matches.</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Upload multiple resumes", type=["pdf", "docx"], accept_multiple_files=True)
    job_description = st.text_area("Paste the job description here", key="batch_job_description", height=200)
    
    if uploaded_files and job_description:
        # Process uploaded files
        with st.spinner("Processing resumes..."):
            batch_results = []
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                
                # Extract text from resume
                resume_text = upload_and_parse_resume(uploaded_file)
                
                if resume_text:
                    # Calculate match scores
                    percentage_match, enhanced_scores, keywords_found, keywords_missing, resume_keywords = calculate_match_score(resume_text, job_description)
                    
                    # Get analysis and optimized resume up front
                    reasons, suggestions, full_response = analyze_low_matching(resume_text, job_description)
                    formatted_response = format_analysis(full_response)
                    
                    optimized_resume = optimize_resume(resume_text, job_description)
                    
                    # Determine classification
                    if percentage_match > 80:
                        classification = "Highly Matching"
                        badge_class = "high-match"
                    elif percentage_match > 50:
                        classification = "Medium Matching"
                        badge_class = "medium-match"
                    else:
                        classification = "Low Matching"
                        badge_class = "low-match"
                    
                    # Add to results
                    batch_results.append({
                        "file_name": uploaded_file.name,
                        "score": percentage_match,
                        "classification": classification,
                        "badge_class": badge_class,
                        "enhanced_scores": enhanced_scores,
                        "keywords_found": keywords_found,
                        "keywords_missing": keywords_missing,
                        "resume_keywords": resume_keywords,
                        "resume_text": resume_text,
                        "analysis": formatted_response,
                        "optimized_resume": optimized_resume
                    })
            
            # Remove progress bar after completion
            progress_bar.empty()
        
        # Sort results by score (highest first)
        batch_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Display summary statistics
        st.markdown('<div class="section-header">Ranking Summary</div>', unsafe_allow_html=True)
        
        # Count resumes by category
        high_matches = sum(1 for result in batch_results if result["classification"] == "Highly Matching")
        medium_matches = sum(1 for result in batch_results if result["classification"] == "Medium Matching")
        low_matches = sum(1 for result in batch_results if result["classification"] == "Low Matching")
        
        # Create a 3-column display for summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="high-match" style="text-align: center; padding: 15px; border-radius: 8px;">
                <div style="font-size: 2.5rem; font-weight: 700;">{high_matches}</div>
                <div>High Matches</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="medium-match" style="text-align: center; padding: 15px; border-radius: 8px;">
                <div style="font-size: 2.5rem; font-weight: 700;">{medium_matches}</div>
                <div>Medium Matches</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="low-match" style="text-align: center; padding: 15px; border-radius: 8px;">
                <div style="font-size: 2.5rem; font-weight: 700;">{low_matches}</div>
                <div>Low Matches</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display individual results
        st.markdown('<div class="section-header">Ranked Results</div>', unsafe_allow_html=True)
        
        for i, result in enumerate(batch_results, 1):
            with st.expander(f"#{i}: {result['file_name']} - {result['score']:.2f}% ({result['classification']})"):
                # Resume details in a card
                st.markdown(f"""
                <div class="resume-card">
                    <div class="resume-title">
                        {result['file_name']} 
                        <span class="score-badge {result['badge_class']}">Score: {result['score']:.2f}%</span>
                    </div>
                    <div class="classification">{result['classification']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed scores
                st.markdown('<div class="section-header">Detailed Scores</div>', unsafe_allow_html=True)
                score_html = "<div class='dark-section'>"
                for category, score in result['enhanced_scores'].items():
                    # Convert category from snake_case to Title Case
                    category_name = ' '.join(word.capitalize() for word in category.split('_'))
                    score_html += f"<p><strong>{category_name}:</strong> {score:.0f}%</p>"
                score_html += "</div>"
                st.markdown(score_html, unsafe_allow_html=True)
                
                # Keywords sections
                col1, col2 = st.columns(2)
                
                with col1:
                    # Resume keywords
                    if result['resume_keywords']:
                        st.markdown('<div class="section-header">Resume Keywords</div>', unsafe_allow_html=True)
                        # Limit to most relevant keywords (up to 30)
                        display_keywords = result['resume_keywords'][:30] if len(result['resume_keywords']) > 30 else result['resume_keywords']
                        st.markdown(f"<div class='dark-section'>{', '.join(display_keywords)}</div>", 
                                  unsafe_allow_html=True)
                
                    # Keywords found (moved below resume keywords)
                    if result['keywords_found']:
                        st.markdown('<div class="section-header">Job Keywords Found</div>', unsafe_allow_html=True)
                        st.markdown(f"<div class='dark-section'>{', '.join(result['keywords_found'])}</div>", 
                                  unsafe_allow_html=True)
                
                with col2:
                    if result['keywords_missing']:
                        st.markdown('<div class="section-header">Job Keywords Missing</div>', unsafe_allow_html=True)
                        st.markdown(f"<div class='dark-section'>{', '.join(result['keywords_missing'])}</div>", 
                                  unsafe_allow_html=True)
                
                # Display detailed analysis (now automatically shown)
                st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="dark-section">
                    {result['analysis']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display optimized resume (now automatically shown)
                st.markdown('<div class="section-header">Optimized Resume</div>', unsafe_allow_html=True)
                
                # Add notification about optimized resume
                st.markdown("""
                <div class="notification-box">
                    This AI-optimized version of the resume has been tailored to better match the job description.
                    Key improvements include enhanced keyword matching and clearer presentation of relevant skills.
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="dark-section">
                    {result['optimized_resume']}
                </div>
                """, unsafe_allow_html=True)

def generate_resume_pdf(resume_text):
    """
    Converts the optimized resume text to a professionally formatted PDF
    
    Args:
        resume_text (str): The optimized resume text
        
    Returns:
        bytes: PDF file as bytes
    """
    try:
        # Create PDF object with updated FPDF API usage
        pdf = FPDF()
        pdf.add_page()
        
        # Use standard fonts with correct API usage (no deprecation warnings)
        pdf.set_font("Helvetica", style="B", size=16)
        
        # Enable auto page breaks
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add a title with updated API usage
        pdf.cell(w=0, h=10, txt="Optimized Resume", align='C')
        pdf.ln(10)
        
        # Process each line of the resume
        if resume_text:
            lines = resume_text.split('\n')
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    pdf.ln(3)
                    continue
                
                # Clean the text to remove problematic characters
                # More aggressive filtering for better compatibility
                clean_line = ''
                for c in line:
                    if ord(c) < 128:  # Keep only ASCII characters
                        clean_line += c
                    else:
                        clean_line += ' '  # Replace non-ASCII with space
                
                # Format section headers
                if clean_line.strip().isupper() or clean_line.strip().endswith(':'):
                    pdf.set_font("Helvetica", style="B", size=12)
                    if i > 1:
                        pdf.ln(5)
                    pdf.cell(w=0, h=6, txt=clean_line)
                    pdf.ln()
                else:
                    pdf.set_font("Helvetica", size=10)
                    # Break long lines into shorter segments to avoid overflow
                    if len(clean_line) > 80:
                        segments = [clean_line[i:i+80] for i in range(0, len(clean_line), 80)]
                        for segment in segments:
                            pdf.multi_cell(w=0, h=5, txt=segment)
                    else:
                        pdf.multi_cell(w=0, h=5, txt=clean_line)
        
        # Return the PDF as bytes with updated API
        return pdf.output()
    except Exception as e:
        print(f"Detailed PDF generation error: {str(e)}")
        
        # Create a very simple error PDF without any risky operations
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", style="B", size=16)
            pdf.cell(w=0, h=10, txt="Basic Resume")
            pdf.ln(10)
            pdf.set_font("Helvetica", size=12)
            # Very basic content that should not fail
            pdf.multi_cell(w=0, h=5, txt="PDF generation failed. Basic version provided.")
            return pdf.output()
        except Exception as inner_e:
            print(f"Even fallback PDF failed: {str(inner_e)}")
            # If even the fallback fails, raise the exception to trigger HTML download
            raise

def create_downloadable_html(resume_text):
    """
    Creates a downloadable HTML version of the resume
    
    Args:
        resume_text (str): The optimized resume text
        
    Returns:
        str: HTML string with download link
    """
    # Convert resume text to HTML format
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Optimized Resume</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ text-align: center; color: #2a5298; }}
            h2 {{ color: #2a5298; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            .section {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Optimized Resume</h1>
        <div class="content">
    """
    
    # Process each line of the resume
    lines = resume_text.split('\n')
    in_paragraph = False
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            if in_paragraph:
                html_content += "</p>\n"
                in_paragraph = False
            continue
        
        # Check if line is a section header
        if line.strip().isupper() or line.strip().endswith(':'):
            if in_paragraph:
                html_content += "</p>\n"
                in_paragraph = False
            html_content += f"<h2>{line}</h2>\n"
        else:
            if not in_paragraph:
                html_content += "<p>"
                in_paragraph = True
            html_content += f"{line}<br>\n"
    
    # Close any open paragraph
    if in_paragraph:
        html_content += "</p>\n"
    
    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Encode HTML for download
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="optimized_resume.html">Download as HTML</a>'
    
    return href

def generate_resume_pdf_reportlab(resume_text):
    """
    Creates a professionally formatted PDF of the resume using ReportLab
    with enhanced styling for a more aesthetic appearance
    
    Args:
        resume_text (str): The optimized resume text
        
    Returns:
        bytes: PDF file as bytes
    """
    try:
        # Create a buffer to receive PDF data
        buffer = io.BytesIO()
        
        # Create the PDF document with letter size
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        # Create styles
        styles = getSampleStyleSheet()
        
        # Create custom styles for resume sections
        name_style = ParagraphStyle(
            'Name',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=1,  # Center alignment
            spaceAfter=6,
            textColor=colors.Color(0.1, 0.3, 0.5),  # Professional dark blue
            fontName='Helvetica-Bold'
        )
        
        contact_style = ParagraphStyle(
            'Contact',
            parent=styles['Normal'],
            fontSize=10,
            alignment=1,  # Center alignment
            spaceAfter=2,
            leading=14
        )
        
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=colors.Color(0.1, 0.3, 0.5),  # Professional dark blue
            spaceAfter=6,
            spaceBefore=12,
            borderWidth=0,
            borderColor=colors.Color(0.1, 0.3, 0.5),
            borderPadding=5,
            leading=14,
            leftIndent=0,
            fontName='Helvetica-Bold'
        )
        
        subsection_style = ParagraphStyle(
            'SubSectionHeader',
            parent=styles['Heading3'],
            fontSize=11,
            textColor=colors.Color(0.2, 0.4, 0.6),  # Slightly lighter blue
            spaceAfter=3,
            spaceBefore=6,
            leading=14,
            leftIndent=0.1*inch,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            leading=14,
            leftIndent=0.2*inch
        )
        
        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            bulletIndent=0.2*inch,
            leftIndent=0.4*inch,
            leading=14
        )
        
        # Container for content elements
        elements = []
        
        # Pre-process the resume text to fix common issues
        # 1. Replace any remaining asterisks with bullet points
        resume_text = resume_text.replace('* ', '• ').replace('*•', '•')
        
        # 2. Ensure section headers are properly separated
        section_headers = ['Summary', 'Education', 'Experience', 'Skills', 'Projects', 
                          'Certifications', 'Languages', 'Interests', 'UI/UX']
        
        for header in section_headers:
            # Ensure there's a newline before each section header
            pattern = r'([^\n])(' + header + r')(\s*?\n)'
            resume_text = re.sub(pattern, r'\1\n\2\3', resume_text)
        
        # Process each line of the resume
        if resume_text:
            # Split into lines and clean them
            lines = resume_text.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    # Remove markdown formatting and ensure proper bullet points
                    clean_line = line.replace('**', '').replace('* ', '• ')
                    clean_lines.append(clean_line)
            
            # Detect structure based on content patterns
            is_first_section = True
            current_section = ""
            
            i = 0
            while i < len(clean_lines):
                line = clean_lines[i]
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Handle the name (first line)
                if i == 0:
                    elements.append(Paragraph(line, name_style))
                    i += 1
                    continue
                
                # Handle contact info (lines after name until first section header)
                if is_first_section:
                    # Check if this line is a section header
                    is_section_header = False
                    for header in section_headers:
                        if line.lower().strip() == header.lower():
                            is_section_header = True
                            is_first_section = False
                            current_section = header
                            break
                    
                    if not is_section_header:
                        # This is still contact info
                        elements.append(Paragraph(line, contact_style))
                        i += 1
                        continue
                
                # Handle section headers
                is_section_header = False
                for header in section_headers:
                    if line.lower().strip() == header.lower():
                        is_section_header = True
                        current_section = header
                        
                        # Add section header with proper styling
                        elements.append(Paragraph(line, section_style))
                        
                        # Add horizontal line under section headers
                        elements.append(Spacer(1, 2))
                        elements.append(HRFlowable(
                            width="100%",
                            thickness=1,
                            color=colors.Color(0.1, 0.3, 0.5),
                            spaceBefore=0,
                            spaceAfter=6
                        ))
                        
                        i += 1
                        break
                
                if is_section_header:
                    continue
                
                # Handle subsections (like UI/UX under Interests)
                # For UI/UX specifically, check if we're in the Interests section
                if current_section.lower() == "interests" and line.lower().strip() == "ui/ux":
                    elements.append(Paragraph(line, subsection_style))
                    i += 1
                    continue
                
                # Handle bullet points
                if line.startswith('•'):
                    # Fix any double bullets
                    if line.startswith('••'):
                        line = '•' + line[2:]
                    
                    # Clean up spaces after bullet
                    if line.startswith('• '):
                        content = line[2:]
                    else:
                        content = line[1:]
                    
                    # Create properly formatted bullet point
                    bullet_text = '• ' + content
                    elements.append(Paragraph(bullet_text, bullet_style))
                    i += 1
                    continue
                
                # Handle normal text lines
                # Check if this might be a subsection header (followed by bullets)
                if i + 1 < len(clean_lines) and clean_lines[i + 1].startswith('•'):
                    elements.append(Paragraph(line, subsection_style))
                else:
                    elements.append(Paragraph(line, normal_style))
                
                i += 1
        
        # Build the PDF
        doc.build(elements)
        
        # Get the value from the buffer
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    except Exception as e:
        print(f"ReportLab PDF generation error: {str(e)}")
        # If ReportLab fails, try the basic FPDF approach as fallback
        try:
            return generate_resume_pdf(resume_text)
        except Exception as inner_e:
            print(f"Fallback PDF generation also failed: {str(inner_e)}")
            # If even the fallback fails, raise the exception to trigger the HTML fallback
            raise

def single_resume_optimization_ui():
    # Add app header with proper styling for single resume optimization
    st.markdown("""
    <div class="optimization-header">
        <div class="optimization-title">Single Resume Optimization</div>
        <div class="optimization-description">Upload your resume and compare it against a job description to get personalized improvement suggestions.</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a resume", type=["pdf", "docx"], key="single_resume")
    job_description = st.text_area("Paste the job description here", key="single_job_description", height=200)
    
    # Initialize session state for optimized resume if not already present
    if 'optimized_resume' not in st.session_state:
        st.session_state.optimized_resume = None
    if 'resume_analyzed' not in st.session_state:
        st.session_state.resume_analyzed = False
    
    if uploaded_file and job_description:
        resume_text = upload_and_parse_resume(uploaded_file)
        if resume_text:
            # Display a processing message
            with st.spinner("Analyzing your resume..."):
                # Calculate match scores
                percentage_match, enhanced_scores, keywords_found, keywords_missing, resume_keywords = calculate_match_score(resume_text, job_description)
                
                # Get analysis
                reasons, suggestions, full_response = analyze_low_matching(resume_text, job_description)
                formatted_response = format_analysis(full_response)
                
                # Mark as analyzed
                st.session_state.resume_analyzed = True
                
                # Determine classification based on score
                if percentage_match > 80:
                    classification = "Highly Matching"
                    badge_class = "high-match"
                elif percentage_match > 50:
                    classification = "Medium Matching"
                    badge_class = "medium-match"
                else:
                    classification = "Low Matching"
                    badge_class = "low-match"
            
            # Display results in an enhanced UI
            st.markdown("""
            <div class="results-container">
                <div class="section-header">Resume Analysis Results</div>
            """, unsafe_allow_html=True)
            
            # Display score with badge
            st.markdown(f"""
                <div class="resume-title">
                    {uploaded_file.name} 
                    <span class="score-badge {badge_class}">Score: {percentage_match:.2f}%</span>
                </div>
                <div class="classification">{classification}</div>
            """, unsafe_allow_html=True)
            
            # Display enhanced scores in dark section
            st.markdown('<div class="section-header">Detailed Scores</div>', unsafe_allow_html=True)
            score_html = "<div class='dark-section'>"
            for category, score in enhanced_scores.items():
                # Convert category from snake_case to Title Case
                category_name = ' '.join(word.capitalize() for word in category.split('_'))
                score_html += f"<p><strong>{category_name}:</strong> {score:.0f}%</p>"
            score_html += "</div>"
            st.markdown(score_html, unsafe_allow_html=True)
            
            # Keywords sections with three columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Resume keywords
                if resume_keywords:
                    st.markdown('<div class="section-header">Resume Keywords</div>', unsafe_allow_html=True)
                    # Limit to most relevant keywords (up to 30)
                    display_keywords = resume_keywords[:30] if len(resume_keywords) > 30 else resume_keywords
                    st.markdown(f"<div class='dark-section'>{', '.join(display_keywords)}</div>", 
                              unsafe_allow_html=True)
                
                # Keywords found (moved below resume keywords)
                if keywords_found:
                    st.markdown('<div class="section-header">Job Keywords Found</div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='dark-section'>{', '.join(keywords_found)}</div>", 
                              unsafe_allow_html=True)
            
            with col2:
                if keywords_missing:
                    st.markdown('<div class="section-header">Job Keywords Missing</div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='dark-section'>{', '.join(keywords_missing)}</div>", 
                              unsafe_allow_html=True)
            
            # Display full analysis automatically (no need for expander or button click)
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="dark-section">
                {formatted_response}
            </div>
            """, unsafe_allow_html=True)
            
            # Add Generate Optimized Resume button
            if st.button("Generate Optimized Resume"):
                with st.spinner("Generating optimized resume..."):
                    optimized_resume = optimize_resume(resume_text, job_description)
                    st.session_state.optimized_resume = optimized_resume
            
            # Display optimized resume if available
            if st.session_state.optimized_resume:
                st.markdown('<div class="section-header">Optimized Resume</div>', unsafe_allow_html=True)
                
                # Add notification about optimized resume
                st.markdown("""
                <div class="notification-box">
                    This AI-optimized version of your resume has been tailored to better match the job description.
                    Key improvements include enhanced keyword matching and clearer presentation of relevant skills.
                </div>
                """, unsafe_allow_html=True)
                
                # Add download buttons for optimized resume
                download_col1, download_col2, download_col3 = st.columns([2, 1, 1])
                
                with download_col2:
                    # Convert optimized resume to a downloadable format (text)
                    resume_txt = st.session_state.optimized_resume
                    st.download_button(
                        label="Download as TXT",
                        data=resume_txt,
                        file_name="optimized_resume.txt",
                        mime="text/plain"
                    )
                
                with download_col3:
                    # Try both PDF options: ReportLab first, then HTML fallback
                    try:
                        pdf_bytes = generate_resume_pdf_reportlab(st.session_state.optimized_resume)
                        st.download_button(
                            label="Download as PDF",
                            data=pdf_bytes,
                            file_name="optimized_resume.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        # If PDF generation fails, provide the HTML download option
                        st.error("PDF generation failed. Using HTML option instead.")
                        print(f"PDF error details: {str(e)}")
                        # Add HTML download option as fallback
                        html_download = create_downloadable_html(st.session_state.optimized_resume)
                        st.markdown("<div style='text-align: center;'><strong>HTML Download:</strong></div>", unsafe_allow_html=True)
                        st.markdown(html_download, unsafe_allow_html=True)
                
                # Display the optimized resume with improved styling
                st.markdown(f"""
                <div class="dark-section">
                    {st.session_state.optimized_resume}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close results container
            
            # Log to terminal for debugging
            print(f"Resume: {uploaded_file.name}, Score: {percentage_match}, Classification: {classification}")

def settings_ui():
    """Settings tab UI for configuring the API key and other settings"""
    st.markdown("""
    <div class="settings-header">
        <div class="settings-title">Application Settings</div>
        <div class="settings-description">Configure your API keys and application preferences</div>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key configuration section
    st.markdown('<div class="section-header">API Configuration</div>', unsafe_allow_html=True)
    
    # Display current API key status
    if st.session_state.api_key_configured:
        st.success("✅ Gemini API is configured and working")
    else:
        st.warning("⚠️ Gemini API key not configured. Please enter your API key below.")
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key", 
        value=st.session_state.api_key,
        type="password",
        key="api_key_input",
        help="Enter your Gemini API key. Get one at https://ai.google.dev/"
    )
    
    # Save API key button
    if st.button("Save API Key", key="save_api_key_btn"):
        if api_key:
            # Store the API key in session state
            st.session_state.api_key = api_key
            
            # Initialize models with the new API key
            if initialize_models():
                st.session_state.api_key_configured = True
                st.success("✅ API key saved and validated successfully!")
                # Force a rerun to update the UI immediately
                st.rerun()
            else:
                st.error("❌ API key could not be validated. Please check your key and try again.")
        else:
            st.error("❌ Please enter an API key.")
    
    # Instructions for getting an API key
    with st.expander("How to get a Gemini API key"):
        st.markdown("""
        1. Go to [Google AI Studio](https://ai.google.dev/)
        2. Sign in with your Google account
        3. Navigate to API keys in your settings or dashboard
        4. Create a new API key
        5. Copy the key and paste it here
        """)
    
    # Advanced settings section
    st.markdown('<div class="section-header">Advanced Settings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Model selection (for future use)
        model_option = st.selectbox(
            "Gemini Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro"],
            index=0,
            help="Choose which Gemini model to use for analysis"
        )
    
    with col2:
        # Score thresholds
        high_match_threshold = st.slider(
            "High Match Threshold (%)",
            min_value=60,
            max_value=95,
            value=80,
            step=5,
            help="Score threshold for high matches"
        )

def main_ui():
    # Load external CSS
    try:
        load_css()
    except Exception as e:
        st.warning(f"Could not load external CSS: {e}")
        # Fall back to inline CSS
        st.markdown("""
        <style>
        .app-header {background: linear-gradient(to right, #1e3c72, #2a5298); color: white; padding: 20px; text-align: center; border-radius: 10px;}
        .app-title {font-size: 2.5rem; font-weight: 700; margin-bottom: 10px;}
        .dark-section {background-color: #1e1e1e; color: white; padding: 15px; border-radius: 5px; margin: 10px 0;}
        .section-header {background-color: #2a5298; color: white; padding: 10px; border-radius: 5px; margin: 15px 0 10px 0; font-weight: 600;}
        </style>
        """, unsafe_allow_html=True)
    
    # Check if API key is configured
    if not st.session_state.api_key_configured:
        st.warning("⚠️ Gemini API key not configured. Please go to Settings to set up your API key.")
    
    # Add app header with title and description
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Resume Rank & Optimization</div>
        <div class="app-description">
            Upload resumes, compare them with job descriptions, and get AI-powered optimization suggestions to improve your chances of landing your dream job.
            <ul style="color: white; text-align: left; max-width: 800px; margin: 10px auto; list-style-position: inside;">
                <li>Match percentage analysis</li>
                <li>Detailed feedback on alignment</li>
                <li>Tailored improvement recommendations</li>
                <li>Resume optimization suggestions</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs with custom styling
    tab1, tab2, tab3 = st.tabs(["Batch Processing", "Single Resume Optimization", "Settings"])
    
    with tab1:
        if st.session_state.api_key_configured:
            batch_processing_ui()
        else:
            st.info("Please configure your Gemini API key in the Settings tab before using this feature.")
    
    with tab2:
        if st.session_state.api_key_configured:
            single_resume_optimization_ui()
        else:
            st.info("Please configure your Gemini API key in the Settings tab before using this feature.")
    
    with tab3:
        settings_ui()
    
    # Add footer with custom styling
    st.markdown("""
    <div class="footer">
        Resume Rank & Optimization Tool © 2025 | Powered by AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_ui()
