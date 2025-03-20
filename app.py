import PyPDF2
import docx
import re
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import streamlit as st
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configure Gemini API
genai.configure(api_key="AIzaSyCQM4lEwFRf5N6XBs21Of4FyMmouo8g00A")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Common technical terms and job-related phrases
COMMON_TECH_TERMS = {
    'python', 'java', 'javascript', 'js', 'typescript', 'ts', 'c++', 'ruby', 'php', 'swift', 'kotlin', 'react', 
    'angular', 'vue', 'node', 'django', 'flask', 'spring', 'express', 'tensorflow', 'pytorch', 'aws', 'azure', 
    'gcp', 'docker', 'kubernetes', 'k8s', 'ci/cd', 'git', 'github', 'gitlab', 'jenkins', 'sql', 'nosql', 
    'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'rest', 'graphql', 'api', 'http', 
    'json', 'xml', 'html', 'css', 'sass', 'less', 'webassembly', 'wasm', 'microservices', 'devops', 'sre', 
    'agile', 'scrum', 'kanban', 'jira', 'confluence', 'figma', 'sketch', 'ai', 'ml', 'machine learning',
    'deep learning', 'nlp', 'computer vision', 'cv', 'data science', 'big data', 'hadoop', 'spark', 'tableau', 
    'power bi', 'etl', 'cicd', 'iot', 'blockchain', 'frontend', 'backend', 'fullstack', 'ui', 'ux', 'data analysis',
    'data analytics', 'cyber security', 'cybersecurity', 'information security', 'infosec', 'penetration testing',
    'penetration tester', 'pentesting', 'cloud computing', 'saas', 'paas', 'iaas', 'serverless', 'neural networks',
    'version control', 'object-oriented', 'oop', 'functional programming', 'web development', 'mobile development',
    'android', 'ios', 'react native', 'flutter', 'junit', 'pytest', 'test-driven', 'tdd', 'restful', 'soap',
    'algorithms', 'data structures', 'distributed systems', 'containerization', 'virtual machines', 'vm', 'linux',
    'unix', 'windows', 'macos', 'bash', 'shell', 'powershell', 'database design', 'orm'
}

# Common education terms
EDUCATION_TERMS = {
    'bachelor', 'bs', 'ba', 'b.s.', 'b.a.', 'master', 'ms', 'ma', 'm.s.', 'm.a.', 'phd', 'ph.d.', 'doctorate', 
    'mba', 'degree', 'certification', 'certificate', 'bootcamp', 'university', 'college', 'institute', 'school',
    'graduate', 'undergraduate', 'postgraduate', 'diploma', 'certified', 'licensed', 'accredited', 'coursework',
    'thesis', 'dissertation', 'academic', 'major', 'minor', 'gpa', 'cum laude', 'magna cum laude', 'summa cum laude',
    'honours', 'honors'
}

# Common soft skills
SOFT_SKILLS = {
    'communication', 'teamwork', 'collaboration', 'leadership', 'problem solving', 'time management', 'adaptability', 
    'flexibility', 'creativity', 'critical thinking', 'analytical', 'interpersonal', 'organizational', 'detail oriented',
    'multitasking', 'prioritization', 'self-motivated', 'enthusiastic', 'negotiation', 'conflict resolution', 'decision making',
    'independent', 'customer service', 'presentation', 'writing', 'verbal', 'listening', 'empathy', 'cultural awareness',
    'team player', 'mentor', 'coaching', 'feedback', 'innovation', 'entrepreneurial', 'strategic thinking', 'patience',
    'emotional intelligence', 'resilience', 'stress management', 'accountability', 'reliability', 'punctuality', 'integrity',
    'proactive', 'problem-solver', 'self-starter', 'results-driven', 'deadline-oriented', 'goal-oriented', 'resourceful',
    'relationship building', 'relationship management', 'coordination', 'project coordination', 'project management'
}

# Common job title terms
JOB_TITLES = {
    'engineer', 'developer', 'analyst', 'manager', 'specialist', 'coordinator', 'director', 'administrator', 'technician',
    'architect', 'designer', 'consultant', 'scientist', 'lead', 'head', 'chief', 'junior', 'senior', 'principal',
    'associate', 'assistant', 'executive', 'officer', 'vice president', 'president', 'intern', 'trainee', 'apprentice'
}

# Common industry domains
INDUSTRY_DOMAINS = {
    'healthcare', 'finance', 'banking', 'insurance', 'retail', 'e-commerce', 'manufacturing', 'logistics', 
    'transportation', 'education', 'government', 'non-profit', 'technology', 'telecommunications', 'media', 
    'entertainment', 'hospitality', 'real estate', 'construction', 'energy', 'oil and gas', 'agriculture', 
    'pharmaceutical', 'legal', 'consulting', 'automotive', 'aerospace', 'defense'
}

# Common noise words to filter out
NOISE_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'resume', 'cv', 'curriculum', 'vitae', 'etc', 'ie', 'eg', 'example', 'use', 'using', 'used', 'like', 'good',
    'well', 'better', 'best', 'able', 'ability', 'looking', 'look', 'year', 'years', 'month', 'months', 'day',
    'days', 'would', 'should', 'could', 'please', 'thank', 'thanks', 'one', 'two', 'three', 'four', 'five', 'page',
    'name', 'email', 'phone', 'address', 'contact', 'ref', 'reference', 'references'
}

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

def extract_keywords(text, is_job_description=False):
    """
    Extract relevant keywords from text using advanced NLP techniques.
    
    Args:
        text: The input text (resume or job description)
        is_job_description: Boolean flag to indicate if the text is a job description
        
    Returns:
        A set of extracted keywords with relevance scores
    """
    try:
        # Convert to lowercase and normalize whitespace
        text = re.sub(r'\s+', ' ', text.lower())
        
        # Extract common patterns first (before tokenization)
        # Years of experience (e.g., "5+ years", "5-7 years", "3 years of experience")
        experience_patterns = re.findall(r'\d+\+?\s+years?|\d+\s*-\s*\d+\s+years?|\d+\s+years?\s+of\s+experience', text)
        
        # Dollar amounts/salary ranges
        salary_patterns = re.findall(r'\$\d+,\d+|\$\d+k|\$\d+\s*-\s*\$\d+k?', text)
        
        # Percentages
        percentage_patterns = re.findall(r'\d+\.?\d*\s*%', text)
        
        # Certifications and degree specifiers
        certification_patterns = re.findall(r'certified\s+\w+|licensed\s+\w+|\w+\s+certification', text)
        degree_patterns = re.findall(r'(bachelor|master|phd)\'?s?\s+(of|in|degree\s+in)\s+\w+', text)
        
        # Now proceed with NLP-based extraction
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        
        # Create enhanced stopwords set using our NOISE_WORDS
        stop_words = set(stopwords.words('english')).union(NOISE_WORDS)
        
        # Filter tokens and lemmatize
        filtered_tokens = []
        for token in tokens:
            if (token.isalnum() and 
                len(token) > 2 and  # Increased minimum length to 3 chars to reduce noise
                token not in stop_words):
                lemma = lemmatizer.lemmatize(token)
                filtered_tokens.append(lemma)
        
        # Generate n-grams for multi-word terms
        bigrams = [' '.join(bg) for bg in ngrams(tokens, 2)]
        trigrams = [' '.join(tg) for tg in ngrams(tokens, 3)]
        
        # Filter n-grams that contain stopwords
        filtered_bigrams = []
        for bg in bigrams:
            words = bg.split()
            if not any(word in stop_words for word in words) and len(bg) > 5:  # Ensure bigrams are substantial
                filtered_bigrams.append(bg)
                
        filtered_trigrams = []
        for tg in trigrams:
            words = tg.split()
            if not any(word in stop_words for word in words) and len(tg) > 7:  # Ensure trigrams are substantial
                filtered_trigrams.append(tg)
        
        # Initialize result containers
        keywords = set()
        
        # Add specialized terms from our predefined sets that appear in the text
        for term in COMMON_TECH_TERMS:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text):
                keywords.add(term)
        
        for term in EDUCATION_TERMS:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text):
                keywords.add(term)
                
        for term in SOFT_SKILLS:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text):
                keywords.add(term)
        
        # Add domain-specific terms if they appear
        for term in INDUSTRY_DOMAINS:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text):
                keywords.add(term)
        
        # Add job title keywords if they appear
        for term in JOB_TITLES:
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text):
                keywords.add(term)
        
        # Add high-value bigrams and trigrams (using more selective criteria)
        for bg in filtered_bigrams:
            # Only add if it contains a technical term, skill, or domain keyword
            words = bg.split()
            if any(word in COMMON_TECH_TERMS or word in SOFT_SKILLS or word in INDUSTRY_DOMAINS for word in words):
                keywords.add(bg)
        
        for tg in filtered_trigrams:
            # Only add if it contains a technical term, skill, or domain keyword
            words = tg.split()
            if any(word in COMMON_TECH_TERMS or word in SOFT_SKILLS or word in INDUSTRY_DOMAINS for word in words):
                keywords.add(tg)
        
        # Add pattern-based extractions
        keywords.update(experience_patterns)
        keywords.update(certification_patterns)
        keywords.update(degree_patterns)
        
        # Job description specific extractions
        if is_job_description:
            # Extract required skills/qualifications
            required_sections = re.findall(r'(requirements|required skills|qualifications|what you\'ll need)(?::|;)?\s*(.*?)(?:(?:\r?\n){2,}|\Z)', 
                                          text, re.IGNORECASE | re.DOTALL)
            
            for section_name, content in required_sections:
                # Extract key phrases from requirement sections
                section_tokens = word_tokenize(content)
                section_bigrams = [' '.join(bg) for bg in ngrams(section_tokens, 2)]
                section_trigrams = [' '.join(tg) for tg in ngrams(section_tokens, 3)]
                
                # Filter for valuable keywords
                for token in section_tokens:
                    if (token.isalnum() and 
                        len(token) > 2 and 
                        token not in stop_words):
                        keywords.add(lemmatizer.lemmatize(token))
                
                # Add high-value section n-grams
                for bg in section_bigrams:
                    words = bg.split()
                    if not any(word in stop_words for word in words):
                        keywords.add(bg)
                        
                for tg in section_trigrams:
                    words = tg.split()
                    if not any(word in stop_words for word in words):
                        keywords.add(tg)
            
            # Extract required vs preferred patterns more effectively
            required = re.findall(r'(required|must have|essential)[:;]?\s*([\w\s,\-\.]+)', text, re.IGNORECASE)
            for req in required:
                req_text = req[1].lower()
                req_words = word_tokenize(req_text)
                for word in req_words:
                    if word.isalnum() and len(word) > 2 and word not in stop_words:
                        keywords.add(word)
                
                # Also add bigrams from required sections
                req_bigrams = [' '.join(bg) for bg in ngrams(req_words, 2)]
                for bg in req_bigrams:
                    if not any(word in stop_words for word in bg.split()):
                        keywords.add(bg)
        
        # Filter out any remaining noise or irrelevant terms
        filtered_keywords = set()
        for keyword in keywords:
            # Keep only substantial keywords (length > 2 and not just digits)
            if (len(keyword) > 2 and not keyword.isdigit() and 
                not all(c in '+-.,!@#$%^&*();:\'"{}[]|\\/><' for c in keyword)):
                filtered_keywords.add(keyword)
        
        return filtered_keywords
        
    except Exception as e:
        print(f"Error in intelligent keyword extraction: {e}")
        # Fallback to simpler word splitting
        words = text.lower().split()
        return set(word for word in words if word.isalnum() and len(word) > 2 and word not in NOISE_WORDS)

def calculate_match_score(resume_text, job_description):
    """
    Calculate a comprehensive match score between a resume and job description using 
    a hybrid approach combining TF-IDF/BERT embeddings and LLM-based refinement.
    
    This implements a two-step scoring process:
    1. Initial scoring using cosine similarity with BERT embeddings
    2. LLM-based refinement for context-aware evaluation
    """
    # Step 1: TF-IDF/Word Embeddings Initial Scoring
    
    # Use sentence transformer model (BERT-based) to encode resume and job description
    resume_embedding = embedding_model.encode([resume_text])
    job_embedding = embedding_model.encode([job_description])
    
    # Calculate cosine similarity between embeddings
    cosine_similarity = np.dot(resume_embedding, job_embedding.T)[0][0]
    
    # Normalize to [0,1] range
    initial_score = (cosine_similarity + 1) / 2
    
    # Extract keywords for additional context (this helps with the rule-based component)
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description, is_job_description=True)
    
    # Get common keywords for analysis
    common_keywords = resume_keywords.intersection(job_keywords)
    missing_keywords = job_keywords - resume_keywords
    
    # Step 2: LLM-Based Refinement with context-aware evaluation
    prompt = f"""
    You are an expert AI resume evaluator with deep knowledge of ATS systems and hiring processes.
    
    Perform a detailed analysis of how well this resume matches the job description.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Initial Match Score (cosine similarity): {initial_score:.4f}
    
    TASK: Evaluate this resume against the job description, and provide a refined match score
    that takes into account context, meaning, and relevance beyond simple keyword matching.
    
    Consider the following categories and assign numerical scores on a scale of 0.0-1.0:
    1. Skill Match: How well the candidate's technical and soft skills align with requirements
    2. Experience Relevance: Quality and relevance of experience, not just years
    3. Education & Certifications: Alignment with specified requirements
    4. Domain Knowledge: Understanding of the industry and domain-specific concepts
    5. Achievement Relevance: How well accomplishments demonstrate required capabilities
    
    For each category, explain your reasoning in 1-2 sentences, then provide your final refined score.
    Your final score should account for both the initial cosine similarity score AND your categorical assessment.
    
    Response format:
    {{
      "skill_match": {{
        "score": <0.0-1.0>,
        "rationale": "<brief explanation>"
      }},
      "experience_relevance": {{
        "score": <0.0-1.0>,
        "rationale": "<brief explanation>"
      }},
      "education_certifications": {{
        "score": <0.0-1.0>,
        "rationale": "<brief explanation>"
      }},
      "domain_knowledge": {{
        "score": <0.0-1.0>,
        "rationale": "<brief explanation>"
      }},
      "achievement_relevance": {{
        "score": <0.0-1.0>,
        "rationale": "<brief explanation>"
      }},
      "keywords_found": [list of important matched keywords],
      "keywords_missing": [list of important missing keywords],
      "refined_score": <0.0-1.0>,
      "explanation": "<1-2 sentence explanation of final score>"
    }}
    """
    
    try:
        # Call the LLM for the contextual refinement
        response = model.generate_content(prompt)
        analysis_text = response.text
        
        # Extract JSON from response (handle potential text before/after JSON)
        json_start = analysis_text.find('{')
        json_end = analysis_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            analysis_json = analysis_text[json_start:json_end]
            analysis = json.loads(analysis_json)
        else:
            raise ValueError("Could not extract valid JSON from response")
        
        # Get the refined score
        refined_score = float(analysis.get("refined_score", initial_score))
        
        # Ensure score is in valid range
        refined_score = max(0.0, min(1.0, refined_score))
        
        # Extract detailed categorical scores
        enhanced_scores = {
            "Skill Match": analysis["skill_match"]["score"] * 100,
            "Experience Relevance": analysis["experience_relevance"]["score"] * 100,
            "Education & Certifications": analysis["education_certifications"]["score"] * 100,
            "Domain Knowledge": analysis["domain_knowledge"]["score"] * 100,
            "Achievement Relevance": analysis["achievement_relevance"]["score"] * 100
        }
        
        # Extract keywords information
        keywords_found = analysis.get("keywords_found", list(common_keywords))
        keywords_missing = analysis.get("keywords_missing", list(missing_keywords))
        
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        # Fallback to initial scoring if LLM analysis fails
        refined_score = initial_score
        
        # Create basic scores based on initial similarity
    enhanced_scores = {
            "Skill Match": initial_score * 100,
            "Experience Relevance": initial_score * 100,
            "Education & Certifications": initial_score * 100,
            "Domain Knowledge": initial_score * 100,
            "Achievement Relevance": initial_score * 100
        }
        
    keywords_found = list(common_keywords)
    keywords_missing = list(missing_keywords)
    
    # Convert refined score to percentage for display
    percentage_match = refined_score * 100
    
    # Remove duplicates and filter out noise from keywords
    keywords_found = list(set(kw for kw in keywords_found if isinstance(kw, str) and len(kw) > 2))
    keywords_missing = list(set(kw for kw in keywords_missing if isinstance(kw, str) and len(kw) > 2))
    
    return percentage_match, enhanced_scores, keywords_found, keywords_missing

def format_optimized_resume(text):
    """Format the optimized resume with modern styling for dark mode."""
    # Add overall container styling
    html_content = """
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                color: #FFFFFF; 
                background-color: #1E1E1E; 
                padding: 30px; 
                border-radius: 8px;
                max-width: 800px;
                margin: 0 auto;">
    """
    
    # Process and style each line
    lines = text.split('\n')
    in_list = False
    current_section = None
    
    # Define section styles
    section_styles = {
        'summary': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '0'},
        'experience': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'},
        'education': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'},
        'skills': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'},
        'projects': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'},
        'certificates': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'},
        'languages': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'},
        'interests': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'},
        'ui/ux': {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'}
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            # Add spacing between paragraphs
            html_content += "<br>"
            continue
            
        # Skip analysis/suggestion lines
        if any(line.startswith(prefix) for prefix in ['*Option', '**', '>', 'Remember to', 'In summary']):
            continue
            
        # Detect section headers (all caps or ending with a colon)
        if line.isupper() or (len(line) > 2 and line[-1] == ':' and line[0].isupper()):
            # Close list if we were in one
            if in_list:
                html_content += "</ul>"
                in_list = False
                
            # Determine section type
            section_type = line.lower().replace(':', '').strip()
            if section_type in section_styles:
                current_section = section_type
                style = section_styles[section_type]
            else:
                current_section = None
                style = {'color': '#4FC3F7', 'font_size': '16px', 'margin_top': '20px'}
            
            # Format as a heading with section-specific styling
            html_content += f"""
            <h2 style="color: {style['color']}; 
                       font-weight: 700; 
                       font-size: {style['font_size']}; 
                       margin-top: {style['margin_top']}; 
                       margin-bottom: 15px; 
                       border-bottom: 2px solid #555;
                       padding-bottom: 5px;">
                {line}
            </h2>"""
            
        # Detect and format bullet points
        elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
            # Start a list if we're not in one
            if not in_list:
                html_content += '<ul style="margin-top: 10px; margin-bottom: 10px; padding-left: 20px;">'
                in_list = True
                
            # Format as a list item
            content = line[1:].strip() if line.startswith('•') or line.startswith('-') or line.startswith('*') else line
            html_content += f'<li style="color: #FFFFFF; margin-bottom: 8px; line-height: 1.5;">{content}</li>'
            
        else:
            # Close list if we were in one
            if in_list:
                html_content += "</ul>"
                in_list = False
                
            # Detect if this looks like a job title or education entry
            if any(word in line.lower() for word in ['experience', 'developer', 'engineer', 'manager', 'specialist', 'analyst', 'coordinator', 'university', 'college', 'bachelor', 'master', 'phd', 'certificate']):
                # Format job titles and education entries
                html_content += f"""
                <div style="color: #4FC3F7; 
                           font-weight: 600; 
                           margin-top: 15px; 
                           margin-bottom: 5px;
                           font-size: 16px;">
                    {line}
                </div>"""
            else:
                # Format regular text with proper spacing and line height
                html_content += f"""
                <p style="color: #FFFFFF; 
                          margin-bottom: 10px; 
                          line-height: 1.6;
                          font-size: 14px;">
                    {line}
                </p>"""
    
    # Close any open list
    if in_list:
        html_content += "</ul>"
        
    # Close the container
    html_content += "</div>"
    
    return html_content

def optimize_resume(resume_text, job_description):
    """Optimize resume text based on job description"""
    try:
        # Get job requirements
        requirements = extract_requirements(job_description)
        
        # Get resume sections
        sections = extract_sections(resume_text)
        
        # Optimize each section
        optimized_sections = {}
        for section_name, section_text in sections.items():
            if section_name.lower() in ['experience', 'skills', 'education']:
                # Get relevant requirements for this section
                section_requirements = [
                    req for req in requirements 
                    if req['type'] == section_name.lower()
                ]
                
                if section_requirements:
                    # Optimize section content
                    optimized_text = optimize_section(
                        section_text,
                        section_requirements,
                        section_name
                    )
                    optimized_sections[section_name] = optimized_text
                else:
                    optimized_sections[section_name] = section_text
            else:
                optimized_sections[section_name] = section_text
        
        # Reconstruct resume with optimized sections
        optimized_resume = reconstruct_resume(optimized_sections)
            
        return optimized_resume
        
    except Exception as e:
        print(f"Error optimizing resume: {str(e)}")
        return resume_text  # Return original text if optimization fails

def analyze_low_matching(resume_text, job_description):
    """
    Analyze a low-matching resume and provide recommendations.
    
    Args:
        resume_text: The text of the resume to analyze
        job_description: The text of the job description to compare against
        
    Returns:
        Analysis text with recommendations for improvement
    """
    prompt = f"""
    You are an expert resume consultant with deep HR and recruiting expertise.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Provide a comprehensive analysis of how this resume could be improved to better match this specific job description.
    
    Structure your response with these sections:
    
    ## Resume Assessment
    Briefly assess the current resume's strengths and critical gaps compared to the job requirements.
    
    ## Key Missing Elements
    Identify the most important requirements from the job description that are missing or underemphasized in the resume.
    
    ## Actionable Improvements
    Provide 3-5 specific, actionable recommendations to improve the resume's match score.
    
    ## Keyword Recommendations
    List the 5-10 most important keywords from the job description that should be added to the resume.
    
    Focus on being specific, constructive, and practical. Avoid generic advice.
    """
    
    try:
        response = model.generate_content(prompt)
        analysis = response.text
        
        # Clean up the response to remove potential markdown formatting
        if "```" in analysis:
            analysis = analysis.replace("```markdown", "").replace("```", "").strip()
            
        return analysis
    except Exception as e:
        print(f"Error in analyzing resume: {e}")
        return "## Error\nUnable to analyze the resume at this time."

def format_streamlit_markdown(html_content):
    """Convert HTML formatted content to Streamlit-friendly Markdown with consistent styling"""
    # Convert <h3> tags to styled headers
    content = re.sub(r'<h3>(.*?)</h3>', r'<div class="section-header">\1</div>', html_content)
    
    # Convert <p>• content</p> to styled bullet points
    content = re.sub(r'<p>•\s*(.*?)</p>', r'<ul><li style="color:#FFFFFF;">\1</li></ul>', content)
    
    # Convert other <p> tags to uniformly styled text
    content = re.sub(r'<p>(.*?)</p>', r'<p style="color:#FFFFFF; font-size:16px; line-height:1.6;">\1</p>', content)
    
    # Remove <br> tags
    content = content.replace("<br>", "")
    
    return content

def format_analysis(text):
    """Format the analysis text with color-coded scores and better visual structure."""
    html_content = ""
    
    # Add title
    html_content += "<div style='text-align:left; margin-bottom:20px;'>"
    html_content += "<h2 style='color:#FFFFFF; font-weight:700; font-size:24px;'>Match Analysis</h2>"
    html_content += "</div>"
    
    # Extract sections from markdown
    sections = text.split("## ")
    
    for section in sections:
        if not section.strip():
                continue
            
        lines = section.split("\n")
        section_title = lines[0].strip()
        section_content = "\n".join(lines[1:]).strip()
        
        html_content += f"<div style='margin-bottom:15px;'>"
        html_content += f"<h3 style='color:#FFFFFF; font-weight:700; font-size:20px;'>{section_title}</h3>"
        html_content += f"<p style='color:#FFFFFF; font-size:16px;'>{section_content}</p>"
        html_content += "</div>"
    
    return html_content

def display_match_scores(enhanced_scores):
    """Display the match scores in a visually pleasing card layout."""
    st.markdown("<div style='margin-bottom:20px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#FFFFFF; font-weight:700; font-size:24px;'>Detailed Scores</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a layout with 3 columns per row for scores
    cols = st.columns(3)
    
    # Display each score in a card-like format
    for i, (category, score) in enumerate(enhanced_scores.items()):
        col_index = i % 3
        
        # Determine color based on score
        if score >= 80:
            color = "#4CAF50"  # Green for high scores
        elif score >= 60:
            color = "#FFC107"  # Yellow/amber for medium scores
        else:
            color = "#F44336"  # Red for low scores
            
        # Format category name for display (convert from snake_case to Title Case)
        display_category = category.replace('_', ' ').title()
        
        # Create a card-like display for each score
        cols[col_index].markdown(
            f"""
            <div style='background-color:#1E1E1E; padding:10px; border-radius:5px; margin-bottom:10px;'>
                <p style='margin:0; color:#FFFFFF; font-size:14px;'>{display_category}</p>
                <p style='margin:0; color:{color}; font-size:20px; font-weight:bold;'>{score:.1f}%</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)

def batch_processing_ui():
    """UI for batch processing multiple resumes against a job description."""
    # Add dark theme and styling
    st.markdown("""
    <style>
    .analysis-box {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .section-header {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .resume-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        border-left: 4px solid #2196F3;
    }
    .high-match {
        border-left: 4px solid #4CAF50 !important;
    }
    .medium-match {
        border-left: 4px solid #FFC107 !important;
    }
    .low-match {
        border-left: 4px solid #F44336 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("## Batch Resume Processing")
    
    # Job description input
    job_description = st.text_area("Paste the job description here", height=200)
    
    # Upload multiple resumes
    uploaded_files = st.file_uploader("Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and job_description:
        if st.button("Analyze All Resumes"):
            st.write("### Results")
            
            # Process each resume
            resume_data = []
            
            with st.spinner("Processing resumes..."):
                for file in uploaded_files:
                    # Parse the resume
                    resume_text = upload_and_parse_resume(file)
                    if resume_text:
                        # Calculate match score
                        match_score, enhanced_scores, keywords_found, keywords_missing = calculate_match_score(resume_text, job_description)
                        
                        # Determine classification
                        if match_score > 75:
                            classification = "High Match"
                            badge_class = "high-match"
                        elif match_score > 60:
                            classification = "Medium Match" 
                            badge_class = "medium-match"
                        else:
                            classification = "Low Match"
                            badge_class = "low-match"
                
                        # Add to results
                        resume_data.append({
                            'file': file,
                            'text': resume_text,
                            'match_score': match_score,
                            'classification': classification,
                            'badge_class': badge_class,
                            'enhanced_scores': enhanced_scores,
                            'keywords_found': keywords_found,
                            'keywords_missing': keywords_missing
                        })
            
            # Sort by match score (highest first)
            resume_data.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Display summary table
            if resume_data:
                # Create dataframe for table
                table_data = []
                for idx, resume in enumerate(resume_data):
                    table_data.append({
                        "Rank": idx + 1,
                        "Filename": resume['file'].name,
                        "Match Score": f"{resume['match_score']:.1f}%",
                        "Classification": resume['classification']
                    })
                
                df = pd.DataFrame(table_data)
                st.table(df)
                
                # Display detailed cards for each resume
                for resume in resume_data:
                    # Create card with appropriate color based on classification
                        st.markdown(f"""
                    <div class="resume-card {resume['badge_class']}">
                        <h3 style="color:#FFFFFF; margin-bottom:10px;">{resume['file'].name}</h3>
                        <p style="color:#FFFFFF; margin-bottom:5px;">Match Score: <span style="font-weight:bold;">{resume['match_score']:.1f}%</span></p>
                        <p style="color:#FFFFFF; margin-bottom:10px;">Classification: <span style="font-weight:bold;">{resume['classification']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                    # Expandable section for details
                with st.expander(f"View Details for {resume['file'].name}"):
                    # Generate optimized resume and display it first
                    st.markdown('<div class="section-header">Optimized Resume</div>', unsafe_allow_html=True)
                    optimized_resume = optimize_resume(resume['text'], job_description)
                    formatted_resume = format_optimized_resume(optimized_resume)
                    st.markdown(formatted_resume, unsafe_allow_html=True)
                    
                    # Add download buttons
                    # Strip HTML tags for plain text version
                    plain_optimized_resume = strip_html_tags(formatted_resume)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            f"Download {resume['file'].name} (HTML)",
                            formatted_resume,
                            file_name=f"optimized_{resume['file'].name}.html",
                            mime="text/html"
                        )
                    with col2:
                        st.download_button(
                            f"Download {resume['file'].name} (Text)",
                            plain_optimized_resume,
                            file_name=f"optimized_{resume['file'].name}.txt",
                            mime="text/plain"
                        )
                    
                    # Display match scores in card format
                    display_match_scores(resume['enhanced_scores'])
                    
                    # Display Keywords found
                    st.markdown('<div class="section-header">Keywords Found</div>', unsafe_allow_html=True)
                    if resume['keywords_found']:
                        keyword_html = "<div class='analysis-box'>"
                        for kw in resume['keywords_found']:
                            keyword_html += f"<span style='display:inline-block; background-color:#1e8449; color:white; margin:3px; padding:5px 10px; border-radius:15px;'>{kw}</span>"
                        keyword_html += "</div>"
                        st.markdown(keyword_html, unsafe_allow_html=True)
                    else:
                        st.warning("No matching keywords found.")
                    
                    # Display Keywords missing
                    st.markdown('<div class="section-header">Keywords Missing</div>', unsafe_allow_html=True)
                    if resume['keywords_missing']:
                        keyword_html = "<div class='analysis-box'>"
                        for kw in resume['keywords_missing']:
                            keyword_html += f"<span style='display:inline-block; background-color:#c0392b; color:white; margin:3px; padding:5px 10px; border-radius:15px;'>{kw}</span>"
                        keyword_html += "</div>"
                        st.markdown(keyword_html, unsafe_allow_html=True)
                    else:
                        st.success("No missing keywords!")
                    
                    # Generate and display analysis for low-matching resumes
                    if resume['match_score'] < 75:
                        st.markdown('<div class="section-header">Improvement Analysis</div>', unsafe_allow_html=True)
                        analysis = analyze_low_matching(resume['text'], job_description)
                        st.markdown(format_analysis(analysis), unsafe_allow_html=True)
            else:
                st.error("No valid resumes were processed.")
            
def single_resume_optimization_ui():
    """UI for optimizing a single resume against a job description."""
    # Add dark theme and styling
    st.markdown("""
    <style>
    .analysis-box {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .section-header {
        color: #FFFFFF;
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
            """, unsafe_allow_html=True)
            
    st.write("## Optimize Your Resume")
    
    # File uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        if resume_file:
            st.success("Resume uploaded successfully!")
            
    with col2:
        job_desc_file = st.file_uploader("Upload job description (PDF)", type=["pdf"])
        if job_desc_file:
            st.success("Job description uploaded successfully!")
            
    # Text areas for manual input
    st.write("### Or paste your content below:")
    col1, col2 = st.columns(2)
    
    with col1:
        resume_text = st.text_area("Resume Text", height=200, placeholder="Paste your resume text here...")
        
    with col2:
        job_desc_text = st.text_area("Job Description", height=200, placeholder="Paste the job description here...")
        
    # Process inputs
    if st.button("Analyze and Optimize"):
        # First validate that we have either files or text
        if not (resume_file or resume_text):
            st.error("Please provide your resume either as a file or as text.")
            st.stop()
            
        if not (job_desc_file or job_desc_text):
            st.error("Please provide the job description either as a file or as text.")
            st.stop()
            
        # Show loading state
        with st.spinner("Processing your resume..."):
            # Parse uploaded files if available
            if resume_file:
                resume_text = upload_and_parse_resume(resume_file)
                
            if job_desc_file:
                job_desc_text = upload_and_parse_resume(job_desc_file)
            
            # Calculate match score
            match_score, enhanced_scores, keywords_found, keywords_missing = calculate_match_score(resume_text, job_desc_text)
            
            # Generate optimized resume first
            with st.spinner("Generating optimized resume..."):
                optimized_resume = optimize_resume(resume_text, job_desc_text)
                formatted_resume = format_optimized_resume(optimized_resume)
            
            # Display results
            st.markdown("""---""")
            st.markdown("<h2 style='color:#FFFFFF; text-align:center;'>Resume Analysis Results</h2>", unsafe_allow_html=True)
            
            # Create tabs for different sections - reordered to show optimized resume first
            tabs = st.tabs(["Optimized Resume", "Match Score", "Detailed Analysis", "Keywords"])
            
            with tabs[0]:
                # Show optimized resume first
                st.markdown('<div class="section-header">Optimized Resume</div>', unsafe_allow_html=True)
                st.markdown(formatted_resume, unsafe_allow_html=True)
                
                # Add download button that only contains the optimized resume
                # Strip HTML tags for plain text version
                plain_optimized_resume = strip_html_tags(formatted_resume)
                
                # Provide both HTML and TXT download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download Optimized Resume (HTML)",
                        formatted_resume,
                        file_name="optimized_resume.html",
                        mime="text/html"
                    )
                with col2:
                    st.download_button(
                        "Download Optimized Resume (Text)",
                        plain_optimized_resume,
                        file_name="optimized_resume.txt",
                        mime="text/plain"
                    )
            
            with tabs[1]:
                # Create score visualization
                score_color = "#4CAF50" if match_score >= 75 else "#FFC107" if match_score >= 60 else "#F44336"
                st.markdown(
                    f"""
                    <div style='text-align:center; margin:20px 0;'>
                        <p style='font-size:18px; color:#FFFFFF;'>Your Resume Match Score</p>
                        <div style='font-size:48px; font-weight:bold; color:{score_color};'>{match_score:.1f}%</div>
                </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            with tabs[2]:
                # Display match scores in card format
                display_match_scores(enhanced_scores)
                
                # Generate and display analysis
                if match_score < 75:
                    analysis = analyze_low_matching(resume_text, job_desc_text)
                    st.markdown(format_analysis(analysis), unsafe_allow_html=True)
            
            with tabs[3]:
                # Display keywords found with positive highlighting
                st.markdown('<div class="section-header">Keywords Found</div>', unsafe_allow_html=True)
                if keywords_found:
                    keyword_html = "<div class='analysis-box'>"
                    for kw in keywords_found:
                        keyword_html += f"<span style='display:inline-block; background-color:#1e8449; color:white; margin:3px; padding:5px 10px; border-radius:15px;'>{kw}</span>"
                    keyword_html += "</div>"
                    st.markdown(keyword_html, unsafe_allow_html=True)
                else:
                    st.warning("No matching keywords found! Your resume may need significant improvement.")
                
                # Display keywords missing with negative highlighting
                st.markdown('<div class="section-header">Keywords Missing</div>', unsafe_allow_html=True)
                if keywords_missing:
                    keyword_html = "<div class='analysis-box'>"
                    for kw in keywords_missing:
                        keyword_html += f"<span style='display:inline-block; background-color:#c0392b; color:white; margin:3px; padding:5px 10px; border-radius:15px;'>{kw}</span>"
                    keyword_html += "</div>"
                    st.markdown(keyword_html, unsafe_allow_html=True)
                else:
                    st.success("Impressive! Your resume contains all the important keywords.")
            
def classify_resume(resume_text, job_description):
    """Classify resume match level based on job description"""
    try:
        # Get job requirements
        requirements = extract_requirements(job_description)
        
        # Get resume sections
        sections = extract_sections(resume_text)
        
        # Calculate match scores for each requirement type
        match_scores = {}
        for req_type in ['experience', 'skills', 'education']:
            if req_type in sections:
                section_text = sections[req_type]
                type_requirements = [req for req in requirements if req['type'] == req_type]
                
                if type_requirements:
                    match_scores[req_type] = calculate_match_score(
                        section_text,
                        type_requirements
                    )
                else:
                    match_scores[req_type] = 0.0
            else:
                match_scores[req_type] = 0.0
        
        # Calculate overall match score
        total_score = sum(match_scores.values()) / len(match_scores)
        
        # Classify based on match score
        if total_score >= 0.8:
            return "Strong Match"
        elif total_score >= 0.6:
            return "Good Match"
        elif total_score >= 0.4:
            return "Moderate Match"
        else:
            return "Low Match"
            
    except Exception as e:
        print(f"Error classifying resume: {str(e)}")
        return "Error: Unable to classify resume at this time."

def extract_requirements(job_description):
    """Extract requirements from job description"""
    try:
        # Initialize requirements list
        requirements = []
        
        # Extract experience requirements
        experience_patterns = [
            r'(\d+)\+?\s*years?\s+of\s+experience',
            r'(\d+)\s*-\s*(\d+)\s*years?\s+of\s+experience',
            r'experience\s+with\s+([^.,]+)',
            r'(\d+)\+?\s*years?\s+in\s+([^.,]+)'
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, job_description, re.IGNORECASE)
            for match in matches:
                requirements.append({
                    'type': 'experience',
                    'text': match.group(0),
                    'value': match.group(1) if len(match.groups()) == 1 else match.group(2)
                })
        
        # Extract skill requirements
        skill_patterns = [
            r'proficiency\s+in\s+([^.,]+)',
            r'knowledge\s+of\s+([^.,]+)',
            r'experience\s+with\s+([^.,]+)',
            r'familiarity\s+with\s+([^.,]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, job_description, re.IGNORECASE)
            for match in matches:
                requirements.append({
                    'type': 'skills',
                    'text': match.group(0),
                    'value': match.group(1)
                })
        
        # Extract education requirements
        education_patterns = [
            r'(bachelor|master|phd)\'?s?\s+degree\s+in\s+([^.,]+)',
            r'(bachelor|master|phd)\'?s?\s+of\s+([^.,]+)',
            r'(bachelor|master|phd)\'?s?\s+in\s+([^.,]+)'
        ]
        
        for pattern in education_patterns:
            matches = re.finditer(pattern, job_description, re.IGNORECASE)
            for match in matches:
                requirements.append({
                    'type': 'education',
                    'text': match.group(0),
                    'value': match.group(2)
                })
        
        return requirements
        
    except Exception as e:
        print(f"Error extracting requirements: {str(e)}")
        return []

def extract_sections(resume_text):
    """Extract sections from resume text"""
    try:
        sections = {}
        
        # Common section headers
        section_headers = {
            'experience': r'(?i)(work\s+experience|professional\s+experience|employment\s+history|experience)',
            'education': r'(?i)(education|academic\s+background|qualifications)',
            'skills': r'(?i)(skills|technical\s+skills|core\s+competencies|expertise)',
            'summary': r'(?i)(summary|profile|objective|career\s+objective)',
            'projects': r'(?i)(projects|portfolio|personal\s+projects)'
        }
        
        # Split text into lines
        lines = resume_text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            is_header = False
            for section_name, pattern in section_headers.items():
                if re.match(pattern, line):
                    # Save previous section if exists
                    if current_section and section_content:
                        sections[current_section] = '\n'.join(section_content)
                    
                    # Start new section
                    current_section = section_name
                    section_content = []
                    is_header = True
                    break
            
            if not is_header and current_section:
                section_content.append(line)
        
        # Save last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
        
    except Exception as e:
        print(f"Error extracting sections: {str(e)}")
        return {}

def optimize_section(section_text, requirements, section_name):
    """Optimize a specific section of the resume based on requirements"""
    try:
        # Get relevant keywords from requirements
        keywords = []
        for req in requirements:
            if req['type'] == section_name.lower():
                # Extract keywords from requirement text
                words = req['text'].lower().split()
                keywords.extend([w for w in words if len(w) > 2 and w not in NOISE_WORDS])
        
        # If no keywords found, return original text
        if not keywords:
            return section_text
        
        # Use LLM to optimize the section
        prompt = f"""
        Optimize this {section_name} section of a resume to better match the job requirements.
        
        Original text:
        {section_text}
        
        Important keywords to incorporate:
        {', '.join(keywords)}
        
        Requirements to address:
        {json.dumps(requirements, indent=2)}
        
        Please optimize the text while maintaining its factual accuracy and professional tone.
        Focus on incorporating relevant keywords naturally and highlighting relevant experience.
        """
        
        response = model.generate_content(prompt)
        optimized_text = response.text
        
        return optimized_text
        
    except Exception as e:
        print(f"Error optimizing section: {str(e)}")
        return section_text

def reconstruct_resume(sections):
    """Reconstruct resume from optimized sections"""
    try:
        # Define section order
        section_order = ['summary', 'experience', 'education', 'skills', 'projects']
        
        # Build resume text
        resume_text = []
        for section in section_order:
            if section in sections:
                # Add section header
                resume_text.append(section.upper())
                # Add section content
                resume_text.append(sections[section])
                # Add spacing between sections
                resume_text.append('')
        
        return '\n'.join(resume_text)
        
    except Exception as e:
        print(f"Error reconstructing resume: {str(e)}")
        return '\n'.join(sections.values())

def strip_html_tags(html_content):
    """Convert HTML content to plain text by removing HTML tags."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', html_content)
    
    # Replace HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Handle line breaks
    text = text.replace('<br>', '\n')
    text = text.replace('<br/>', '\n')
    text = text.replace('<br />', '\n')
    text = text.replace('</p>', '\n')
    text = text.replace('</h2>', '\n')
    text = text.replace('</h3>', '\n')
    text = text.replace('</div>', '\n')
    text = text.replace('</li>', '\n')
    
    # Remove any remaining tags
    text = re.sub(r'<.*?>', '', text)
    
    # Fix multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def main_ui():
    # Add custom styling directly here instead of loading external CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    .stMarkdown, .stText {
        color: #FFFFFF;
    }
    .css-10trblm {
        color: #FFFFFF;
    }
    .css-1g4k55y {
        padding: 2rem 1rem;
    }
    .stButton > button {
        background-color: #2196F3;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #0D47A1;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E1E1E;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header and title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #FFFFFF; font-size: 2.5rem;">Resume Ranker</h1>
        <p style="color: #AAAAAA; font-size: 1.2rem;">AI-Powered Resume Analysis & Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main navigation
    tab1, tab2 = st.tabs(["Single Resume", "Batch Processing"])
    
    with tab1:
        single_resume_optimization_ui()
    
    with tab2:
        batch_processing_ui()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #444444;">
        <p style="color: #AAAAAA; font-size: 0.8rem;">
            Resume Ranker | Powered by AI | © 2023
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main_ui()
