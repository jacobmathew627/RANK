import streamlit as st
import re
from app import upload_and_parse_resume, calculate_match_score, optimize_resume, analyze_low_matching, format_analysis

def apply_uniform_styling():
    """Apply uniform styling for better consistency in the UI"""
    st.markdown("""
    <style>
    /* Uniform font sizes */
    .stMarkdown h1 {font-size: 28px !important;}
    .stMarkdown h2 {font-size: 24px !important;}
    .stMarkdown h3 {font-size: 20px !important;}
    .stMarkdown p, .stMarkdown li, .stMarkdown ul {font-size: 16px !important; line-height: 1.6 !important;}
    
    /* Analysis box styling */
    .analysis-box {
        background-color: #f8f8f8;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        font-size: 16px !important;
    }
    
    /* Consistent bullet points */
    .analysis-box ul li {
        margin-bottom: 8px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 20px !important;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

def format_streamlit_markdown(html_content):
    """Convert HTML formatted content to Streamlit-friendly Markdown with consistent styling"""
    # Convert <h3> tags to styled headers
    content = re.sub(r'<h3>(.*?)</h3>', r'<div class="section-header">\1</div>', html_content)
    
    # Convert <p>• content</p> to styled bullet points
    content = re.sub(r'<p>•\s*(.*?)</p>', r'<ul><li>\1</li></ul>', content)
    
    # Convert other <p> tags to uniformly styled text
    content = re.sub(r'<p>(.*?)</p>', r'<p style="font-size: 16px; line-height: 1.6;">\1</p>', content)
    
    # Remove <br> tags
    content = content.replace("<br>", "")
    
    return content

def batch_processing_ui():
    st.header("Batch Processing")
    uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx"], accept_multiple_files=True)
    job_description = st.text_area("Paste the job description here")
    
    if uploaded_files and job_description:
        all_resumes = []
        for uploaded_file in uploaded_files:
            resume_text = upload_and_parse_resume(uploaded_file)
            if resume_text:
                all_resumes.append({
                    "name": uploaded_file.name,
                    "text": resume_text
                })
        
        if all_resumes:
            # Calculate scores and sort resumes
            for resume in all_resumes:
                resume['score'], resume['enhanced_scores'], resume['keywords_found'], resume['keywords_missing'] = calculate_match_score(resume['text'], job_description)
            
            # Sort resumes by score
            all_resumes.sort(key=lambda x: x['score'], reverse=True)
            
            st.write("## Resume Analysis")
            for idx, resume in enumerate(all_resumes, start=1):
                if resume['score'] > 75:
                    classification = "Highly Matching"
                elif resume['score'] > 50:
                    classification = "Medium Matching"
                else:
                    classification = "Low Matching"
                
                with st.expander(f"{idx}. {resume['name']} - {resume['score']:.0f}% Match ({classification})"):
                    # Display enhanced scores in a consistent format
                    st.markdown('<div class="section-header">Detailed Scores</div>', unsafe_allow_html=True)
                    score_html = "<div class='analysis-box'>"
                    for category, score in resume['enhanced_scores'].items():
                        # Convert category from snake_case to Title Case
                        category_name = ' '.join(word.capitalize() for word in category.split('_'))
                        score_html += f"<p><strong>{category_name}:</strong> {score:.0f}%</p>"
                    score_html += "</div>"
                    st.markdown(score_html, unsafe_allow_html=True)
                    
                    # Keywords found and missing with consistent styling
                    col1, col2 = st.columns(2)
                    with col1:
                        if resume['keywords_found']:
                            st.markdown('<div class="section-header">Keywords Found</div>', unsafe_allow_html=True)
                            st.markdown(f"<div class='analysis-box'>{', '.join(resume['keywords_found'])}</div>", 
                                      unsafe_allow_html=True)
                    
                    with col2:
                        if resume['keywords_missing']:
                            st.markdown('<div class="section-header">Keywords Missing</div>', unsafe_allow_html=True)
                            st.markdown(f"<div class='analysis-box'>{', '.join(resume['keywords_missing'])}</div>", 
                                      unsafe_allow_html=True)
                    
                    # Get full analysis
                    reasons, suggestions, full_analysis = analyze_low_matching(resume['text'], job_description)
                    formatted_html = format_analysis(full_analysis)
                    
                    # Apply consistent styling to the formatted analysis
                    formatted_html = format_streamlit_markdown(formatted_html)
                    
                    # Display the formatted analysis with consistent styling
                    st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='analysis-box'>{formatted_html}</div>", unsafe_allow_html=True)
                    
                    # Show the optimized resume option
                    if st.button(f"Generate Optimized Resume for {resume['name']}", key=f"optimize_{idx}"):
                        optimized_resume = optimize_resume(resume['text'], job_description)
                        st.markdown('<div class="section-header">Optimized Resume</div>', unsafe_allow_html=True)
                        st.markdown(f"<div class='analysis-box'>{optimized_resume}</div>", unsafe_allow_html=True)

def single_resume_optimization_ui():
    st.header("Single Resume Optimization")
    resume_file = st.file_uploader("Upload a single resume", type=["pdf", "docx"], key="single_resume")
    single_job_description = st.text_area("Paste the job description here", key="single_job")
    
    if resume_file and single_job_description:
        resume_text = upload_and_parse_resume(resume_file)
        if resume_text:
            match_score, enhanced_scores, keywords_found, keywords_missing = calculate_match_score(resume_text, single_job_description)
            
            # Display match score with consistent styling
            st.markdown(f'<div class="section-header">Overall Match Score: {match_score:.0f}%</div>', unsafe_allow_html=True)
            
            # Display enhanced scores in a consistent format
            st.markdown('<div class="section-header">Detailed Scores</div>', unsafe_allow_html=True)
            score_html = "<div class='analysis-box'>"
            for category, score in enhanced_scores.items():
                # Convert category from snake_case to Title Case
                category_name = ' '.join(word.capitalize() for word in category.split('_'))
                score_html += f"<p><strong>{category_name}:</strong> {score:.0f}%</p>"
            score_html += "</div>"
            st.markdown(score_html, unsafe_allow_html=True)
            
            # Keywords found and missing with consistent styling
            col1, col2 = st.columns(2)
            with col1:
                if keywords_found:
                    st.markdown('<div class="section-header">Keywords Found</div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='analysis-box'>{', '.join(keywords_found)}</div>", 
                              unsafe_allow_html=True)
            
            with col2:
                if keywords_missing:
                    st.markdown('<div class="section-header">Keywords Missing</div>', unsafe_allow_html=True)
                    st.markdown(f"<div class='analysis-box'>{', '.join(keywords_missing)}</div>", 
                              unsafe_allow_html=True)
            
            # Get full analysis
            reasons, suggestions, full_analysis = analyze_low_matching(resume_text, single_job_description)
            formatted_html = format_analysis(full_analysis)
            
            # Apply consistent styling to the formatted analysis
            formatted_html = format_streamlit_markdown(formatted_html)
            
            # Display the formatted analysis with consistent styling
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='analysis-box'>{formatted_html}</div>", unsafe_allow_html=True)
            
            # Show the optimized resume option
            if st.button("Generate Optimized Resume"):
                optimized_resume = optimize_resume(resume_text, single_job_description)
                st.markdown('<div class="section-header">Optimized Resume</div>', unsafe_allow_html=True)
                st.markdown(f"<div class='analysis-box'>{optimized_resume}</div>", unsafe_allow_html=True)

def main_ui():
    st.title("Resume Analyzer & Optimizer")
    # Apply custom styling for uniformity
    apply_uniform_styling()
    
    st.markdown("""
    Upload your resume and job description to get:
    - Match percentage analysis
    - Detailed feedback on alignment
    - Tailored improvement recommendations
    - Resume optimization suggestions
    """)
    
    tab1, tab2 = st.tabs(["Single Resume Analysis", "Batch Processing"])
    
    with tab1:
        single_resume_optimization_ui()
    
    with tab2:
        batch_processing_ui()

if __name__ == "__main__":
    main_ui() 