import streamlit as st
from app import upload_and_parse_resume, calculate_match_score, optimize_resume, analyze_low_matching

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
                resume['score'] = calculate_match_score(resume['text'], job_description)
            
            # Sort resumes by score
            all_resumes.sort(key=lambda x: x['score'], reverse=True)
            
            st.write("## Resume Analysis")
            for idx, resume in enumerate(all_resumes, start=1):
                if resume['score'] > 8:
                    classification = "Highly Matching"
                elif resume['score'] > 5:
                    classification = "Medium Matching"
                else:
                    classification = "Low Matching"
                
                with st.expander(f"{idx}. {resume['name']}"):
                    st.markdown(f"**Match Score:** {resume['score']:.2f} - **{classification}**")
                    
                    # Show analysis when name is clicked
                    reasons, suggestions = analyze_low_matching(resume['text'], job_description)
                    st.markdown("**Reasons for Match Score:**")
                    for reason in reasons:
                        st.markdown(f"- {reason}")
                    st.markdown("**Suggestions for Improvement:**")
                    for suggestion in suggestions:
                        st.markdown(f"- {suggestion}")
                    
                    optimized_resume = optimize_resume(resume['text'], job_description)
                    st.markdown("**Optimized Resume:**")
                    st.markdown(optimized_resume)

def single_resume_optimization_ui():
    st.header("Single Resume Optimization")
    resume_file = st.file_uploader("Upload a single resume", type=["pdf", "docx"], key="single_resume")
    single_job_description = st.text_area("Paste the job description here", key="single_job")
    
    if resume_file and single_job_description:
        resume_text = upload_and_parse_resume(resume_file)
        if resume_text:
            match_score = calculate_match_score(resume_text, single_job_description)
            optimized_resume = optimize_resume(resume_text, single_job_description)
            
            st.markdown(f"**Match Score:** {match_score:.2f}")
            st.markdown("## Optimized Resume")
            st.markdown(optimized_resume)

def main_ui():
    st.title("Resume Optimization and Classification with RAG using Gemini API")
    
    tab1, tab2 = st.tabs(["Batch Processing", "Single Resume Optimization"])
    
    with tab1:
        batch_processing_ui()
    
    with tab2:
        single_resume_optimization_ui()

if __name__ == "__main__":
    main_ui() 