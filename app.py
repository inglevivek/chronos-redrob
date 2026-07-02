import streamlit as st
import subprocess
import pandas as pd
from pathlib import Path
import sys
import os
import json

st.set_page_config(page_title="Candidate Ranking Pipeline", page_icon="🏆", layout="wide")

@st.cache_data
def load_candidates(candidates_file):
    candidates = []
    if not candidates_file.exists():
        return []
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                candidates.append(c)
            except json.JSONDecodeError:
                pass
    return candidates

@st.cache_data
def load_eligible_ids(ids_file):
    if not ids_file.exists():
        return set()
    with open(ids_file, "r", encoding="utf-8") as f:
        return set(json.load(f))

def main():
    st.title("🏆 Intelligent Candidate Discovery & Ranking")
    
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / 'data'
    candidates_file = data_dir / "candidates_full.jsonl"
    eligible_ids_file = data_dir / "candidate_ids.json"
    
    # Load data
    try:
        candidates = load_candidates(candidates_file)
        eligible_ids = load_eligible_ids(eligible_ids_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
        
    if not candidates:
        st.warning(f"Could not find candidates at {candidates_file}")
        return
        
    def format_candidate(c):
        cid = c.get("candidate_id")
        name = c.get("profile", {}).get("anonymized_name", "Unknown")
        title = c.get("profile", {}).get("current_title", "")
        if cid in eligible_ids:
            return f"{cid} | {name} - {title}"
        else:
            return f"{cid} | {name} - {title} (Ineligible - No Artifacts)"
            
    candidate_options = {format_candidate(c): c for c in candidates}
    
    tab1, tab2, tab3, tab4 = st.tabs(["Main Inference", "Custom Ranking (Test Cases)", "Overall Leaderboard", "Candidate Explorer"])
    
    with tab1:
        st.header("Main Inference Pipeline")
        st.markdown("Run the complete retrieval and ranking pipeline on the full dataset.")
        if st.button("🚀 Run Complete Pipeline", type="primary"):
            st.subheader("Execution Logs")
            log_placeholder = st.empty()
            output_log = "Pipeline started...\n"
            log_placeholder.text_area("Logs", value=output_log, height=300, label_visibility="collapsed")
            
            script_path = repo_root / "pipeline.py"
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            if "TEST_CANDIDATES" in env:
                del env["TEST_CANDIDATES"]
                
            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script_path)],
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    bufsize=1
                )
                
                output_log = ""
                for line in iter(process.stdout.readline, ''):
                    # Clean carriage returns from tqdm to avoid messy UI
                    if '\r' in line:
                        clean_line = line.split('\r')[-1]
                        if clean_line.strip():
                            output_log += clean_line
                    else:
                        output_log += line
                    # Update text area in real-time
                    log_placeholder.text_area("Logs", value=output_log, height=400, label_visibility="collapsed")
                    
                process.wait()
                if process.returncode == 0:
                    output_log += f"\n✅ pipeline.py completed successfully.\n"
                    log_placeholder.text_area("Logs", value=output_log, height=400, label_visibility="collapsed")
                
                submission_file = repo_root / "submission.csv"
                if submission_file.exists():
                    df = pd.read_csv(submission_file)
                    st.success("Pipeline executed successfully! Below is the final Leaderboard.")
                    st.dataframe(df, width='stretch')
            except Exception as e:
                output_log += f"\n❌ pipeline.py failed: {str(e)}\n"
                if 'process' in locals() and process.returncode != 0:
                    output_log += f"Exit code: {process.returncode}\n"
                log_placeholder.text_area("Logs", value=output_log, height=400, label_visibility="collapsed")

    with tab2:
        st.header("Test Pipeline with Custom Candidates")
        st.markdown("Search and select multiple candidates to evaluate them directly using the Cross-Encoder pipeline. **Only eligible candidates can be evaluated.**")
        
        selected_labels = st.multiselect("Search and select candidates", list(candidate_options.keys()))
        
        selected_cids = []
        has_ineligible = False
        for label in selected_labels:
            c = candidate_options[label]
            cid = c.get("candidate_id")
            if cid not in eligible_ids:
                has_ineligible = True
            else:
                selected_cids.append(cid)
                
        if has_ineligible:
            st.warning("⚠️ You have selected one or more ineligible candidates. They will be ignored by the pipeline.")
            
        if st.button("🚀 Run Pipeline on Selected Candidates", type="primary", disabled=len(selected_cids)==0):
            st.subheader("Execution Logs")
            log_placeholder = st.empty()
            output_log = f"Pipeline started for {len(selected_cids)} candidates...\n"
            log_placeholder.text_area("Logs", value=output_log, height=300, label_visibility="collapsed")
            
            script_path = repo_root / "pipeline.py"
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["TEST_CANDIDATES"] = ",".join(selected_cids)
            
            try:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script_path)],
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    env=env,
                    bufsize=1
                )
                
                output_log = ""
                for line in iter(process.stdout.readline, ''):
                    if '\r' in line:
                        clean_line = line.split('\r')[-1]
                        if clean_line.strip():
                            output_log += clean_line
                    else:
                        output_log += line
                    log_placeholder.text_area("Logs", value=output_log, height=400, label_visibility="collapsed")
                    
                process.wait()
                if process.returncode == 0:
                    output_log += f"\n✅ pipeline.py completed successfully.\n"
                    log_placeholder.text_area("Logs", value=output_log, height=400, label_visibility="collapsed")
                
                # Show custom results
                custom_submission = repo_root / "custom_submission.csv"
                if custom_submission.exists():
                    df = pd.read_csv(custom_submission)
                    st.success("Pipeline executed successfully! Below are the ranked results for your selected candidates.")
                    st.dataframe(df, width='stretch')
            except Exception as e:
                output_log += f"\n❌ pipeline.py failed: {str(e)}\n"
                if 'process' in locals() and process.returncode != 0:
                    output_log += f"Exit code: {process.returncode}\n"
                log_placeholder.text_area("Logs", value=output_log, height=400, label_visibility="collapsed")

    with tab3:
        st.header("Overall Leaderboard")
        submission_file = repo_root / "submission.csv"
        if submission_file.exists():
            df = pd.read_csv(submission_file)
            st.dataframe(df, width='stretch')
        else:
            st.info("No submission.csv found. The full pipeline has not been run yet.")
            
    with tab4:
        st.header("Candidate Explorer")
        st.markdown("Browse the full dataset of 100k candidates.")
        explorer_label = st.selectbox("Select a candidate to view their profile", list(candidate_options.keys()))
        if explorer_label:
            c = candidate_options[explorer_label]
            cid = c.get("candidate_id")
            prof = c.get("profile", {})
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{prof.get('anonymized_name', 'Unknown')} ({cid})")
                st.markdown(f"**{prof.get('current_title', 'No Title')}** at **{prof.get('current_company', 'No Company')}**")
                st.markdown(f"*{prof.get('location', 'Unknown Location')}* • {prof.get('years_of_experience', 0)} years experience")
            with col2:
                if st.button("🚀 Calculate Score", key=f"score_btn_{cid}", type="primary"):
                    st.session_state["score_target"] = cid

            st.divider()
            
            if st.session_state.get("score_target") == cid:
                st.markdown("### Evaluation Results")
                with st.spinner("Running pipeline for this candidate..."):
                    script_path = repo_root / "pipeline.py"
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"
                    env["TEST_CANDIDATES"] = cid
                    try:
                        subprocess.run(
                            [sys.executable, str(script_path)],
                            cwd=str(repo_root),
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            env=env,
                            check=True
                        )
                        custom_submission = repo_root / "custom_submission.csv"
                        if custom_submission.exists():
                            df = pd.read_csv(custom_submission)
                            if not df.empty:
                                row = df.iloc[0]
                                col_a, col_b, col_c = st.columns(3)
                                col_a.metric("Cross-Encoder Match", f"{row.get('ce_score', 0):.2%}")
                                col_b.metric("Rubric Composite", f"{row.get('rubric_composite', 0):.4f}")
                                col_c.metric("Final Absolute Score", f"{row.get('score', 0):.2%}")
                                st.info(f"**Reasoning:** {row.get('reasoning', '')}")
                            else:
                                st.error("No score returned.")
                    except subprocess.CalledProcessError as e:
                        st.error(f"Pipeline failed to score candidate.\nError: {e.stderr}")
                st.divider()

            st.markdown("### Summary")
            st.write(prof.get("summary", "No summary provided."))
            
            st.markdown("### Experience")
            for job in c.get("career_history", []):
                with st.expander(f"{job.get('title')} at {job.get('company')} ({job.get('duration_months', 0)} months)"):
                    st.write(job.get("description", ""))
                    
            st.markdown("### Skills")
            skills = [s.get("name") for s in c.get("skills", [])]
            if skills:
                st.markdown(", ".join([f"`{s}`" for s in skills]))
            else:
                st.write("No skills listed.")

if __name__ == "__main__":
    main()
