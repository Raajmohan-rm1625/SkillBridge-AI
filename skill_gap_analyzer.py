"""
SkillBridge AI — Skill Gap Analyzer Service
Runs on AMD Instinct MI300X via ROCm / PyTorch
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import json
from typing import List, Dict

# AMD ROCm device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SkillBridge] Running on: {device}")

# --- Industry Skill Taxonomy (simplified) ---
ROLE_SKILL_MAP = {
    "Software Engineer": [
        "Python", "Data Structures", "Algorithms", "System Design",
        "Git", "SQL", "REST APIs", "Docker", "Cloud (AWS/GCP/Azure)",
        "Unit Testing", "CI/CD"
    ],
    "Data Scientist": [
        "Python", "Machine Learning", "Statistics", "Pandas", "NumPy",
        "Scikit-learn", "Deep Learning", "SQL", "Data Visualization",
        "Feature Engineering", "Model Deployment"
    ],
    "Product Manager": [
        "Product Roadmapping", "User Research", "Agile/Scrum",
        "Data Analysis", "A/B Testing", "Stakeholder Communication",
        "Competitive Analysis", "Wireframing", "SQL (basic)", "OKRs"
    ],
}


class SkillExtractor:
    """Extract skills from resume text using NER / keyword matching."""

    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1,
        )
        # Curated skill vocab for fuzzy matching
        self.skill_vocab = set(
            skill for skills in ROLE_SKILL_MAP.values() for skill in skills
        )

    def extract_from_resume(self, resume_text: str) -> List[str]:
        """Return list of skills detected in resume text."""
        # NER for tech entities
        entities = self.model(resume_text)
        ner_skills = [e["word"].strip() for e in entities if len(e["word"]) > 2]

        # Keyword matching against curated vocab
        matched_skills = [
            skill
            for skill in self.skill_vocab
            if skill.lower() in resume_text.lower()
        ]

        all_skills = list(set(ner_skills + matched_skills))
        return all_skills


class SkillGapAnalyzer:
    """
    Compares extracted skills vs target role requirements.
    Returns gap report with priority recommendations.
    """

    def __init__(self):
        self.extractor = SkillExtractor()

    def analyze(self, resume_text: str, target_role: str) -> Dict:
        if target_role not in ROLE_SKILL_MAP:
            raise ValueError(f"Role '{target_role}' not in taxonomy. Add it first.")

        required_skills = set(ROLE_SKILL_MAP[target_role])
        current_skills = set(self.extractor.extract_from_resume(resume_text))

        gaps = required_skills - current_skills
        strengths = required_skills & current_skills
        extra_skills = current_skills - required_skills

        # Priority: skills at top of list are more foundational
        prioritized_gaps = [
            s for s in ROLE_SKILL_MAP[target_role] if s in gaps
        ]

        report = {
            "target_role": target_role,
            "skills_matched": list(strengths),
            "skill_gaps": prioritized_gaps,
            "extra_skills": list(extra_skills),
            "readiness_score": round(len(strengths) / len(required_skills) * 100, 1),
            "top_3_to_learn": prioritized_gaps[:3],
        }
        return report


# --- Demo ---
if __name__ == "__main__":
    sample_resume = """
    John Doe — Software Engineer
    Skills: Python, Git, SQL, REST APIs, Docker
    Experience: 2 years building Django/FastAPI backends.
    """
    analyzer = SkillGapAnalyzer()
    report = analyzer.analyze(sample_resume, "Software Engineer")
    print(json.dumps(report, indent=2))
    # Output includes readiness_score, gaps, top 3 things to learn next
