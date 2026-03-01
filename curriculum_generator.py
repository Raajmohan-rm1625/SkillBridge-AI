"""
SkillBridge AI — LLM Curriculum Generator
Runs Llama 3 / Mistral via AMD ROCm + vLLM for high-throughput inference
"""

from vllm import LLM, SamplingParams
import json
from typing import List, Dict

# AMD MI300X: vLLM auto-detects ROCm GPU
# Install: pip install vllm (ROCm build)
# Model: meta-llama/Meta-Llama-3-8B-Instruct (fits in MI300X 192GB HBM3)

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = LLM(
    model=MODEL_ID,
    dtype="float16",          # Use bfloat16 on MI300X for better perf
    gpu_memory_utilization=0.85,
    max_model_len=8192,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1500,
)


CURRICULUM_PROMPT = """You are SkillBridge AI, an expert career coach and curriculum designer.

A learner wants to become a {target_role}.
Their current skills: {current_skills}
Their skill gaps (prioritized): {skill_gaps}
Available time: {hours_per_week} hours/week
Preferred learning style: {learning_style}

Generate a structured {weeks}-week personalized learning curriculum in JSON format.
Each week should have:
- week_number
- focus_topic (the main skill to learn)
- daily_tasks (array of 5 tasks, one per weekday)
- resources (2-3 specific free resources: course name + URL)
- milestone (what the learner can build/demonstrate by end of week)

Respond ONLY with valid JSON, no preamble.
"""


def generate_curriculum(
    target_role: str,
    current_skills: List[str],
    skill_gaps: List[str],
    hours_per_week: int = 10,
    learning_style: str = "visual + hands-on",
    weeks: int = 8,
) -> Dict:
    """Generate a personalized curriculum using Llama 3 on AMD MI300X."""

    prompt = CURRICULUM_PROMPT.format(
        target_role=target_role,
        current_skills=", ".join(current_skills),
        skill_gaps=", ".join(skill_gaps),
        hours_per_week=hours_per_week,
        learning_style=learning_style,
        weeks=weeks,
    )

    messages = [
        {"role": "system", "content": "You are SkillBridge AI curriculum engine."},
        {"role": "user", "content": prompt},
    ]

    # Format for Llama 3 instruct template
    formatted = f"<|begin_of_text|>"
    for msg in messages:
        formatted += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n"

    outputs = llm.generate([formatted], sampling_params)
    raw_text = outputs[0].outputs[0].text.strip()

    # Parse JSON from LLM response
    try:
        # Strip markdown code fences if present
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
        curriculum = json.loads(raw_text)
    except json.JSONDecodeError:
        curriculum = {"raw_output": raw_text, "error": "JSON parse failed"}

    return curriculum


# --- Demo ---
if __name__ == "__main__":
    curriculum = generate_curriculum(
        target_role="Data Scientist",
        current_skills=["Python", "SQL", "Excel"],
        skill_gaps=["Machine Learning", "Statistics", "Deep Learning", "Feature Engineering"],
        hours_per_week=12,
        learning_style="project-based",
        weeks=6,
    )
    print(json.dumps(curriculum, indent=2))
