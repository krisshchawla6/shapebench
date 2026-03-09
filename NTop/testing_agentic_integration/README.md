# Agentic Design Optimization - Testing Suite

## Milestone Testing Strategy

### Milestone 1: Template Generation
**File:** `test_milestone_1.py`  
**Tests:** ntopcl -t command runs successfully and generates input_template.json  
**Run:** `python test_milestone_1.py`

### Milestone 2: AI Variation Generation
**File:** `test_milestone_2.py`  
**Tests:** Gemini generates valid design variations from template  
**Run:** `python test_milestone_2.py`

### Milestone 3: Full Integration
**File:** `test_milestone_3.py`  
**Tests:** Complete main.py workflow (setup → generation → saving)  
**Run:** `python test_milestone_3.py`

### Final Production Test
Once all milestones pass, run the full system:
```bash
cd ../design_optimization
python main.py ../testing/plane_model.ntop "Create lightweight high-altitude aircraft variants" --variations 5 --images ../testing/images
```

## Requirements
- `.env` file with: `ntopcl_path`, `ntop_username`, `ntop_password`, `gemini_key`
- nTop license active
- Google Gemini API access
