# Autoresearch to improve system_prompts 

can you create @ai_specs/s8-autoresearch.md this plan for a coding agent, it derives from 
@opensrc/autoresearch repo so read it for background and approach specificially the program.md need to run 
optimisation where the codes goal is to improve the system prompt for the question asked by learning from 
the llm agent identifying the issue for wrong answers and proposing a rule added to the system prompt. 

## Autoresearch optimisation loop 
The whole flow would start with input model_id 

### Step 1: use prompts-file and llm to provide answer 

Loop for every contract-id to score: 

uv run python agent.py \
    --context-mode raw \
    --model-id eval-raw \
    --question-index 7 \
    --prompts-file prompts/system_prompts_v2.py \
    --sample-size 50 \
    --seed 42 

### Step 2: Run diagnostic llm on incorrect answer how it would update the system prompt for question-index 
Loop for every contract-id to score: 

INPUTS : 
 * current system prompt 
 * provided answer 
 * golden answer 
 * system prompt for triage agent, give it background its role in looking an incorect evals and coming up with an updated.

OUTPUT: 
 * Provide a reason for why the system prompt wasn't able to retrieve the right answer and what needs to change to get this answer correct 

### Step 3: 
AFTER loop complete for Step 1 & Step 2 for all contract-ids.  
It will take the Step 2 outputs for reasons system prompt failed and proposed changes and have the step 1 INPUT system prompt and OUTPUT a new system prompt. 

### Step 4: 
Takes the new system prompt from STEP 3 and scores the NEW system prompt against STEP 1 eval sed 

uv run python agent.py \
    --context-mode raw \
    --model-id exp_v2_i1 \
    --question-index 7 \
    --prompts-file prompts/system_prompts_v2.py \
    --sample-size 50 \
    --seed 42 

After scoring the NEW prompt and comparing the accuracy against Step 1. If it beats score keep the new system_prompt or go back to old system_prompt. 





