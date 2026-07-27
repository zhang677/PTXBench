# Prompt construction
1. Prepare primitives in /structual_doc/headers. Reference: Triton-distributed and MatmulTutorial
2. Add background knowledge required by primitives to /structural_doc/document. Reference: PTX, CUDA, and CUDA Runtime docs
3. Run `python build_doc_v2.py --force` to update base prompts in /multiturn/prompt_configs
4. Run `extract_reference_fns.py` to generate a full list of primitives
5. Prepare patterns in /structual_doc/patterns. Reference: Triton-distributed and MatmulTutorial
6. Run `simple_lsp.py` to check primitves used by patterns are covered by /structual_doc/headers

# Trajectory collection
1. Prepare prompt configs in AccRL-exps/prompt_configs
2. Run `run_parallel_v2.py`

# Trajectory analysis
1. Analyze the diversity using AST by running `analyze_patterns_batch.py` (statistics) and `draw_pattern_trees_batch.py` (figures)
2. Run `plot_turn_categories.py` to identify:
    - infra bugs
    - primitive usage at PTX level
    - best speedup
    - llm api bugs ("Extraction error")
3. Run `plot_token_breakdown.py` to identify:
    - reasoning behaviors
    - llm api bugs (unstable reasonings)
4. Run `plot_success_trajectories` to identify:
    - per trajectory evolution
5. Run `analyze_kernel_per_turn` to identify:
    - env bugs (feedback messages)
    - inspect actual kernels to inform #Prompt construction and #Trajectory collection

# Benefits
1. LLM can work as a "just-in-time" compiler that conducts PTX instruction polymorphism:
    - Prompt tma_load_2d => Generate tma_load_4d
2. LLM can do some routine compiler optimizations with high probability:
    - Constant value replacement
3. LLM can have higher success rate from utilizing human architects’ knowledge
4. Make it easier to measure diversity in a constrained action space using static analysis

