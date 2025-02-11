from ares import ARES

synth_config = { 
    "document_filepaths": ["/nq_ratio_0.5.tsv"],
    "few_shot_prompt_filename": "/hotpotqa_few_shot_prompt_for_synthetic_query_generation.tsv",
    "synthetic_queries_filenames": ["/data/synthetic_hotpotQA_output.tsv"],
    "model_choice": "google/flan-t5-xxl",
    "documents_sampled": 10
}

ares = ARES(synthetic_query_generator=synth_config)
results = ares.generate_synthetic_data()
print(results)