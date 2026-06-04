# Llm Sustainability Analysis
This project analyzes how user behavior and prompt design affect resource efficiency in large language model interactions, using both real-world datasets and controlled experiments.


📌 Project Overview

Large Language Models are increasingly integrated into everyday workflows. However, their usage patterns vary significantly in efficiency, often leading to unnecessary computational overhead.

This project investigates how prompt design and user interaction patterns influence resource consumption in LLM-based systems, using a combination of dataset analysis and controlled experiments.

❓ Research Questions

- How does prompt structure affect token consumption and response length?
- Are there identifiable, more efficient topics or tasks in the prompts?
- Can usage efficiency be modeled using machine learning?

📊 Data Sources

This project uses a combination of:

- Open dialogue datasets (e.g., ShareGPT-style conversations)
- Synthetic prompt experiments generated for controlled comparison

⚙️ Methodology

1. Exploratory Data Analysis
- Prompt length distribution
- Token usage patterns
- Novelty Embedding
2. Topic Modeling
- Sentence Transformer
- CountVectorizer 
- UMAP
- BERTopic Modeling
3. Efficiency Modeling
- Regression models to predict target_success and target_cost
- Feature importance analysis (SHAP / permutation importance)
4. Controlled Experiments
- Comparison of prompt variants
- Measurement of efficiency vs output quality trade-offs

📈 Metrics

- target success: ChatGPT responds without asking for clarification [~df["first_response"].str.contains(r"?")]
- target cost: Minimum number of tokens in the first response

🧠 Key Tools & Libraries

- Python (pandas, numpy)
- scikit-learn
- matplotlib / seaborn
- sentence-transformers
- tiktoken

📌 Expected Outcomes

- Identification of inefficient LLM usage patterns
- Quantitative model of prompt efficiency
- Evidence-based recommendations for efficient prompting

data versions
01_raw: original downloaded files
02_processed: cleaned and language-filtered conversations
03_features: engineered conversation-level feature tables


⚠️ Disclaimer

This project uses proxy metrics for resource consumption. Actual energy usage of LLM systems is not directly measured but approximated via token counts and interaction complexity.
