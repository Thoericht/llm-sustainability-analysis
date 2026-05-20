# llm-sustainability-analysis
This project analyzes how user behavior and prompt design affect resource efficiency in large language model interactions, using both real-world datasets and controlled experiments.


📌 Project Overview

Large Language Models are increasingly integrated into everyday workflows. However, their usage patterns vary significantly in efficiency, often leading to unnecessary computational overhead.

This project investigates how prompt design and user interaction patterns influence resource consumption in LLM-based systems, using a combination of dataset analysis and controlled experiments.

❓ Research Questions

- Which user interaction patterns lead to inefficient LLM usage?
- How does prompt structure affect token consumption and response length?
- Can usage efficiency be modeled using machine learning?
- Are there identifiable clusters of user behavior in LLM interactions?
- Optional ideas: Orthographics, topic (e.g. email)

📊 Data Sources

This project uses a combination of:

- Open dialogue datasets (e.g., ShareGPT-style conversations)
- OpenAssistant conversation data
- Synthetic prompt experiments generated for controlled comparison

⚙️ Methodology

1. Exploratory Data Analysis
- Prompt length distribution
- Conversation depth analysis
- Token usage patterns
- Interaction structure analysis
2. User Behavior Clustering
- K-Means clustering on interaction features
- Identification of usage archetypes
3. Efficiency Modeling
- Regression models to predict token usage
- Classification of efficient vs inefficient prompts
- Feature importance analysis (SHAP / permutation importance)
4. Controlled Experiments
- Comparison of prompt variants
- Measurement of efficiency vs output quality trade-offs

📈 Metrics

- Input / output token count
- Conversation length (turns)
- Response latency (if available)
- Prompt efficiency score (quality / cost proxy)

🧠 Key Tools & Libraries

- Python (pandas, numpy)
- scikit-learn
- matplotlib / seaborn
- sentence-transformers
- tiktoken

📌 Expected Outcomes

- Identification of inefficient LLM usage patterns
- Quantitative model of prompt efficiency
- Clustering of user interaction types
- Evidence-based recommendations for efficient prompting

⚠️ Disclaimer

This project uses proxy metrics for resource consumption. Actual energy usage of LLM systems is not directly measured but approximated via token counts and interaction complexity.
