## Introduction
The[Kaggle Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data) is suitable for a RAG project. Because though an LLM may already have prior knowledge of some products in this dataset (e.g., what *TWIZZLERS* are), it does not inherently know customer reviews. 

Detailed definition of each column can be found on the kaggle website listed above. The `Score` column represents the user’s rating (1–5) and may serve as a target variable for downstream analysis. Notably, the score distribution is **left-skewed**.

## Vector Database setup

---

### Data Filtering and Preprocessing

To ensure better generalization, I filtered the dataset to only keep users and products that appear more than 300 times (`UserId` and `ProductId` frequency > 300). Rows that contain NA are also removed.

- Original raw data file: `Reviews.csv`
- After EDA and cleaning: `Reviews_clean.csv`

---

### Chunking and Embedding

Text chunking and embedding are performed in `Embedding_chunk.py`.

- Embedding model: `all-MiniLM-L6-v2`
- Chunking method: `RecursiveCharacterTextSplitter`
  - `chunk_size = 100`
  - `chunk_overlap = 20`

After chunking and embedding, the processed data is saved as:

- `Reviews_clean_chunks_emb.csv`

---

### Vector Store Construction

The `vectorspace.py` is used to load the embedded chunks and store them into a **Chroma vector database**.

---

### Vector Store Validation

The `vector_validation.ipynb` validates that the Chroma vector database was successfully built by querying with held-out rows and confirming that similar rows are retrieved correctly. 
The retrieval quality is supported by manual keyword searches in Excel that match the returned results.

---

## RAG

Since this project uses **gpt-4o-mini**, the environment must contain an `OPENAI_API_KEY` variable for the LLM to function properly.

I compared four different setups:

1. **LLM only**
2. **LLM + RAG**
3. **LLM + RAG + HyDE**
4. **LLM + RAG + Summarization**

The evaluation was conducted using the following five questions:

1. *How do five-star and one-star reviews differ in the aspects of taste they mention?*  
2. *For ProductId = B0090X8IPM, what is the overall user sentiment regarding taste, based on a synthesis of multiple reviews?*  
3. *What are the most frequently mentioned negative issues in the Amazon Fine Food Reviews dataset?*  
4. *Which reviews in the Amazon Fine Food Reviews dataset explicitly mention that a product is “too salty,” and what characteristics do these products share?*  
5. *For UserId = A2AWVROFGSZU4E, what is the overall user sentiment regarding taste?*

---

## Results and Observations

The results are stored in `outputs_demo.ipynb`. By comparing the outputs across the four versions, we observe that for relatively general questions (for example, what aspects are typically mentioned in one-star or five-star reviews), the base LLM is able to produce reasonable answers using its pretrained knowledge alone.

However, for more specific questions, such as identifying a particular user’s opinions or summarizing reviews for a specific Amazon Fine Food product, the original LLM fails to provide accurate or grounded answers. This limitation is effectively addressed by introducing **RAG**, which allows the model to retrieve and condition on relevant review text.

Because the five evaluation questions used in this project are relatively short and written in plain English with clear keywords, the **HyDE (Hypothetical Document Embeddings)** approach does not lead to a significant improvement over the basic RAG setup.

In contrast, the **summarization-based RAG** approach produces noticeably better outputs. For example, in Question 5, the summarization step successfully captures the mixed sentiment expressed by `UserId = A2AWVROFGSZU4E`. This improvement is likely due to summarization reducing redundancy and noise across the **top-10 retrieved documents** in RAG. As a result, the final answer can focus on the **dominant sentiment patterns** rather than isolated or repetitive excerpts.

---

## Environment Setup

The `requirements.txt` file lists all required Python packages needed to run the project.


## Author

This project was designed by:

**Kaixuan Chen**  
M.S. in Machine Learning & Data Science  
Northwestern University
