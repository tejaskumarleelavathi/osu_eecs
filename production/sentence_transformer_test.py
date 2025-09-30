from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the model

embedding_models = [
    "BAAI/bge-m3",
    "WhereIsAI/UAE-Large-V1",
    "ibm-granite/granite-embedding-125m-english",
    "NovaSearch/stella_en_400M_v5",
    "Snowflake/snowflake-arctic-embed-m-v1.5",
"sentence-transformers/allenai-specter",
"allenai/scibert_scivocab_uncased",
"sentence-transformers/all-MiniLM-L6-v2",
"Lajavaness/bilingual-embedding-small",
"intfloat/e5-small-v2"
]
# model = SentenceTransformer("Linq-AI-Research/Linq-Embed-Mistral")
# model = SentenceTransformer("Lajavaness/bilingual-embedding-small",trust_remote_code=True)
# model = SentenceTransformer("NovaSearch/stella_en_400M_v5",trust_remote_code=True)

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(embedding_models[0])

# 1. MODEL SIZE (Parameters)
print("1. MODEL SIZE (Parameters):")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = total_params * 4 / (1024**2)  # Assuming float32

# Convert to millions
total_params_m = total_params / 1_000_000
trainable_params_m = trainable_params / 1_000_000

print(f"   Total parameters: {total_params_m:.1f}M ({total_params:,})")
print(f"   Trainable parameters: {trainable_params_m:.1f}M ({trainable_params:,})")
print(f"   Model size: {model_size_mb:.1f} MB")
print(f"   Purpose: More parameters = more computation = slower inference")
print(f"   Impact: Directly proportional to inference time\n")

# 3. MAXIMUM SEQUENCE LENGTH
print("2. MAXIMUM SEQUENCE LENGTH:")
if hasattr(model, 'tokenizer'):
    max_length = getattr(model.tokenizer, 'model_max_length', 'Unknown')
    print(f"   Max sequence length: {max_length} tokens")
    
    # Test with different lengths
    test_texts = [
        "Short text",
        "This is a medium length text with several words to test tokenization",
        "This is a much longer text that contains many more words and should demonstrate how tokenization works with longer sequences and how it affects the maximum sequence length parameter" * 2
    ]
    
    for i, text in enumerate(test_texts):
        tokens = model.tokenizer.encode(text)
        print(f"   Test text {i+1}: {len(tokens)} tokens (length: {len(text)} chars)")
else:
    print("   No tokenizer found")

print(f"   Purpose: Defines computational complexity - longer sequences = quadratic growth")
print(f"   Impact: Even short texts use full attention matrix up to max_length\n")

# 4. EMBEDDING DIMENSION
print("3. EMBEDDING DIMENSION:")
test_embedding = model.encode(["Test sentence for embedding"])
embedding_dim = test_embedding.shape[1]
print(f"   Embedding dimension: {embedding_dim}")
print(f"   Output shape: {test_embedding.shape}")

# Memory calculation
embedding_memory_kb = embedding_dim * 4 / 1024  # float32 in KB
print(f"   Memory per embedding: {embedding_memory_kb:.2f} KB")
print(f"   Purpose: Higher dimensions can capture more semantic information")
print(f"   Impact: Affects final layer computation and memory usage\n")
    

print(f"Using model {embedding_models[0]}")
# Encode the query and passages
# query = "give me papers published by Alan Fern"
query = "Provide publications having Alan Fern as an author"
query = "Provide papers having Alan Fern as an author"

query_embedding = model.encode([query])  # Note that we wrap the query in a list

passages = [
    """Columbia. His research areas are in data mining and machine learning, with
specific interests in anomaly detection, surveillance algorithms, and mining large
scale datasets. His Ph.D. thesis was entitled "Data Mining Algorithms for the Early
Detection of Disease Outbreaks" and he is involved in the field of disease
outbreak surveillance.
Additional Links
Learn more about Weng-Keen Wong
Related StoriesContact Info
College of Engineering
Oregon State University
101 Covell Hall""",
    """Sim-To-Real Learning for Humanoid Box Loco-Manipulation
Jeremy Dao, Helei Duan, Alan Fern
(Accepted) IEEE International Conference on Robotics and Automation (ICRA), May 2024
Supplemental Video
2023
Optimizing Bipedal Locomotion for The 100m Dash With Comparison to Human Running
Devin Crowley, Jeremy Dao, Helei Duan, Kevin Green, Jonathan Hurst, Alan Fern
IEEE International Conference on Robotics and Automation (ICRA), July 2023
2022""",
    """IEEE International Conference on Robotics and Automation (ICRA), July 2023
2022
Dynamic Bipedal Turning through Sim-to-Real Reinforcement Learning
Fangzhou Yu, Ryan Batke, Jeremy Dao, Jonathan Hurst, Kevin Green, Alan Fern
IEEE International Conference on Humanoid Robots (Humanoids), November 2022
Preprint
Optimizing Bipedal Maneuvers of Single Rigid-Body Models for Reinforcement Learning
Ryan Batke, Fangzhou Yu, Jeremy Dao, Jonathan Hurst, Ross L. Hatton, Alan Fern, Kevin
Green"""
]

# Encode the passages
passage_embeddings = model.encode(passages)

# Compute cosine similarity
similarity_scores = cosine_similarity(query_embedding, passage_embeddings)

# Display similarity scores
print("Similarity scores:", similarity_scores)

# Find the most similar passage
most_similar_idx = np.argmax(similarity_scores)
print(f"The most similar passage is: '{passages[most_similar_idx]}' with a similarity score of {similarity_scores[0][most_similar_idx]}")
