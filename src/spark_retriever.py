"""
TF-IDF retrieval implemented with PySpark ML.

This module provides a SparkRetriever class that builds a TF-IDF index
from document chunks and supports similarity-based retrieval for RAG pipelines.
"""

from typing import List, Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF


class SparkRetriever:
    """
    Retriever that finds the most relevant chunks using TF-IDF in Spark.

    This class implements a lexical retrieval system using Term Frequency-Inverse
    Document Frequency (TF-IDF) vectorization. It's designed for use in RAG pipelines
    where relevant document chunks need to be retrieved based on query similarity.

    Attributes:
        spark: Active SparkSession instance.
        num_features: Dimensionality of the TF-IDF feature vectors.
        model: Fitted Spark ML Pipeline (Tokenizer -> HashingTF -> IDF).
        index_df: DataFrame containing indexed chunks with TF-IDF vectors.

    Example:
        >>> retriever = SparkRetriever(spark)
        >>> retriever.build_index(chunks_df)
        >>> results = retriever.retrieve("capital markets hedging", top_k=5)
    """

    def __init__(self, spark: SparkSession, num_features: int = 1 << 18):
        """
        Initialize the SparkRetriever.

        Args:
            spark: Active SparkSession instance for distributed processing.
            num_features: Number of features for the hashing TF vectorizer.
                         Default is 2^18 (262144) which balances memory and accuracy.
        """
        self.spark = spark
        self.num_features = num_features
        self.model = None
        self.index_df = None

    def build_index(self, chunks_df: DataFrame) -> None:
        """
        Build a TF-IDF index from document chunks.

        Creates a Spark ML pipeline that tokenizes text, computes term frequencies
        using hashing, and applies IDF weighting. The resulting vectors are stored
        with precomputed norms for efficient cosine similarity calculation.

        Args:
            chunks_df: Spark DataFrame with columns:
                - year (int): Document year
                - chunk_id (int): Unique chunk identifier
                - text (str): Chunk text content

        Returns:
            None. Sets self.model and self.index_df as side effects.

        Raises:
            ValueError: If chunks_df is empty or missing required columns.
        """
        # STEP 1: Create the TF-IDF pipeline components
        # Tokenizer: splits text into words using non-word characters as delimiters
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+")
        # HashingTF: converts words to term frequency vectors using hashing trick
        hashing_tf = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=self.num_features)
        # IDF: applies Inverse Document Frequency weighting (rare words get higher scores)
        idf = IDF(inputCol="tf", outputCol="features")

        # STEP 2: Fit the pipeline on all chunks to learn IDF weights
        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
        self.model = pipeline.fit(chunks_df)

        # STEP 3: Compute vector norms for efficient cosine similarity later
        def _norm(v):
            """Calculate L2 (Euclidean) norm of a vector."""
            return float(v.norm(2)) if v is not None else 0.0

        norm_udf = F.udf(_norm, DoubleType())  # Make it a Spark UDF

        # STEP 4: Transform chunks and store with precomputed norms
        self.index_df = (
            self.model.transform(chunks_df)
            .withColumn("norm", norm_udf(F.col("features")))  # Add norm column
            .select("year", "chunk_id", "text", "features", "norm")
        )

    def retrieve(self, query: str, year: int = None, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the top-K most similar chunks for a given query.

        Computes cosine similarity between the query's TF-IDF vector and all
        indexed chunk vectors, returning the highest-scoring matches.

        Args:
            query: Search query string to match against indexed chunks.
            year: Optional year filter. If provided, only chunks from that
                  year are considered. If None, searches all chunks.
            top_k: Maximum number of results to return. Default is 5.

        Returns:
            List of dictionaries, each containing:
                - year (int): Document year
                - chunk_id (int): Chunk identifier
                - text (str): Chunk text content
                - score (float): Cosine similarity score (0.0 to 1.0)

        Raises:
            ValueError: If build_index() has not been called first.

        Example:
            >>> results = retriever.retrieve("hedging effect", top_k=3)
            >>> print(results[0]['score'])  # e.g., 0.847
        """
        # Ensure build_index() was called first
        if self.model is None or self.index_df is None:
            raise ValueError("Index not built. Call build_index() first.")

        # STEP 1: Convert query to TF-IDF vector using the same pipeline
        query_df = self.spark.createDataFrame([{"text": query}])
        q_row = self.model.transform(query_df).select("features").collect()[0]
        q_vec = q_row["features"]  # Query's TF-IDF vector
        q_norm = float(q_vec.norm(2)) if q_vec is not None else 0.0  # Query's L2 norm

        # STEP 2: Define cosine similarity function
        # Cosine similarity = (A · B) / (||A|| × ||B||)
        def _cosine(v, v_norm):
            """Compute cosine similarity between query vector and a chunk vector."""
            if v is None or v_norm is None or v_norm == 0.0 or q_norm == 0.0:
                return 0.0
            return float(v.dot(q_vec) / (v_norm * q_norm))

        cosine_udf = F.udf(_cosine, DoubleType())  # Make it a Spark UDF

        # STEP 3: Filter by year if specified
        df = self.index_df
        if year is not None:
            df = df.filter(F.col("year") == int(year))

        # STEP 4: Compute similarity scores and return top-K results
        results = (
            df.withColumn("score", cosine_udf(F.col("features"), F.col("norm")))
            .filter(F.col("score") > 0)  # Only keep chunks with some similarity
            .orderBy(F.col("score").desc())  # Highest scores first
            .limit(top_k)  # Take only top K results
            .select("year", "chunk_id", "text", "score")
            .collect()  # Bring results to driver
        )

        return [
            {
                "year": r["year"],
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "score": float(r["score"]),
            }
            for r in results
        ]

    def retrieve_for_variable(
        self, variable_name: str, year: int, queries: List[str], top_k: int = 3
    ) -> List[Dict]:
        """
        Run multiple queries for one variable and merge deduplicated results.

        Executes several related queries (e.g., different phrasings of the same
        concept) and combines their results. Duplicate chunks are removed and
        results are sorted by score.

        Args:
            variable_name: Name of the variable being extracted (e.g.,
                          "capital_markets_hedging_effect"). Added to results
                          for traceability.
            year: Year filter for retrieval, or None to search all years.
            queries: List of query strings to execute. Multiple queries help
                    capture different phrasings of the target information.
            top_k: Number of top results per individual query. Final results
                   may contain up to top_k * 2 unique chunks.

        Returns:
            List of dictionaries sorted by score (descending), each containing:
                - year (int): Document year
                - chunk_id (int): Chunk identifier
                - text (str): Chunk text content
                - score (float): Cosine similarity score
                - variable (str): The variable_name parameter
                - query (str): The query that retrieved this chunk

        Example:
            >>> queries = ["Capital Markets hedging", "effect was billion"]
            >>> results = retriever.retrieve_for_variable(
            ...     "capital_markets_hedging_effect", None, queries, top_k=5
            ... )
        """
        all_results = []  # Combined results from all queries
        seen_chunk_ids = set()  # Track which chunks we've already seen (avoid duplicates)

        # Run each query and collect unique results
        for query in queries:
            results = self.retrieve(query, year=year, top_k=top_k)
            for r in results:
                # Use (year, chunk_id) tuple as unique identifier
                chunk_key = (r["year"], r["chunk_id"])
                if chunk_key not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_key)
                    r["variable"] = variable_name  # Add metadata for traceability
                    r["query"] = query  # Track which query found this chunk
                    all_results.append(r)

        # Sort by score descending and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[: top_k * 2]  # Return up to 2x top_k unique chunks
