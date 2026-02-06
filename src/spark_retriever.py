# TF-IDF based chunk retrieval using PySpark ML

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF


class SparkRetriever:
    """Finds the most relevant chunks using TF-IDF cosine similarity."""

    def __init__(self, spark, num_features=1 << 18):
        self.spark = spark
        self.num_features = num_features
        self.model = None
        self.index_df = None

    def build_index(self, chunks_df):
        """Tokenize chunks, compute TF-IDF vectors, and store with precomputed norms."""

        # Build the TF-IDF pipeline: text -> tokens -> term frequencies -> IDF weights
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+")
        hashing_tf = HashingTF(inputCol="tokens", outputCol="tf", numFeatures=self.num_features)
        idf = IDF(inputCol="tf", outputCol="features")

        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
        self.model = pipeline.fit(chunks_df)

        # Precompute vector norms for fast cosine similarity later
        norm_udf = F.udf(lambda v: float(v.norm(2)) if v is not None else 0.0, DoubleType())

        self.index_df = (
            self.model.transform(chunks_df)
            .withColumn("norm", norm_udf(F.col("features")))
            .select("year", "chunk_id", "text", "features", "norm")
        )

    def retrieve(self, query, year=None, top_k=5):
        """Find the top-K most similar chunks for a query using cosine similarity."""
        if self.model is None or self.index_df is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Turn the query into a TF-IDF vector using the same pipeline
        query_df = self.spark.createDataFrame([{"text": query}])
        q_vec = self.model.transform(query_df).select("features").collect()[0]["features"]
        q_norm = float(q_vec.norm(2)) if q_vec is not None else 0.0

        # Cosine similarity = dot(A, B) / (norm(A) * norm(B))
        def cosine(v, v_norm):
            if v is None or v_norm is None or v_norm == 0.0 or q_norm == 0.0:
                return 0.0
            return float(v.dot(q_vec) / (v_norm * q_norm))

        cosine_udf = F.udf(cosine, DoubleType())

        # Filter by year if given
        df = self.index_df
        if year is not None:
            df = df.filter(F.col("year") == int(year))

        # Score, sort, and return top results
        results = (
            df.withColumn("score", cosine_udf(F.col("features"), F.col("norm")))
            .filter(F.col("score") > 0)
            .orderBy(F.col("score").desc())
            .limit(top_k)
            .select("year", "chunk_id", "text", "score")
            .collect()
        )

        return [
            {"year": r["year"], "chunk_id": r["chunk_id"], "text": r["text"], "score": float(r["score"])}
            for r in results
        ]

    def retrieve_for_variable(self, variable_name, year, queries, top_k=3):
        """Run multiple queries for one variable, deduplicate, and return top results."""
        all_results = []
        seen = set()

        for query in queries:
            for r in self.retrieve(query, year=year, top_k=top_k):
                chunk_key = (r["year"], r["chunk_id"])
                if chunk_key not in seen:
                    seen.add(chunk_key)
                    r["variable"] = variable_name
                    r["query"] = query
                    all_results.append(r)

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k * 2]
