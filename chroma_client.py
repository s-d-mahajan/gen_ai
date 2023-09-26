import chromadb
from chromadb.utils import embedding_functions

from processing import create_chunks

client = chromadb.PersistentClient(path="/Users/saurabhmahajan/Saurabh/gen_ai/data")

ef = embedding_functions.InstructorEmbeddingFunction(model_name="hkunlp/instructor-xl")
collection = client.get_collection(name="my_collection", embedding_function=ef)


def insert():
    chunks = create_chunks()
    for id, _chunk in enumerate(chunks):
        collection.add(
            documents=[_chunk],
            metadatas=[{"chunk": id}],
            ids=str(id)
        )


def run_query(query):
    response = collection.query(
        query_texts=[query],
        n_results=2
    )

    for r in response["documents"][0]:
        print("found answer in \n")
        print("______________")
        print(r)
        print("\n")
    return response
