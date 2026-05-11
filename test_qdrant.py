from qdrant_client import QdrantClient
import requests

print("=== test 1: requests ===")
r = requests.get("http://localhost:6333/collections", timeout=10)
print("status:", r.status_code, "body:", r.text[:200])

print("=== test 2: requests на 127.0.0.1 ===")
r = requests.get("http://127.0.0.1:6333/collections", timeout=10)
print("status:", r.status_code, "body:", r.text[:200])

print("=== test 3: QdrantClient localhost ===")
client = QdrantClient(host="localhost", port=6333, prefer_grpc=False, https=False, timeout=60)
print("collections:", client.get_collections())

print("=== test 4: QdrantClient 127.0.0.1 ===")
client = QdrantClient(host="127.0.0.1", port=6333, prefer_grpc=False, https=False, timeout=60)
print("collections:", client.get_collections())
