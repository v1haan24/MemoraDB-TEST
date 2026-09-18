# MemoraDB ONNX embedding runtime

MemoraDB no longer requires Python, `sentence-transformers`, PyTorch, pip, or
an embedding queue worker. Semantic embeddings are created directly inside the
database process using the open-source `sentence-transformers/all-MiniLM-L6-v2`
model and ONNX Runtime.

## Build

Configure and build normally:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

During configuration CMake downloads the pinned ONNX Runtime release and the
complete inference assets into `build/models/all-MiniLM-L6-v2/`:

- `model.onnx` — MiniLM encoder graph
- `vocab.txt` — WordPiece tokenizer vocabulary
- `tokenizer_config.json` and `special_tokens_map.json` — tokenizer metadata
- `config.json` — model architecture metadata

The build copies those assets alongside `memora`, together with
`onnxruntime.dll` on Windows. Start only `memora`; semantic INSERT, UPDATE, and
`SIMILAR TO` operations run inference directly in that process.

MemoraDB performs WordPiece tokenization, adds `[CLS]`/`[SEP]`, runs the ONNX
graph, mean-pools the token embeddings, and L2-normalizes the resulting
384-dimensional vector. This matches the sentence-transformers encoding
pipeline used by the previous worker.
