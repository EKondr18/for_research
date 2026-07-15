import logging

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import MissingConfigError, settings
from drive import DriveAccessError, DriveClient
from embeddings import embed_passages, embed_query, get_embedding_dim
from llm import GroqProvider
from pdf_processing import chunk_pages, extract_pages_text
from qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="PDF Semantic Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_qdrant_store() -> QdrantStore:
    return QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        chunks_collection=settings.chunks_collection,
        meta_collection=settings.meta_collection,
    )


def get_llm() -> GroqProvider:
    return GroqProvider(api_key=settings.groq_api_key, model=settings.groq_model)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    status = {}

    try:
        DriveClient(settings.google_drive_api_key, settings.google_drive_folder_id).list_pdf_files()
        status["google_drive"] = "ok"
    except MissingConfigError as e:
        status["google_drive"] = f"error: {e}"
    except DriveAccessError as e:
        status["google_drive"] = f"error: {e}"
    except Exception as e:
        status["google_drive"] = f"error: {e}"

    try:
        store = get_qdrant_store()
        store.client.get_collections()
        status["qdrant"] = "ok"
    except MissingConfigError as e:
        status["qdrant"] = f"error: {e}"
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    try:
        get_llm().health_check()
        status["groq"] = "ok"
    except MissingConfigError as e:
        status["groq"] = f"error: {e}"
    except Exception as e:
        status["groq"] = f"error: {e}"

    overall_ok = all(v == "ok" for v in status.values())
    return {"status": "ok" if overall_ok else "degraded", "services": status}


# ---------------------------------------------------------------------------
# POST /reindex
# ---------------------------------------------------------------------------

@app.post("/reindex")
def reindex(x_reindex_token: str | None = Header(default=None)):
    if settings.reindex_token is not None and x_reindex_token != settings.reindex_token:
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-Reindex-Token")

    try:
        drive = DriveClient(settings.google_drive_api_key, settings.google_drive_folder_id)
        store = get_qdrant_store()
        embedding_dim = get_embedding_dim(settings.embedding_model)
        store.ensure_collections(embedding_dim)

        remote_files = drive.list_pdf_files()
    except MissingConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except DriveAccessError as e:
        raise HTTPException(status_code=502, detail=str(e))

    indexed = store.get_indexed_files()

    to_process = [f for f in remote_files if indexed.get(f.id) != f.modified_time]
    logger.info("К индексации: %d из %d файлов (новые/изменённые)", len(to_process), len(remote_files))

    processed, skipped_no_text, errors = [], [], []

    for f in to_process:
        try:
            pdf_bytes = drive.download_file(f.id, f.name)
        except DriveAccessError as e:
            logger.error("Пропуск файла '%s': %s", f.name, e)
            errors.append({"file": f.name, "error": str(e)})
            continue

        pages = extract_pages_text(pdf_bytes, f.name)
        if not pages:
            skipped_no_text.append(f.name)
            # Still record as "indexed" (with no chunks) so we don't retry
            # the same unreadable file on every /reindex call.
            store.delete_chunks_for_file(f.id)
            store.upsert_file_meta(f.id, f.name, f.modified_time)
            continue

        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        texts = [c.text for c in chunks]
        vectors = embed_passages(texts, settings.embedding_model)

        store.delete_chunks_for_file(f.id)
        store.upsert_chunks(f.id, f.name, [(c.text, c.page) for c in chunks], vectors)
        store.upsert_file_meta(f.id, f.name, f.modified_time)

        processed.append({"file": f.name, "chunks": len(chunks)})
        logger.info("Проиндексирован файл '%s': %d чанков", f.name, len(chunks))

    return {
        "total_files_in_drive": len(remote_files),
        "processed": processed,
        "skipped_no_text_layer": skipped_no_text,
        "errors": errors,
        "unchanged_files": len(remote_files) - len(to_process),
    }


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    file_name: str
    page: int
    drive_link: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    try:
        store = get_qdrant_store()
        query_vector = embed_query(question, settings.embedding_model)
        hits = store.search(query_vector, settings.search_top_k)
    except MissingConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not hits:
        return AskResponse(
            answer="В индексе пока нет документов (или не найдено ничего релевантного). "
            "Убедитесь, что был выполнен POST /reindex.",
            sources=[],
        )

    context_parts = []
    sources = []
    for hit in hits:
        payload = hit.payload or {}
        file_name = payload.get("file_name", "unknown")
        page = payload.get("page", 0)
        text = payload.get("text", "")
        drive_file_id = payload.get("drive_file_id", "")
        context_parts.append(f"[{file_name}, стр. {page}]\n{text}")
        sources.append(
            Source(file_name=file_name, page=page, drive_link=DriveClient.file_link(drive_file_id))
        )

    context = "\n\n---\n\n".join(context_parts)

    try:
        llm = get_llm()
        answer = llm.generate_answer(question, context)
    except MissingConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Ошибка вызова Groq API: %s", e)
        raise HTTPException(status_code=502, detail=f"Ошибка обращения к Groq API: {e}")

    return AskResponse(answer=answer, sources=sources)
