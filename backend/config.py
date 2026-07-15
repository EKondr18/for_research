"""Application configuration loaded from environment variables."""
import os


class MissingConfigError(RuntimeError):
    pass


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingConfigError(f"Переменная окружения {name} не задана")
    return value


class Settings:
    """Lazily-validated settings. Reading a property raises MissingConfigError
    with a clear message if the corresponding env var is absent, instead of
    failing silently or crashing the whole process on import."""

    @property
    def google_drive_api_key(self) -> str:
        return _require("GOOGLE_DRIVE_API_KEY")

    @property
    def google_drive_folder_id(self) -> str:
        return _require("GOOGLE_DRIVE_FOLDER_ID")

    @property
    def qdrant_url(self) -> str:
        return _require("QDRANT_URL")

    @property
    def qdrant_api_key(self) -> str:
        return _require("QDRANT_API_KEY")

    @property
    def groq_api_key(self) -> str:
        return _require("GROQ_API_KEY")

    @property
    def groq_model(self) -> str:
        # openai/gpt-oss-120b — актуальная модель на бесплатном тире Groq (июль 2026),
        # пришла на смену депрекейтнутой llama-3.3-70b-versatile.
        # Можно переопределить, например на qwen/qwen3-32b, если качество на
        # русском покажется хуже.
        return os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    @property
    def reindex_token(self) -> str | None:
        # Необязательный секрет для защиты POST /reindex от посторонних вызовов.
        # Если не задан — эндпоинт открыт для любого, кто знает URL backend'а.
        value = os.environ.get("REINDEX_TOKEN", "").strip()
        return value or None

    @property
    def embedding_model(self) -> str:
        return os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    @property
    def chunks_collection(self) -> str:
        return os.environ.get("QDRANT_CHUNKS_COLLECTION", "pdf_chunks")

    @property
    def meta_collection(self) -> str:
        return os.environ.get("QDRANT_META_COLLECTION", "documents_meta")

    @property
    def chunk_size(self) -> int:
        return int(os.environ.get("CHUNK_SIZE", "800"))

    @property
    def chunk_overlap(self) -> int:
        return int(os.environ.get("CHUNK_OVERLAP", "100"))

    @property
    def search_top_k(self) -> int:
        return int(os.environ.get("SEARCH_TOP_K", "5"))

    @property
    def cors_allow_origins(self) -> list[str]:
        raw = os.environ.get("CORS_ALLOW_ORIGINS", "*")
        return [o.strip() for o in raw.split(",") if o.strip()]


settings = Settings()
