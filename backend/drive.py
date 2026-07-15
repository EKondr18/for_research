"""Minimal Google Drive v3 client using a plain API key (no OAuth, no Service
Account). Works only against files/folders shared as "Anyone with the link"."""
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger("drive")

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"


class DriveAccessError(RuntimeError):
    """Raised when the API key can list the folder but a specific file (or the
    whole folder) is not actually readable -- almost always means the folder
    isn't really shared as "Anyone with the link", or the Drive API isn't
    enabled for the given API key's project."""


@dataclass
class DriveFile:
    id: str
    name: str
    modified_time: str


class DriveClient:
    def __init__(self, api_key: str, folder_id: str):
        self.api_key = api_key
        self.folder_id = folder_id

    def list_pdf_files(self) -> list[DriveFile]:
        """List all PDF files directly inside the configured folder.
        Raises DriveAccessError with a clear message on auth/permission issues.
        """
        files: list[DriveFile] = []
        page_token: str | None = None
        query = (
            f"'{self.folder_id}' in parents "
            "and mimeType='application/pdf' and trashed=false"
        )
        while True:
            params = {
                "q": query,
                "fields": "nextPageToken, files(id, name, modifiedTime)",
                "pageSize": 100,
                "key": self.api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            resp = requests.get(f"{DRIVE_API_BASE}/files", params=params, timeout=30)
            self._raise_for_drive_error(resp, context=f"листинг папки {self.folder_id}")

            data = resp.json()
            for f in data.get("files", []):
                files.append(DriveFile(id=f["id"], name=f["name"], modified_time=f["modifiedTime"]))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info("Найдено %d PDF-файлов в папке Google Drive", len(files))
        return files

    def download_file(self, file_id: str, file_name: str) -> bytes:
        """Download raw file bytes via alt=media. Raises DriveAccessError with
        a clear, actionable message if the API key can list the folder but
        cannot actually read this specific file's content (common cause: the
        folder sharing setting is not truly "Anyone with the link", or the
        file was moved/trashed/restricted after listing)."""
        params = {"alt": "media", "key": self.api_key}
        resp = requests.get(f"{DRIVE_API_BASE}/files/{file_id}", params=params, timeout=120)
        self._raise_for_drive_error(resp, context=f"скачивание файла '{file_name}' (id={file_id})")
        return resp.content

    def _raise_for_drive_error(self, resp: requests.Response, context: str) -> None:
        if resp.status_code == 200:
            return

        try:
            body = resp.json()
            reason = body.get("error", {}).get("message", resp.text)
        except ValueError:
            reason = resp.text

        if resp.status_code == 403:
            msg = (
                f"Google Drive API вернул 403 Forbidden при операции: {context}. "
                "Наиболее вероятные причины: (1) папка НЕ расшарена как "
                "'Все у кого есть ссылка' (Anyone with the link), (2) в Google Cloud "
                "Console для использованного API-ключа не включён Google Drive API, "
                "(3) на API-ключе настроены ограничения (API restrictions), не "
                f"разрешающие Drive API. Ответ Google: {reason}"
            )
            logger.error(msg)
            raise DriveAccessError(msg)

        if resp.status_code == 404:
            msg = (
                f"Google Drive API вернул 404 Not Found при операции: {context}. "
                "Проверьте, что GOOGLE_DRIVE_FOLDER_ID указан верно и файл/папка "
                f"не были удалены. Ответ Google: {reason}"
            )
            logger.error(msg)
            raise DriveAccessError(msg)

        msg = f"Google Drive API вернул ошибку {resp.status_code} при операции: {context}. Ответ Google: {reason}"
        logger.error(msg)
        raise DriveAccessError(msg)

    @staticmethod
    def file_link(file_id: str) -> str:
        return f"https://drive.google.com/file/d/{file_id}/view"
