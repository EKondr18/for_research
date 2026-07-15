"use client";

import { useState, FormEvent } from "react";
import styles from "./page.module.css";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

type Source = {
  file_name: string;
  page: number;
  drive_link: string;
};

type AskResponse = {
  answer: string;
  sources: Source[];
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      if (!API_URL) {
        throw new Error(
          "Не задан NEXT_PUBLIC_API_URL — адрес backend'а на Render не сконфигурирован."
        );
      }
      const resp = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        throw new Error(body?.detail ?? `Backend вернул ошибку ${resp.status}`);
      }

      const data: AskResponse = await resp.json();
      setResult(data);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Не удалось получить ответ. Backend на Render мог заснуть после простоя — попробуйте ещё раз через 30-60 секунд.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <h1 className={styles.title}>Поиск и вопросы по документам</h1>
        <p className={styles.subtitle}>
          Задайте вопрос — ответ будет найден по содержимому PDF-документов из
          подключённой папки Google Drive.
        </p>

        <form onSubmit={handleSubmit} className={styles.form}>
          <textarea
            className={styles.textarea}
            placeholder="Например: какие сроки указаны в договоре поставки?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            disabled={loading}
          />
          <button
            type="submit"
            className={styles.button}
            disabled={loading || !question.trim()}
          >
            {loading ? "Ищу ответ…" : "Спросить"}
          </button>
        </form>

        {error && <div className={styles.error}>{error}</div>}

        {result && (
          <div className={styles.result}>
            <h2 className={styles.answerHeading}>Ответ</h2>
            <p className={styles.answerText}>{result.answer}</p>

            {result.sources.length > 0 && (
              <>
                <h3 className={styles.sourcesHeading}>Источники</h3>
                <ul className={styles.sourcesList}>
                  {result.sources.map((s, i) => (
                    <li key={i} className={styles.sourceItem}>
                      <a href={s.drive_link} target="_blank" rel="noopener noreferrer">
                        {s.file_name}
                      </a>
                      <span className={styles.sourcePage}> — стр. {s.page}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
