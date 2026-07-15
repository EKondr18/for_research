import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Поиск по документам",
  description: "Семантический поиск и ответы на вопросы по PDF-документам",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
