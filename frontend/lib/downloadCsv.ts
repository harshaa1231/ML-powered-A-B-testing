export function downloadCsv(filename: string, rows: Record<string, unknown>[]): void {
  if (rows.length === 0) return;
  const columns = Object.keys(rows[0]);
  const escape = (value: unknown) => {
    const s = String(value ?? "");
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [columns.join(","), ...rows.map((row) => columns.map((c) => escape(row[c])).join(","))].join("\n");

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
