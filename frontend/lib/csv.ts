/** Minimal CSV parser: handles quoted fields and commas inside quotes. Good enough for
 * typical experiment exports; for anything exotic (embedded newlines in quoted fields
 * across chunks, alternate delimiters) a real CSV library would be worth pulling in. */
export function parseCsv(text: string): Record<string, string>[] {
  const rows = splitCsvRows(text.trim());
  if (rows.length === 0) return [];

  const [header, ...body] = rows;
  return body
    .filter((row) => row.length > 0)
    .map((row) => Object.fromEntries(header.map((key, i) => [key, row[i] ?? ""])));
}

function splitCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];

    if (inQuotes) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

/** Coerce numeric-looking strings to numbers so downstream stats/ML endpoints see real numbers. */
export function coerceRowTypes(rows: Record<string, string>[]): Record<string, unknown>[] {
  return rows.map((row) => {
    const coerced: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(row)) {
      if (value !== "" && !Number.isNaN(Number(value))) {
        coerced[key] = Number(value);
      } else {
        coerced[key] = value;
      }
    }
    return coerced;
  });
}
