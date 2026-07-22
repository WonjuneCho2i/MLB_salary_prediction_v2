// ==UserScript==
// @name         FanGraphs Visible Leaderboard CSV Export
// @namespace    mlb-salary-homework
// @version      1.0.0
// @description  Save the currently visible FanGraphs leaderboard table with playerid and position.
// @match        https://www.fangraphs.com/leaders/major-league*
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  const PANEL_ID = "mlb-salary-fg-export-panel";

  function normalizeText(value) {
    return String(value ?? "")
      .replace(/\u00a0/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    if (/[",\r\n]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  }

  function isVisible(element) {
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  }

  function headerCells(table) {
    const headerRow = table.querySelector("thead tr") || table.querySelector("tr");
    if (!headerRow) return [];
    return Array.from(headerRow.querySelectorAll("th, td")).map((cell) => normalizeText(cell.innerText));
  }

  function findLeaderboardTable() {
    const candidates = Array.from(document.querySelectorAll("table"))
      .filter(isVisible)
      .map((table) => ({
        table,
        headers: headerCells(table),
        rowCount: table.querySelectorAll("tbody tr").length,
      }))
      .filter(({ headers, rowCount }) =>
        headers.includes("Name") &&
        headers.includes("Team") &&
        rowCount > 0 &&
        (headers.includes("PA") || headers.includes("WAR"))
      )
      .sort((a, b) => b.rowCount - a.rowCount || b.headers.length - a.headers.length);

    return candidates.length ? candidates[0] : null;
  }

  function getPlayerMetadata(row, nameIndex) {
    const cells = Array.from(row.querySelectorAll("td"));
    const nameCell = cells[nameIndex];
    const link = nameCell?.querySelector('a[href*="/players/"]');
    if (!link) return { playerid: "", player_position: "" };

    const href = link.getAttribute("href") || "";
    const idMatch = href.match(/\/players\/[^/]+\/(\d+)\//i);
    let playerPosition = "";
    try {
      const url = new URL(href, window.location.origin);
      playerPosition = url.searchParams.get("position") || "";
    } catch (_) {
      playerPosition = "";
    }
    return {
      playerid: idMatch ? idMatch[1] : "",
      player_position: playerPosition,
    };
  }

  function reportName() {
    const type = new URL(window.location.href).searchParams.get("type");
    if (type === "0") return "standard";
    if (type === "6") return "value";
    return `type_${type || "unknown"}`;
  }

  function seasonRange() {
    const url = new URL(window.location.href);
    const seasons = [url.searchParams.get("season"), url.searchParams.get("season1")]
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value) && value > 1800);
    if (!seasons.length) return "unknown_seasons";
    return `${Math.min(...seasons)}_${Math.max(...seasons)}`;
  }

  function downloadCsv() {
    const result = findLeaderboardTable();
    if (!result) {
      setStatus("저장할 Leaderboard 표를 찾지 못했습니다. 표 로딩과 필터를 확인하세요.", true);
      return;
    }

    const { table, headers } = result;
    const nameIndex = headers.indexOf("Name");
    const rows = Array.from(table.querySelectorAll("tbody tr"))
      .map((row) => Array.from(row.querySelectorAll("td")))
      .filter((cells) => cells.length === headers.length);

    if (!rows.length) {
      setStatus("표 데이터 행이 없습니다. 로딩이 끝난 뒤 다시 누르세요.", true);
      return;
    }

    const collectedAt = new Date().toISOString();
    const outputHeaders = [...headers, "playerid", "player_position", "_source_url", "_collected_at"];
    const lines = [outputHeaders.map(csvEscape).join(",")];

    for (const cells of rows) {
      const row = cells[nameIndex]?.closest("tr");
      const metadata = getPlayerMetadata(row, nameIndex);
      const values = cells.map((cell) => normalizeText(cell.innerText));
      values.push(metadata.playerid, metadata.player_position, window.location.href, collectedAt);
      lines.push(values.map(csvEscape).join(","));
    }

    const csv = `\uFEFF${lines.join("\r\n")}`;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = `fangraphs_${reportName()}_${seasonRange()}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
    setStatus(`${rows.length.toLocaleString()}개 행 저장 완료: ${link.download}`, false);
  }

  function setStatus(message, isError) {
    const status = document.querySelector(`#${PANEL_ID} [data-role="status"]`);
    if (!status) return;
    status.textContent = message;
    status.style.color = isError ? "#b91c1c" : "#166534";
  }

  function addPanel() {
    if (document.getElementById(PANEL_ID)) return;
    const panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.style.cssText = [
      "position:fixed",
      "right:18px",
      "bottom:18px",
      "z-index:2147483647",
      "width:290px",
      "padding:12px",
      "background:#ffffff",
      "border:1px solid #9ca3af",
      "border-radius:8px",
      "box-shadow:0 4px 16px rgba(0,0,0,.18)",
      "font:13px/1.4 Arial,sans-serif",
      "color:#111827",
    ].join(";");

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "FG 현재 표 CSV 저장";
    button.style.cssText = [
      "width:100%",
      "padding:9px 10px",
      "border:0",
      "border-radius:6px",
      "background:#2563eb",
      "color:white",
      "font-weight:700",
      "cursor:pointer",
    ].join(";");
    button.addEventListener("click", downloadCsv);

    const note = document.createElement("div");
    note.textContent = "먼저 Split Seasons를 켜고 Page Size를 Infinity로 바꾸세요.";
    note.style.cssText = "margin-top:8px;color:#4b5563";

    const status = document.createElement("div");
    status.dataset.role = "status";
    status.textContent = "표 로딩 완료 후 저장 버튼을 누르세요.";
    status.style.cssText = "margin-top:7px;font-size:12px;color:#4b5563";

    panel.append(button, note, status);
    document.body.appendChild(panel);
  }

  addPanel();
})();

