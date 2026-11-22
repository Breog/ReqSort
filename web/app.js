// Lógica de la página.
//
// Qué hace, en orden:
//   1. Carga el modelo de IA (carpeta web/ai/requirements-classifier) dentro del navegador
//      con Transformers.js, la librería de Hugging Face para ejecutar modelos en JavaScript.
//   2. Al pulsar "Clasificar", pasa cada línea del cuadro de texto por el modelo, que devuelve
//      una probabilidad para cada una de las cuatro categorías.
//   3. Revisa la redacción de cada frase con quality.js (reglas, sin IA).
//   4. Dibuja los resultados: una tarjeta detallada para una frase o una lista para varias.
//
// Todo ocurre en el dispositivo del usuario: no hay servidor y el texto no se envía a ningún sitio.

// Transformers.js se descarga de jsDelivr con la versión fijada (3.8.1) para que una
// actualización de la librería no pueda romper la web.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js";
import { analyzeRequirement, describeIssue } from "./quality.js";

// Configuración de Transformers.js: no buscar el modelo en internet (Hugging Face),
// sino en la carpeta ai/ de esta misma web.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL("ai/", document.baseURI).href;

// Nombre de la carpeta del modelo dentro de web/ai/.
const MODEL_ID = "requirements-classifier";
// Las cuatro categorías, con los mismos nombres que se usaron al entrenar el modelo.
const LABELS = ["FUNCTIONAL", "NON_FUNCTIONAL", "BUSINESS_RULE", "CONSTRAINT"];
// Máximo de líneas por clasificación, para que el navegador no se quede colgado con textos enormes.
const MAX_LINES = 50;

// Ejemplos de los botones "Prueba un ejemplo" y de las tarjetas de categorías.
const EXAMPLES = [
  { label: "FUNCTIONAL", text: "Customers can track the location of their parcel on a map in real time." },
  { label: "NON_FUNCTIONAL", text: "The search page must return results in under one second for 95% of queries." },
  { label: "BUSINESS_RULE", text: "Orders above 500 euros require approval from a sales manager." },
  { label: "CONSTRAINT", text: "The backend must be deployed on the company's existing Azure subscription." },
];

// Ejemplo mal redactado para enseñar la revisión de calidad.
const POOR_EXAMPLE = "The system should be fast and user-friendly, and it must support PDF, Word, etc. if possible.";

// Todos los textos de la interfaz, en español (es) e inglés (en).
// En el HTML, cada elemento con data-i18n="clave" recibe de aquí su texto (y data-i18n-placeholder, su placeholder).
// Algunos valores son funciones porque llevan números dentro (porcentajes, contadores...).
const I18N = {
  es: {
    pageTitle: "Clasificador de requisitos",
    brand: "ReqSort",
    eyebrow: "IA en tu navegador",
    title: "Clasifica requisitos de software al instante",
    subtitle: "Escribe un requisito y la IA te dirá si es funcional, no funcional, una regla de negocio o una restricción, y cómo mejorar su redacción.",
    inputLabel: "Requisito (en inglés)",
    placeholder: "Ej.: The system shall send an email to the customer when an order is shipped.",
    hint: "Escribe una frase por línea para clasificar varias a la vez.",
    examplesTitle: "Prueba un ejemplo:",
    batchExample: "Varios a la vez",
    poorExample: "Requisito mejorable",
    classify: "Clasificar",
    loading: "Descargando el modelo de IA…",
    loadingPct: (pct, mb) => `Descargando el modelo de IA… ${pct} % (${mb} MB)`,
    preparing: "Preparando el modelo…",
    ready: "Modelo listo. Solo se descarga la primera vez.",
    classifying: "Clasificando…",
    loadError: "No se pudo cargar el modelo. Recarga la página o prueba con otro navegador.",
    runError: "Ha ocurrido un error al clasificar. Inténtalo de nuevo.",
    empty: "Escribe al menos un requisito.",
    tooMany: (n) => `Solo se clasifican las primeras ${n} líneas.`,
    confidence: "confianza",
    batchTitle: (n) => `${n} requisitos clasificados`,
    batchReview: (k) => `${k} con sugerencias de mejora`,
    qualityTitle: "Calidad de la redacción",
    qualityLevels: { ok: "Clara", minor: "Se puede mejorar", review: "Conviene revisarla" },
    qualityOk: "No se han detectado términos vagos, ambigüedades ni otros problemas habituales.",
    qualityPill: (n) => (n === 1 ? "1 sugerencia" : `${n} sugerencias`),
    showSuggestions: (n) => (n === 1 ? "Ver 1 sugerencia" : `Ver ${n} sugerencias`),
    templateTitle: "Estructura recomendada",
    templateExample: "Ejemplo:",
    qualityNote: "Revisión automática con reglas basadas en ISO/IEC/IEEE 29148. Es orientativa y puede dar falsos positivos.",
    categoriesTitle: "Las cuatro categorías",
    footerModel: "Modelo DistilBERT ajustado con ~1.400 requisitos · 96 % de acierto en un test de 400 frases.",
    footerPrivacy: "Todo se ejecuta en tu dispositivo: el texto que escribes no se envía a ningún servidor.",
    labels: {
      FUNCTIONAL: "Funcional",
      NON_FUNCTIONAL: "No funcional",
      BUSINESS_RULE: "Regla de negocio",
      CONSTRAINT: "Restricción",
    },
    descriptions: {
      FUNCTIONAL: "Describe algo que el sistema debe hacer: una acción, una función o un comportamiento que ve el usuario.",
      NON_FUNCTIONAL: "Describe cómo de bien debe funcionar el sistema: rendimiento, disponibilidad, seguridad, usabilidad o accesibilidad.",
      BUSINESS_RULE: "Describe una política de la organización: precios, plazos, aprobaciones, límites o condiciones de elegibilidad.",
      CONSTRAINT: "Describe un límite impuesto desde fuera: tecnología obligatoria, estándares, formatos, normativa, presupuesto o plazos.",
    },
  },
  en: {
    pageTitle: "Requirement Classifier",
    brand: "ReqSort",
    eyebrow: "AI in your browser",
    title: "Classify software requirements instantly",
    subtitle: "Write a requirement and the AI will tell you whether it is functional, non-functional, a business rule or a constraint, and how to word it better.",
    inputLabel: "Requirement (in English)",
    placeholder: "E.g.: The system shall send an email to the customer when an order is shipped.",
    hint: "Write one sentence per line to classify several at once.",
    examplesTitle: "Try an example:",
    batchExample: "Several at once",
    poorExample: "Needs work",
    classify: "Classify",
    loading: "Downloading the AI model…",
    loadingPct: (pct, mb) => `Downloading the AI model… ${pct}% (${mb} MB)`,
    preparing: "Preparing the model…",
    ready: "Model ready. It is only downloaded the first time.",
    classifying: "Classifying…",
    loadError: "The model could not be loaded. Reload the page or try another browser.",
    runError: "Something went wrong while classifying. Please try again.",
    empty: "Write at least one requirement.",
    tooMany: (n) => `Only the first ${n} lines are classified.`,
    confidence: "confidence",
    batchTitle: (n) => `${n} requirements classified`,
    batchReview: (k) => `${k} with suggestions`,
    qualityTitle: "Wording quality",
    qualityLevels: { ok: "Clear", minor: "Could be improved", review: "Needs review" },
    qualityOk: "No vague terms, ambiguities or other common problems were found.",
    qualityPill: (n) => (n === 1 ? "1 suggestion" : `${n} suggestions`),
    showSuggestions: (n) => (n === 1 ? "Show 1 suggestion" : `Show ${n} suggestions`),
    templateTitle: "Recommended structure",
    templateExample: "Example:",
    qualityNote: "Automatic review with rules based on ISO/IEC/IEEE 29148. It is a guide and may flag false positives.",
    categoriesTitle: "The four categories",
    footerModel: "DistilBERT fine-tuned on ~1,400 requirements · 96% accuracy on a 400-sentence test set.",
    footerPrivacy: "Everything runs on your device: the text you write is never sent to a server.",
    labels: {
      FUNCTIONAL: "Functional",
      NON_FUNCTIONAL: "Non-functional",
      BUSINESS_RULE: "Business rule",
      CONSTRAINT: "Constraint",
    },
    descriptions: {
      FUNCTIONAL: "Describes something the system must do: an action, a feature or a behaviour the user sees.",
      NON_FUNCTIONAL: "Describes how well the system must work: performance, availability, security, usability or accessibility.",
      BUSINESS_RULE: "Describes an organisational policy: prices, deadlines, approvals, limits or eligibility conditions.",
      CONSTRAINT: "Describes a limit imposed from outside: mandatory technology, standards, formats, regulations, budget or deadlines.",
    },
  },
};

// ---------- estado y elementos de la página ----------
// $ es un atajo para buscar un elemento del HTML por su id.
const $ = (id) => document.getElementById(id);
const input = $("input");
const button = $("classify");
const statusBox = $("status");
const statusText = $("status-text");
const progress = $("progress");
const progressBar = $("progress-bar");
const results = $("results");

let lang = detectLanguage();       // idioma actual de la interfaz ("es" o "en")
let classifier = null;             // el modelo, cuando termina de cargar
let status = { key: "loading" };   // mensaje que se muestra junto al botón
let lastResults = null;            // últimos resultados, para redibujarlos al cambiar de idioma

// ---------- idioma ----------
// Idioma inicial: el último que eligió el visitante (guardado en su navegador) o, si no hay, el del navegador.
function detectLanguage() {
  try {
    const saved = localStorage.getItem("lang");
    if (saved === "es" || saved === "en") return saved;
  } catch { /* storage unavailable */ }
  return (navigator.language || "es").toLowerCase().startsWith("es") ? "es" : "en";
}

// Devuelve el texto de una clave en el idioma actual. Si el texto es una función, la llama con los argumentos.
function t(key, ...args) {
  const value = I18N[lang][key];
  return typeof value === "function" ? value(...args) : value;
}

// Pone todos los textos de la página en el idioma actual y vuelve a dibujar lo que depende de él.
function applyLanguage() {
  document.documentElement.lang = lang;
  document.title = t("pageTitle");
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
  renderChips();
  renderCategories();
  renderStatus();
  if (lastResults) renderResults(lastResults);
}

// Botones ES / EN: cambian el idioma y lo recuerdan para la próxima visita.
document.querySelectorAll("[data-lang]").forEach((b) => {
  b.addEventListener("click", () => {
    lang = b.dataset.lang;
    try { localStorage.setItem("lang", lang); } catch { /* ignore */ }
    applyLanguage();
  });
});

// ---------- secciones fijas ----------
// Botones de ejemplo bajo el cuadro de texto. Al pulsarlos rellenan el texto y clasifican.
function renderChips() {
  const chips = $("chips");
  chips.replaceChildren();
  for (const ex of EXAMPLES) {
    chips.append(makeChip(I18N[lang].labels[ex.label], ex.label, ex.text));
  }
  chips.append(makeChip(t("batchExample"), null, EXAMPLES.map((e) => e.text).join("\n")));
  chips.append(makeChip(t("poorExample"), null, POOR_EXAMPLE));
}

// Crea un botón de ejemplo. label es la categoría (para darle su color) o null si no tiene.
function makeChip(text, label, value) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip" + (label ? ` cat-${label}` : "");
  if (label) {
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = "var(--cat)";
    chip.append(dot);
  }
  chip.append(text);
  chip.addEventListener("click", () => {
    input.value = value;
    input.focus();
    if (classifier) classify();
  });
  return chip;
}

// Tarjetas que explican las cuatro categorías, al final de la página.
function renderCategories() {
  const grid = $("category-grid");
  grid.replaceChildren();
  for (const label of LABELS) {
    const card = document.createElement("div");
    card.className = `category cat-${label}`;
    const h3 = document.createElement("h3");
    h3.textContent = I18N[lang].labels[label];
    const p = document.createElement("p");
    p.textContent = I18N[lang].descriptions[label];
    const ex = document.createElement("p");
    ex.className = "example";
    ex.textContent = `“${EXAMPLES.find((e) => e.label === label).text}”`;
    card.append(h3, p, ex);
    grid.append(card);
  }
}

// ---------- mensaje de estado ----------
// Cambia el mensaje junto al botón (descargando, listo, error...). kind da el estilo: "ready" (punto verde) o "error" (rojo).
function setStatus(key, args = [], kind = "") {
  status = { key, args, kind };
  renderStatus();
}

function renderStatus() {
  statusText.textContent = t(status.key, ...(status.args || []));
  statusBox.className = `status ${status.kind || ""}`;
}

// ---------- carga del modelo ----------
// Descarga y prepara el modelo. La primera vez son unos 64 MB; después el navegador lo guarda en caché.
//   - dtype "q8": versión cuantizada a 8 bits (4 veces más pequeña que la original y casi igual de precisa).
//   - device "wasm": se ejecuta en la CPU con WebAssembly, que funciona en cualquier navegador moderno.
//   - progress_callback: avisa de lo descargado para mover la barra de progreso.
async function loadModel() {
  const files = new Map();
  setStatus("loading");
  progress.hidden = false;
  try {
    classifier = await pipeline("text-classification", MODEL_ID, {
      dtype: "q8",
      device: "wasm",
      progress_callback: (event) => {
        if (event.status === "progress" && event.total) {
          files.set(event.file, { loaded: event.loaded, total: event.total });
          let loaded = 0;
          let total = 0;
          for (const f of files.values()) { loaded += f.loaded; total += f.total; }
          const pct = Math.min(100, Math.round((loaded / total) * 100));
          progressBar.style.width = `${pct}%`;
          setStatus("loadingPct", [pct, (total / 1048576).toFixed(0)]);
        } else if (event.status === "done") {
          setStatus("preparing");
        }
      },
    });
    progress.hidden = true;
    setStatus("ready", [], "ready");
    button.disabled = false;
    if (input.value.trim()) classify();
  } catch (error) {
    console.error(error);
    progress.hidden = true;
    setStatus("loadError", [], "error");
  }
}

// ---------- clasificación ----------
// Clasifica una frase. top_k: null pide la probabilidad de las cuatro categorías, no solo la ganadora.
// Después revisa su redacción con quality.js.
async function predict(text) {
  let output = await classifier(text, { top_k: null });
  if (Array.isArray(output[0])) output = output[0];
  const scores = Object.fromEntries(output.map((o) => [o.label, o.score]));
  const label = LABELS.reduce((best, l) => ((scores[l] ?? 0) > (scores[best] ?? 0) ? l : best), LABELS[0]);
  return { text, label, scores, quality: analyzeRequirement(text, label, scores) };
}

// Se ejecuta al pulsar "Clasificar" (o Ctrl+Enter): separa el texto en líneas, clasifica cada una y dibuja el resultado.
async function classify() {
  if (!classifier || button.disabled) return;
  const lines = input.value.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) {
    setStatus("empty", [], "error");
    input.focus();
    return;
  }

  button.disabled = true;
  setStatus("classifying");
  try {
    const items = [];
    for (const line of lines.slice(0, MAX_LINES)) items.push(await predict(line));
    lastResults = items;
    renderResults(items);
    if (lines.length > MAX_LINES) setStatus("tooMany", [MAX_LINES]);
    else setStatus("ready", [], "ready");
  } catch (error) {
    console.error(error);
    setStatus("runError", [], "error");
  } finally {
    button.disabled = false;
  }
}

// En español se escribe "96 %" (con espacio); en inglés, "96%".
const pct = (x) => (lang === "es" ? `${Math.round(x * 100)} %` : `${Math.round(x * 100)}%`);

// ---------- dibujar resultados ----------
// Una sola frase: tarjeta detallada. Varias: lista compacta.
function renderResults(items) {
  results.replaceChildren(items.length === 1 ? renderSingle(items[0]) : renderBatch(items));
}

// Tarjeta de una frase: categoría, confianza, texto (con los términos problemáticos resaltados),
// barras de probabilidad y revisión de calidad. Se construye a partir del <template id="single-result"> del HTML.
function renderSingle(item) {
  const node = $("single-result").content.firstElementChild.cloneNode(true);
  node.classList.add(`cat-${item.label}`);
  const badge = node.querySelector(".badge");
  badge.textContent = I18N[lang].labels[item.label];

  const confidence = node.querySelector(".confidence");
  const strong = document.createElement("strong");
  strong.textContent = pct(item.scores[item.label]);
  confidence.append(strong, ` ${t("confidence")}`);

  node.querySelector(".result-text").append("“", highlight(item.text, item.quality), "”");
  node.querySelector(".result-desc").textContent = I18N[lang].descriptions[item.label];

  const bars = node.querySelector(".bars");
  const sorted = [...LABELS].sort((a, b) => (item.scores[b] ?? 0) - (item.scores[a] ?? 0));
  for (const label of sorted) {
    const score = item.scores[label] ?? 0;
    const row = document.createElement("div");
    row.className = `bar-row cat-${label}`;
    row.innerHTML = '<span class="bar-label"></span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-value"></span>';
    row.querySelector(".bar-label").textContent = I18N[lang].labels[label];
    row.querySelector(".bar-value").textContent = pct(score);
    const fill = row.querySelector(".bar-fill");
    // Se espera a que el navegador pinte la barra a 0 antes de darle su ancho, para que se vea crecer.
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${(score * 100).toFixed(1)}%`; }));
    bars.append(row);
  }
  node.append(renderQuality(item.quality));
  return node;
}

// Lista para varias frases. Cada fila lleva su categoría, la confianza, un indicador de calidad
// (tick o número de sugerencias) y, si hay problemas, un desplegable con las sugerencias.
function renderBatch(items) {
  const card = document.createElement("article");
  card.className = "card result";
  const title = document.createElement("p");
  title.className = "batch-title";
  const toReview = items.filter((i) => i.quality.level !== "ok").length;
  title.textContent = t("batchTitle", items.length) + (toReview ? ` · ${t("batchReview", toReview)}` : "");
  card.append(title);

  for (const item of items) {
    const row = document.createElement("div");
    row.className = "batch-row";
    const text = document.createElement("span");
    text.className = "batch-text";
    text.append(highlight(item.text, item.quality));
    const meta = document.createElement("span");
    meta.className = `batch-meta cat-${item.label}`;
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = I18N[lang].labels[item.label];
    const value = document.createElement("span");
    value.className = "pct";
    value.textContent = pct(item.scores[item.label]);
    meta.append(badge, value, qualityPill(item.quality, true));
    row.append(text, meta);
    if (item.quality.level !== "ok") {
      const details = document.createElement("details");
      details.className = "batch-quality";
      const summary = document.createElement("summary");
      summary.textContent = t("showSuggestions", item.quality.issues.length);
      details.append(summary, renderQuality(item.quality, { compact: true }));
      row.append(details);
    }
    card.append(row);
  }
  card.append(qualityNote());
  return card;
}

// ---------- calidad de la redacción ----------
// Resalta en el texto los términos que han provocado avisos.
function highlight(text, quality) {
  const marks = quality.issues
    .flatMap((issue) => issue.ranges.map(([start, end]) => ({ start, end, severity: issue.severity })))
    .sort((a, b) => a.start - b.start || (a.severity === "warning" ? -1 : 1));
  const fragment = document.createDocumentFragment();
  let pos = 0;
  for (const m of marks) {
    if (m.start < pos) continue; // solapa con una marca anterior
    fragment.append(text.slice(pos, m.start));
    const mark = document.createElement("mark");
    mark.className = `q-mark q-${m.severity}`;
    mark.textContent = text.slice(m.start, m.end);
    fragment.append(mark);
    pos = m.end;
  }
  fragment.append(text.slice(pos));
  return fragment;
}

// Etiqueta de color con el nivel de calidad. short = versión mínima (tick o número) para la lista.
function qualityPill(quality, short = false) {
  const pill = document.createElement("span");
  pill.className = `q-pill q-level-${quality.level}`;
  const levelText = I18N[lang].qualityLevels[quality.level];
  if (short) {
    pill.textContent = quality.level === "ok" ? "✓" : String(quality.issues.length);
    pill.title = quality.level === "ok" ? levelText : `${levelText}: ${t("qualityPill", quality.issues.length)}`;
    pill.setAttribute("aria-label", pill.title);
  } else {
    pill.textContent = levelText;
  }
  return pill;
}

// Aviso de que la revisión es automática y orientativa.
function qualityNote() {
  const note = document.createElement("p");
  note.className = "quality-note";
  note.textContent = t("qualityNote");
  return note;
}

// Sección "Calidad de la redacción": nivel, lista de problemas con su solución y estructura recomendada.
// compact = sin cabecera ni aviso (se usa dentro de los desplegables de la lista).
function renderQuality(quality, { compact = false } = {}) {
  const box = document.createElement("section");
  box.className = `quality q-level-${quality.level}`;

  if (!compact) {
    const head = document.createElement("div");
    head.className = "quality-head";
    const h3 = document.createElement("h3");
    h3.textContent = t("qualityTitle");
    head.append(h3, qualityPill(quality));
    box.append(head);
  }

  if (quality.level === "ok") {
    const ok = document.createElement("p");
    ok.className = "quality-ok";
    ok.textContent = t("qualityOk");
    box.append(ok);
  } else {
    const list = document.createElement("ul");
    list.className = "issues";
    for (const issue of quality.issues) {
      const { title, fix } = describeIssue(issue, lang, { labels: I18N[lang].labels, pct });
      const li = document.createElement("li");
      li.className = `issue q-${issue.severity}`;
      const strong = document.createElement("strong");
      strong.textContent = title;
      const p = document.createElement("p");
      p.textContent = fix;
      li.append(strong, p);
      list.append(li);
    }
    box.append(list);
    if (quality.template) box.append(renderTemplate(quality.template));
  }

  if (!compact) box.append(qualityNote());
  return box;
}

// Caja "Estructura recomendada". Los huecos <así> se pintan en otro color.
function renderTemplate(template) {
  const box = document.createElement("div");
  box.className = "template";
  const title = document.createElement("p");
  title.className = "template-title";
  title.textContent = t("templateTitle");
  const pattern = document.createElement("p");
  pattern.className = "template-pattern";
  for (const part of template.pattern.split(/(<[^>]+>)/)) {
    if (!part) continue;
    if (part.startsWith("<")) {
      const slot = document.createElement("span");
      slot.className = "slot";
      slot.textContent = part;
      pattern.append(slot);
    } else {
      pattern.append(part);
    }
  }
  const example = document.createElement("p");
  example.className = "template-example";
  example.textContent = `${t("templateExample")} ${template.example}`;
  box.append(title, pattern, example);
  return box;
}

// ---------- eventos ----------
button.addEventListener("click", classify);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    classify();
  }
});

// Al abrir la página: poner los textos en su idioma y empezar a descargar el modelo.
applyLanguage();
loadModel();
