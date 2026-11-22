// quality.js: revisa la redacción de un requisito  y propone mejoras.
//
// No usa IA: son reglas escritas a mano. Cada regla es una lista de palabras o
// expresiones que suelen indicar un requisito poco claro (siguiendo los "términos a
// evitar" de la norma ISO/IEC/IEEE 29148): términos subjetivos ("fast"), superlativos,
// comparaciones sin referencia, cláusulas de escape ("if possible"), enumeraciones
// abiertas ("etc."), pronombres ambiguos, etc.
// Además aprovecha la salida del clasificador: avisa si un requisito no funcional no
// tiene ninguna cifra y si el modelo duda entre dos categorías.
//
// Funciones que usa app.js:
//   - analyzeRequirement(): analiza una frase y devuelve el nivel, los problemas
//     encontrados (con su posición en el texto, para resaltarlos) y una plantilla.
//   - describeIssue(): devuelve el título y la solución de un problema en español o inglés.
//
// Para añadir una palabra a una regla basta con añadirla a su lista. Después conviene
// comprobar que no salta demasiado con los requisitos bien redactados de data/*.csv.

// "one" no cuenta como cifra cuando es pronombre ("the current one", "each one").
const NUMBER = /\d|(?<!\b(?:the|this|that|each|any|no|every|another|which|current|new|old|same)\s+)\bone\b|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|hundred|thousand|million|billion|half|once|twice|percent)\b/i;

// Lista de palabras/frases -> RegExp global con límites de palabra.
// Un espacio admite cualquier blanco y un guion es opcional ("user-friendly" = "user friendly").
function phrases(list) {
  const alts = [...list]
    .sort((a, b) => b.length - a.length)
    .map((p) => p.replace(/[.*+?^${}()|[\]\\\/]/g, "\\$&").replace(/ /g, "\\s+").replace(/-/g, "[\\s-]?"));
  return new RegExp(`\\b(?:${alts.join("|")})\\b`, "gi");
}

const VAGUE = [
  "fast", "quick", "quickly", "rapid", "rapidly", "promptly", "slow", "timely",
  "efficient", "efficiently", "user-friendly", "easy", "easily", "easy to use", "intuitive", "intuitively",
  "simple", "convenient", "flexible", "robust", "scalable", "reliable", "seamless", "seamlessly",
  "adequate", "adequately", "appropriate", "appropriately", "sufficient", "sufficiently",
  "reasonable", "reasonably", "acceptable", "proper", "properly", "suitable", "good", "nice",
  "modern", "state-of-the-art", "cutting-edge", "lightweight", "responsive",
  "minimal", "maximal", "optimal", "optimally", "optimized", "optimised",
  "significant", "significantly", "substantial", "large", "small", "huge",
  "highly", "high performance", "high availability", "high quality", "low latency", "low cost",
  "as soon as possible", "as fast as possible", "as much as possible",
];

const PASSIVE_IRREGULAR = "given|taken|written|chosen|hidden|broken|frozen|shown|seen|known|drawn|driven|forgotten|stolen|spoken|proven|sent|built|done|made|kept|held|set|put|run|sold|paid|told|found|read|lost|brought|bought|caught|taught|left|meant|shut|split|spent|understood|withdrawn";
// Solo la pasiva tras un verbo de obligación ("shall be stored"): en oraciones
// subordinadas ("once the order is paid") es natural y no esconde a ningún responsable.
const PASSIVE = new RegExp(`\\b(?:shall|must|will|should|can|may|has to|have to|needs to)\\s+(?:not\\s+)?be\\s+(?:[a-z]+ly\\s+)?([a-z]{3,}ed|${PASSIVE_IRREGULAR})\\b(?!-)`, "gi");
// Participios que suelen describir un estado o una norma, no una acción sin responsable.
const PASSIVE_OK = new Set([
  "required", "allowed", "permitted", "prohibited", "forbidden", "based", "expected", "limited", "needed",
  "entitled", "located", "intended", "supposed", "considered", "designed", "interested", "concerned",
  "involved", "related", "dedicated", "logged", "signed", "registered", "enrolled", "subscribed",
  "qualified", "connected", "called", "named", "composed", "authorized", "authorised",
]);

// Reglas léxicas, en orden: una coincidencia que solapa con otra ya encontrada
// por una regla anterior se ignora (así "as appropriate" no cuenta también como "appropriate").
// Cada regla tiene: id (para buscar su mensaje), severity ("warning" = conviene cambiarlo,
// "info" = sugerencia menor), re (expresiones a buscar) y, opcionalmente, when (solo se
// aplica si se cumple) y filter (descarta coincidencias concretas).
const RULES = [
  {
    id: "incomplete",
    severity: "warning",
    re: [phrases(["TBD", "TBC", "TBA", "to be determined", "to be defined", "to be decided", "to be confirmed", "TODO"])],
  },
  {
    id: "openEnded",
    severity: "warning",
    re: [
      phrases(["etc", "and so on", "and so forth", "among others", "but not limited to", "including but not limited to", "and/or", "or similar", "or equivalent"]),
      /\.\.\.|…/g,
    ],
  },
  {
    id: "loophole",
    severity: "warning",
    re: [phrases([
      "if possible", "where possible", "when possible", "whenever possible", "as far as possible", "to the extent possible",
      "as appropriate", "where appropriate", "if appropriate", "as applicable", "where applicable", "if applicable",
      "if necessary", "when necessary", "where necessary", "as necessary", "as needed", "if needed", "when needed",
      "if required", "if feasible", "where feasible", "ideally", "preferably", "optionally",
    ])],
  },
  {
    id: "superlative",
    severity: "warning",
    re: [phrases(["best", "worst", "fastest", "quickest", "easiest", "simplest", "greatest", "the most", "the least"])],
  },
  {
    id: "comparative",
    severity: "warning",
    when: (c) => !c.hasNumber,
    re: [
      phrases(["better", "worse", "faster", "slower", "quicker", "easier", "simpler", "cheaper", "higher", "lower", "improved", "enhanced"]),
      /\b(?:more|less)\s+(?!than\b|or\b|and\b|of\b)[a-z]+/gi,
    ],
  },
  {
    id: "vague",
    // Si el requisito ya incluye una cifra, el término vago probablemente está respaldado.
    severity: (c) => (c.hasNumber ? "info" : "warning"),
    re: [phrases(VAGUE)],
  },
  {
    id: "vagueQuantity",
    severity: "warning",
    re: [phrases([
      "several", "many", "few", "a few", "some", "various", "numerous", "a lot of", "lots of", "most",
      "usually", "often", "frequently", "occasionally", "sometimes", "generally", "typically", "normally",
      "regularly", "periodically", "rarely", "almost", "nearly",
    ])],
  },
  {
    id: "totality",
    severity: "info",
    re: [phrases(["always", "never", "at all times", "every time", "under all circumstances", "under any circumstances", "without exception"])],
  },
  {
    id: "weakModal",
    severity: "info",
    re: [phrases(["should", "may", "might", "could", "would"])],
  },
  {
    id: "pronoun",
    severity: "info",
    // Solo al inicio: dentro de la frase, "it"/"they" casi siempre se refieren a algo de la propia frase.
    re: [/^\s*(?:it|they|this|these|those)(?=\s+(?:shall|must|will|should|can|may|is|are|has|have|needs?)\b)/gi],
  },
  {
    id: "passive",
    severity: "info",
    // En las restricciones la pasiva es natural ("must be written in Java").
    when: (c) => c.label !== "CONSTRAINT" && !/\bby\b/i.test(c.text),
    re: [PASSIVE],
    filter: (m) => !PASSIVE_OK.has(m[1].toLowerCase()),
  },
];

const MODALS = /\b(?:shall|must|will|should|has to|have to|needs to|need to)\b/gi;
const MEASURABLE = /\b(?:fast|quick|speed|respond|responds|response\s+times?|responsive|latency|load(?:s|ing)?\s+time|performance|throughput|slow|uptime|downtime|availability|outage|scal\w*|concurrent|simultaneous|easy|easily|intuitive|user[\s-]?friendly|usability)\b/i;
const OBLIGATION = /\b(?:shall|must|will|should|can|cannot|may|might|could|would|has to|have to|needs? to|requires?|required|allowed|permitted|prohibited|forbidden)\b/i;

// Plantillas de "Estructura recomendada" para cada categoría. Los no funcionales tienen
// una por tema (rendimiento, disponibilidad...) porque cada uno se mide de forma distinta.
const TEMPLATES = {
  FUNCTIONAL: {
    pattern: "When <trigger>, the <system> shall <action> <object>.",
    example: "When an order is shipped, the system shall email the tracking number to the customer.",
  },
  BUSINESS_RULE: {
    pattern: "If <condition>, then <actor or object> shall <consequence>.",
    example: "If an order exceeds 500 EUR, then it shall require approval from a sales manager.",
  },
  CONSTRAINT: {
    pattern: "The <system or component> shall <use / comply with> <specific technology, version or standard>.",
    example: "The backend shall store its data in PostgreSQL 15 or later.",
  },
  NFR_performance: {
    pattern: "The <system> shall <operation> in under <time> for <percentile> of requests with <load>.",
    example: "The search page shall return results in under 1 s for 95% of queries with 500 concurrent users.",
  },
  NFR_availability: {
    pattern: "The <system> shall be available <percentage> of the time per <period>, excluding <planned maintenance>.",
    example: "The checkout service shall be available 99.9% of the time per month, excluding planned maintenance windows.",
  },
  NFR_usability: {
    pattern: "A <type of user> shall be able to <task> in under <time> <without help / after N minutes of training>.",
    example: "A first-time user shall be able to place an order in under 3 minutes without help.",
  },
  NFR_scalability: {
    pattern: "The <system> shall support <N> <concurrent users / transactions per second> while <performance criterion>.",
    example: "The platform shall support 10,000 concurrent users while keeping response times under 2 s.",
  },
  NFR_security: {
    pattern: "The <system> shall <protect what> using <specific mechanism or standard>.",
    example: "The system shall store passwords hashed with bcrypt using a cost factor of at least 12.",
  },
  NFR_generic: {
    pattern: "The <system> shall <quality attribute> <measurable value> under <conditions>.",
    example: "The mobile app shall start in under 2 s on a mid-range device.",
  },
};

const NFR_TOPICS = [
  ["availability", /availab|uptime|downtime|outage|failover|recover|reliab/i],
  ["security", /secur|encrypt|protect|attack|password|authent|privacy|confidential/i],
  ["usability", /easy|easily|intuitive|user[\s-]?friendly|usab|learn|simple|accessib|convenient/i],
  ["scalability", /scal|concurrent|growth|volume|simultaneous/i],
  ["performance", /fast|quick|speed|respon|latency|load|performance|throughput|slow|time/i],
];

function pickTemplate(text, label) {
  if (label !== "NON_FUNCTIONAL") return TEMPLATES[label] ?? null;
  const topic = NFR_TOPICS.find(([, re]) => re.test(text))?.[0] ?? "generic";
  return TEMPLATES[`NFR_${topic}`];
}

// Busca en el texto las coincidencias de una regla y devuelve sus posiciones [inicio, fin].
// claimed son las posiciones ya encontradas por reglas anteriores, que se saltan.
function collect(rule, text, claimed) {
  const ranges = [];
  for (const re of rule.re) {
    for (const m of text.matchAll(re)) {
      if (rule.filter && !rule.filter(m)) continue;
      const start = m.index;
      const end = start + m[0].length;
      if (claimed.some(([s, e]) => start < e && s < end)) continue;
      ranges.push([start, end]);
    }
  }
  return ranges;
}

// Términos únicos (sin distinguir mayúsculas), con la grafía de su primera aparición.
function termsOf(text, ranges) {
  const terms = new Map();
  for (const [s, e] of ranges) {
    const term = text.slice(s, e).trim().replace(/\s+/g, " ");
    if (!terms.has(term.toLowerCase())) terms.set(term.toLowerCase(), term);
  }
  return [...terms.values()];
}

/**
 * Analiza la redacción de un requisito.
 * @param {string} text  requisito (en inglés)
 * @param {string} label clase predicha por el modelo
 * @param {Record<string, number>} scores probabilidades por clase
 * @returns {{level: "ok"|"minor"|"review", issues: Array, template: {pattern: string, example: string}|null}}
 */
export function analyzeRequirement(text, label, scores) {
  const ctx = { text, label, hasNumber: NUMBER.test(text) };
  const issues = [];
  const claimed = [];

  for (const rule of RULES) {
    if (rule.when && !rule.when(ctx)) continue;
    const ranges = collect(rule, text, claimed);
    if (ranges.length === 0) continue;
    claimed.push(...ranges);
    const severity = typeof rule.severity === "function" ? rule.severity(ctx) : rule.severity;
    issues.push({ id: rule.id, severity, ranges, terms: termsOf(text, ranges), data: {} });
  }

  const modals = [...text.matchAll(MODALS)];
  if (modals.length >= 2) {
    const ranges = modals.map((m) => [m.index, m.index + m[0].length]);
    issues.push({ id: "compound", severity: "info", ranges, terms: termsOf(text, ranges), data: {} });
  }

  const words = text.split(/\s+/).filter(Boolean).length;
  if (words > 35) issues.push({ id: "long", severity: "info", ranges: [], terms: [], data: { words } });

  // En reglas de negocio y restricciones el presente es habitual ("Refunds are issued to...").
  if ((label === "FUNCTIONAL" || label === "NON_FUNCTIONAL") && !OBLIGATION.test(text)) {
    issues.push({ id: "noModal", severity: "info", ranges: [], terms: [], data: {} });
  }

  // Solo se exige una cifra en atributos que se miden (no en seguridad, mantenibilidad...) o si hay términos vagos.
  const vagueFound = issues.some((i) => ["vague", "comparative", "superlative"].includes(i.id));
  if (label === "NON_FUNCTIONAL" && !ctx.hasNumber && (vagueFound || MEASURABLE.test(text))) {
    issues.push({ id: "nfrMetric", severity: "warning", ranges: [], terms: [], data: {} });
  }

  const [first, second] = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  if (second && second[1] >= 0.25) {
    issues.push({
      id: "modelDoubt",
      severity: "info",
      ranges: [],
      terms: [],
      data: { a: first[0], pa: first[1], b: second[0], pb: second[1] },
    });
  }

  issues.sort((a, b) => (a.severity === b.severity ? 0 : a.severity === "warning" ? -1 : 1));
  const level = issues.length === 0 ? "ok" : issues.some((i) => i.severity === "warning") ? "review" : "minor";
  return { level, issues, template: issues.length ? pickTemplate(text, label) : null };
}

// ---------- textos ----------
// Título y solución de cada problema, en los dos idiomas de la interfaz.
const quote = (terms) => terms.map((t) => `“${t}”`).join(", ");

const MESSAGES = {
  es: {
    incomplete: {
      title: (i) => `Requisito incompleto: ${quote(i.terms)}`,
      fix: "Contiene marcadores pendientes. Sustitúyelos por el valor o la decisión concreta antes de darlo por cerrado.",
    },
    openEnded: {
      title: (i) => `Enumeración abierta: ${quote(i.terms)}`,
      fix: "Lo que no aparece en la lista no se puede verificar. Enumera todos los casos o remite a una lista o documento concreto.",
    },
    loophole: {
      title: (i) => `Cláusula de escape: ${quote(i.terms)}`,
      fix: "Expresiones como «if possible» permiten no cumplir el requisito. Indica la condición exacta en la que aplica o elimínala.",
    },
    superlative: {
      title: (i) => `Superlativo: ${quote(i.terms)}`,
      fix: "«El mejor» o «el más rápido» no se pueden comprobar. Sustitúyelo por un valor objetivo.",
    },
    comparative: {
      title: (i) => `Comparación sin referencia: ${quote(i.terms)}`,
      fix: "¿Mejor o más rápido que qué? Indica el valor de referencia y la mejora esperada con cifras.",
    },
    vague: {
      title: (i) => `Términos vagos: ${quote(i.terms)}`,
      fix: (i) => (i.severity === "info"
        ? "El requisito ya incluye una cifra: comprueba que concreta estos términos o elimínalos para evitar dudas."
        : "Son subjetivos y no se pueden probar. Sustitúyelos por un criterio medible (tiempo, porcentaje, número de pasos…)."),
    },
    vagueQuantity: {
      title: (i) => `Cantidad o frecuencia imprecisa: ${quote(i.terms)}`,
      fix: "Indica una cantidad o una frecuencia exacta (p. ej. «at most 5», «every 24 hours»).",
    },
    totality: {
      title: (i) => `Términos absolutos: ${quote(i.terms)}`,
      fix: "Comprueba que es realmente absoluto y que se puede probar. Si hay excepciones, enuméralas.",
    },
    weakModal: {
      title: (i) => `Verbo poco preciso: ${quote(i.terms)}`,
      fix: "«should» o «could» expresan un deseo, no una obligación, y «may» puede ser permiso o posibilidad. Usa «shall»/«must» si es obligatorio o «is allowed to» si es un permiso.",
    },
    pronoun: {
      title: (i) => `Pronombre ambiguo: ${quote(i.terms)}`,
      fix: "Un requisito debe entenderse por sí solo. Sustituye el pronombre por el sistema, actor u objeto al que se refiere.",
    },
    passive: {
      title: (i) => `Voz pasiva sin responsable: ${quote(i.terms)}`,
      fix: "No queda claro quién realiza la acción. Indica el actor: «the system shall…», «the administrator shall…».",
    },
    compound: {
      title: () => "Varios requisitos en uno",
      fix: (i) => `Hay varios verbos de obligación (${quote(i.terms)}). Divide la frase en requisitos separados para poder verificar y rastrear cada uno.`,
    },
    long: {
      title: (i) => `Frase muy larga (${i.data.words} palabras)`,
      fix: "Las frases largas suelen mezclar varias ideas. Intenta quedarte con una sola condición y una sola acción.",
    },
    noModal: {
      title: () => "Sin verbo de obligación",
      fix: "No se indica si es obligatorio. Usa «shall» o «must» (p. ej. «The system shall…»).",
    },
    nfrMetric: {
      title: () => "Requisito no funcional sin métrica",
      fix: "Un requisito no funcional necesita un valor medible para poder verificarse. Mira la estructura recomendada.",
    },
    modelDoubt: {
      title: (i, f) => `El modelo duda entre ${f.labels[i.data.a]} (${f.pct(i.data.pa)}) y ${f.labels[i.data.b]} (${f.pct(i.data.pb)})`,
      fix: "Puede que el requisito mezcle varios tipos o que no esté claro qué pretende. Prueba a concretarlo o a dividirlo.",
    },
  },
  en: {
    incomplete: {
      title: (i) => `Incomplete requirement: ${quote(i.terms)}`,
      fix: "It contains placeholders. Replace them with the actual value or decision before considering it done.",
    },
    openEnded: {
      title: (i) => `Open-ended list: ${quote(i.terms)}`,
      fix: "Anything not listed cannot be verified. List every case or refer to a specific list or document.",
    },
    loophole: {
      title: (i) => `Escape clause: ${quote(i.terms)}`,
      fix: "Phrases like “if possible” make the requirement optional. State the exact condition under which it applies, or remove it.",
    },
    superlative: {
      title: (i) => `Superlative: ${quote(i.terms)}`,
      fix: "“The best” or “the fastest” cannot be tested. Replace it with an objective value.",
    },
    comparative: {
      title: (i) => `Comparison without a baseline: ${quote(i.terms)}`,
      fix: "Better or faster than what? State the baseline and the expected improvement in numbers.",
    },
    vague: {
      title: (i) => `Vague terms: ${quote(i.terms)}`,
      fix: (i) => (i.severity === "info"
        ? "The requirement already includes a number: check that it pins these terms down, or remove them to avoid doubt."
        : "They are subjective and cannot be tested. Replace them with a measurable criterion (time, percentage, number of steps…)."),
    },
    vagueQuantity: {
      title: (i) => `Imprecise quantity or frequency: ${quote(i.terms)}`,
      fix: "State an exact quantity or frequency (e.g. “at most 5”, “every 24 hours”).",
    },
    totality: {
      title: (i) => `Absolute terms: ${quote(i.terms)}`,
      fix: "Check that it really is absolute and testable. If there are exceptions, list them.",
    },
    weakModal: {
      title: (i) => `Imprecise verb: ${quote(i.terms)}`,
      fix: "“should” or “could” express a wish, not an obligation, and “may” can mean permission or possibility. Use “shall”/“must” if it is mandatory, or “is allowed to” if it is a permission.",
    },
    pronoun: {
      title: (i) => `Ambiguous pronoun: ${quote(i.terms)}`,
      fix: "A requirement should stand on its own. Replace the pronoun with the system, actor or object it refers to.",
    },
    passive: {
      title: (i) => `Passive voice with no actor: ${quote(i.terms)}`,
      fix: "It is not clear who performs the action. Name the actor: “the system shall…”, “the administrator shall…”.",
    },
    compound: {
      title: () => "Several requirements in one",
      fix: (i) => `There are several obligation verbs (${quote(i.terms)}). Split the sentence into separate requirements so each one can be verified and traced.`,
    },
    long: {
      title: (i) => `Very long sentence (${i.data.words} words)`,
      fix: "Long sentences tend to mix several ideas. Try to keep to a single condition and a single action.",
    },
    noModal: {
      title: () => "No obligation verb",
      fix: "It does not say whether it is mandatory. Use “shall” or “must” (e.g. “The system shall…”).",
    },
    nfrMetric: {
      title: () => "Non-functional requirement without a metric",
      fix: "A non-functional requirement needs a measurable value to be verifiable. See the recommended structure.",
    },
    modelDoubt: {
      title: (i, f) => `The model hesitates between ${f.labels[i.data.a]} (${f.pct(i.data.pa)}) and ${f.labels[i.data.b]} (${f.pct(i.data.pb)})`,
      fix: "The requirement may mix several kinds or its intent may be unclear. Try making it more specific or splitting it.",
    },
  },
};

/**
 * Texto de un problema en el idioma de la interfaz.
 * @param {object} issue elemento de analyzeRequirement().issues
 * @param {"es"|"en"} lang
 * @param {{labels: Record<string, string>, pct: (x: number) => string}} format
 */
export function describeIssue(issue, lang, format) {
  const msg = MESSAGES[lang][issue.id];
  const fix = typeof msg.fix === "function" ? msg.fix(issue, format) : msg.fix;
  return { title: msg.title(issue, format), fix };
}
