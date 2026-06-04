# Prueba Técnica — Asistente de Soporte RAG (MineCatalog)

Un asistente de soporte que responde preguntas  usando únicamente la
documentación técnica brindada. Las respuestas están en el corpus: si la documentación no contiene la respuesta, el asistente lo indica
explícitamente en lugar de inventarla.

Alcance de este repositorio: la *pipeline de ingesta* + la *API REST de recuperación*. La
generación con un LLM y la orquestación viven en n8n (proyecto aparte). La API no llama a ningún
LLM: devuelve el contexto recuperado y un *prompt completo en español* que el nodo LLM de n8n usa
para generar la respuesta final. El workflow de n8n se adjunta por separado en la raíz del repo como
`Prueba-Tecnica-n8n-Workflow.json`.

Flujo completo: **n8n (webhook) → API `/retrieve` → n8n (nodo LLM) → respuesta en español**.

---

## Estructura del proyecto

```
Prueba-Tecnica-Cesar-Nunez/
├── app/
│   ├── data_ingest.py      # Carga + limpieza/normalización + chunking del corpus
│   ├── embedding_docs.py   # EmbeddingManager (modelo local multilingual-e5-small)
│   ├── vector_store.py     # VectorStore (ChromaDB persistente, distancia coseno)
│   ├── rag_retrieval.py    # RAGRetriever (umbral de relevancia + prompt de anclaje en español)
│   ├── api.py              # API FastAPI (GET /health, POST /retrieve)
│   └── main.py             # Punto de entrada de ingesta (load → clean → chunk → embed → store)
├── docs/                   # Tu corpus aquí (.txt .md .pdf .json) — ignorado por git
├── specs/                  # Enunciado de la prueba
├── .env.example            # Plantilla de configuración
├── requirements.txt        # Dependencias
├── Prueba-Tecnica-n8n-Workflow.json   # Workflow de n8n (se adjunta por separado)
└── README.md
```

---

## Requisitos

- **Python 3.14**
- **[uv](https://docs.astral.sh/uv/)** (recomendado) o `pip`/`venv` estándar
- **n8n** (vía `npx` o Docker) — para el workflow
- Un **LLM** para el nodo de n8n: una **API key de OpenAI** *o* **[Ollama](https://ollama.com/)** local

---

## Instalación

> Ejecuta todos los comandos **desde la raíz del repositorio**.

### 1. Crear y activar el entorno virtual

**bash (Linux/macOS):**
```bash
uv venv                       # o: python -m venv .venv
source .venv/bin/activate
```

**cmd (Windows):**
```cmd
uv venv
.venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
uv add -r requirements.txt
```
Con pip estándar: `pip install -r requirements.txt`.

### 3. Configurar las variables de entorno

Copia la plantilla a `.env`:

**bash:**
```bash
cp .env.example .env
```

**cmd:**
```cmd
copy .env.example .env
```

Edita `.env` con tu editor (`nano .env`, `vim .env`, o el editor de Windows) si quieres cambiar
rutas o parámetros — los valores por defecto funcionan tal cual. **No se necesita ninguna API key
aquí:** la generación (y por tanto la key de OpenAI) se configura en n8n. Ver
[Referencia de configuración](#referencia-de-configuración).

> La primera ejecución descargará el modelo de embeddings `multilingual-e5-small` (una sola vez).

---

## Uso

### Paso 1 — Añadir documentos

Coloca tus archivos de documentación en `docs/` (la carpeta se ignora en git, así que empieza
vacía). Formatos soportados: **`.txt`, `.md`, `.pdf`, `.json`**. Puedes usar el corpus de ejemplo
descrito en `specs/`.

### Paso 2 — Ejecutar la ingesta

```bash
python app/main.py
# Ingesting corpus...
# Done. Vector store holds N chunks.
```
Es **idempotente** (usa `upsert` con IDs por contenido), así que puedes re-ejecutarlo tras cambiar
el corpus sin duplicar. El índice se guarda en `docs/chroma-db/`.

### Paso 3 — Iniciar la API

```bash
fastapi dev app/api.py        # servidor de desarrollo con recarga + Swagger en /docs
# alternativa: python app/api.py   (uvicorn en 127.0.0.1:8000)
```

Comprobaciones rápidas:
```bash
# Estado del servicio
curl localhost:8000/health
# {"status":"ok"}

# Pregunta relevante -> sufficient_context: true + contexto
curl -X POST localhost:8000/retrieve -H "Content-Type: application/json" \
  -d '{"question":"¿Cómo soluciono el error de conexión con la base de datos?"}'

# Pregunta vacía -> 422 (validación)
curl -X POST localhost:8000/retrieve -H "Content-Type: application/json" \
  -d '{"question":"   "}'
```

`POST /retrieve` devuelve:
```json
{
  "question": "...",
  "sufficient_context": true,
  "contexts": [{"text": "...", "source": "Documentación 4.json", "similarity": 0.88}],
  "sources": ["Documentación 4.json"],
  "grounding_prompt": "Eres un asistente de soporte técnico... CONTEXTO: ... PREGUNTA: ..."
}
```
Cuando `sufficient_context` es `false`, el `grounding_prompt` ya instruye al LLM para que indique
que no dispone de esa información.

### Paso 4 — Workflow de n8n

#### 4.1 Ejecutar n8n

**Host (npx o app de escritorio):** abre `http://localhost:5678`
```bash
npx n8n
```

**Docker:**
```bash
docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n
```

#### 4.2 Importar el workflow

En n8n: **Workflows → Import from File →** selecciona `Prueba-Tecnica-n8n-Workflow.json`
(en la raíz del repo).

#### 4.3 Conectar el nodo HTTP Request a la API

El workflow llama a `POST /retrieve`. Ajusta la URL según cómo corras n8n:

| n8n corre en… | URL de la API |
|---------------|---------------|
| Host (npx / escritorio) | `http://localhost:8000/retrieve` |
| Docker | `http://host.docker.internal:8000/retrieve` |

> En Docker, `localhost` apunta al contenedor, no a tu máquina: por eso se usa
> `host.docker.internal`.

#### 4.4 Conectar el nodo Basic LLM Chain

El nodo usa el `grounding_prompt` devuelto por la API como prompt de entrada.

**Opción A — OpenAI (recomendada):**
1. Crea una credencial *OpenAI* en n8n con tu API key (de `https://platform.openai.com`).
2. Selecciona un modelo, p. ej. `gpt-4o-mini`.

**Opción B — Ollama (local, sin coste):**
1. Instala Ollama y arráncalo:
   ```bash
   ollama serve            # escucha en 127.0.0.1:11434
   ollama pull llama3.1    # o el modelo que prefieras
   ```
2. En n8n configura la credencial/URL base de Ollama:

   | n8n corre en… | URL base de Ollama |
   |---------------|--------------------|
   | Host | `http://localhost:11434` |
   | Docker | `http://host.docker.internal:11434` |
3. Selecciona el modelo descargado.

#### 4.5 Probar

Envía una pregunta al webhook del workflow (con curl o Postman):
```bash
curl -X POST <URL-del-webhook-n8n> -H "Content-Type: application/json" \
  -d '{"question":"El sistema devuelve error de conexión, ¿qué significa?"}'
```

---

## Referencia de configuración

Variables en `.env` (valores por defecto entre paréntesis):

| Variable | Descripción | Por defecto |
|----------|-------------|-------------|
| `DOCS_DIR` | Carpeta del corpus | `docs` |
| `CHROMA_DIR` | Carpeta de persistencia de ChromaDB | `docs/chroma-db` |
| `CHROMA_COLLECTION` | Nombre de la colección | `ChromaDB_collection_for_corpus` |
| `EMBEDDING_MODEL` | Modelo local de embeddings | `intfloat/multilingual-e5-small` |
| `TOP_K` | Nº de fragmentos candidatos a recuperar | `2` |
| `SIMILARITY_THRESHOLD` | Similitud coseno mínima (1 − distancia) para considerar un fragmento relevante | `0.8` |
| `CHUNK_SIZE` | Tamaño máximo de fragmento (caracteres) | `800` |
| `CHUNK_OVERLAP` | Solapamiento entre fragmentos | `200` |

> **Sobre `SIMILARITY_THRESHOLD`:** e5 produce similitudes en una banda alta y estrecha (lo relevante
> ronda ≥0.85 y lo no relacionado ~0.75), por eso el umbral está en `0.8`. Súbelo/bájalo según los
> valores `similarity` que veas en las respuestas. `TOP_K` controla cuántos candidatos se evalúan
> antes de aplicar el umbral.

---

## Cómo funciona

1. **Ingesta** (`app/main.py`): `cargar → limpiar/normalizar (NFKC) → trocear (chunking) → embeddings → almacenar` en ChromaDB.
2. **Recuperación** (`POST /retrieve`): embebe la pregunta (prefijo `query:` de e5), busca por coseno,
   aplica el **umbral de relevancia** y arma un **prompt de anclaje en español**.
3. **Generación** (n8n, fuera de este repo): el nodo LLM usa ese prompt para responder. Si no hubo
   contexto suficiente (`sufficient_context: false`), el prompt obliga al modelo a decir que la
   información no está en la documentación → **sin alucinaciones, siempre en español**.

---

## Solución de problemas

- **`ModuleNotFoundError` / no encuentra `docs` o `.env`:** ejecuta los comandos **desde la raíz del
  repo** (los módulos usan rutas relativas).
- **`/retrieve` devuelve `sufficient_context: false` siempre:** primero ejecuta `python app/main.py`
  para poblar el índice; si aun así no recupera, baja `SIMILARITY_THRESHOLD`.
- **n8n no alcanza la API (Docker):** usa `http://host.docker.internal:8000`, no `localhost`.
- **La primera ejecución tarda:** está descargando el modelo de embeddings (una sola vez).
