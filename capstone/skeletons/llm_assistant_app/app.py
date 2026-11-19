"""Assistant web app: LLM chat with optional retrieval over your own documents (RAG) + Gradio.

    uv run python app.py                                  # plain chat
    uv run python app.py --data faq.csv --column text     # answers grounded on the documents of the CSV
    uv run python app.py --provider anthropic --share     # force a provider, public link

The LLM key is read from a `.env` file (current folder or a parent folder) or from the environment.
The course provides OPENAI_API_KEY, ANTHROPIC_API_KEY and MISTRAL_API_KEY for a limited time. For a project kept online after the
course, create your own key and store it as a secret of the hosting platform (Hugging Face Space secret): never in the repository.
Any OpenAI-compatible service works with LLM_BASE_URL, LLM_API_KEY and LLM_MODEL.
"""
import argparse, os
import numpy as np
import pandas as pd
import gradio as gr

# --- BEGIN llm_helper (identical to the cell of Lab 5) ---
import os, re, time, random
import requests

# ---------------------------------------------------------------------------
# Access to an LLM through the chat-completions protocol shared by OpenAI, Anthropic and Mistral.
# Keys are read from a `.env` file (current folder or a parent folder), from the environment,
# or from Colab secrets. They are never written in the notebook.
# ---------------------------------------------------------------------------
LLM_PROVIDERS = {
    "openai": {"url": "https://api.openai.com/v1/chat/completions", "model": "gpt-4o-mini",
               "tokens_param": "max_completion_tokens", "temperature": True,
               "key_names": ("OPENAI_API_KEY", "OPEN_AI_API_KEY")},
    "anthropic": {"url": "https://api.anthropic.com/v1/chat/completions", "model": "claude-haiku-5-5",
                  "tokens_param": "max_tokens", "temperature": False,     # recent Claude models reject `temperature`
                  "extra": {"thinking": {"type": "disabled"}},            # otherwise hidden reasoning uses up max_tokens
                  "key_names": ("ANTHROPIC_API_KEY",)},
    "mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "model": "ministral-8b-latest",
                "tokens_param": "max_tokens", "temperature": True,        # mistral-small/medium are not available on free keys
                "key_names": ("MISTRAL_API_KEY", "MIXTRAL_API_KEY")},
}


def load_env(filename=".env"):
    """Load the KEY=VALUE lines of a .env file found in the current folder or in a parent folder. Returns its path or None."""
    folder = os.path.abspath(os.getcwd())
    for _ in range(5):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            with open(path, encoding="utf-8-sig") as f:          # utf-8-sig: files saved by Windows Notepad start with a BOM
                for line in f:
                    m = re.match(r"\s*(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
                    if m and not line.lstrip().startswith("#"):
                        os.environ.setdefault(m.group(1), m.group(2).strip("\"'"))
            return path
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return None


def _find_key(provider):
    """Key of a provider: exact variable name, then any variable starting with it (OPENAI_API_KEY_COURSE...), then Colab secrets."""
    names = LLM_PROVIDERS[provider]["key_names"]
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    for var, value in os.environ.items():
        if value and any(var.startswith(name) for name in names):
            return value
    try:
        from google.colab import userdata
        for name in names:
            try:
                return userdata.get(name)
            except Exception:
                pass
    except ImportError:
        pass
    return ""


def setup_llm(ask_if_missing=True):
    """Choose the provider and model. Returns {"provider", "model", "key", "url", "env_file"}.

    Order: LLM_PROVIDER (if set), then openai, anthropic, mistral. LLM_MODEL overrides the default model.
    Any OpenAI-compatible service can be used with LLM_BASE_URL (ending with /chat/completions), LLM_API_KEY and LLM_MODEL.
    """
    env_file = load_env()
    cfg = {"provider": None, "model": None, "key": "", "url": None, "env_file": env_file}
    if os.environ.get("LLM_DISABLED"):                      # used by the grading scripts: never call a paid API
        print("LLM calls are disabled (LLM_DISABLED is set).")
        return cfg
    if os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY"):
        LLM_PROVIDERS["custom"] = {"url": os.environ["LLM_BASE_URL"], "model": os.environ.get("LLM_MODEL", ""),
                                   "tokens_param": "max_tokens", "temperature": True, "key_names": ("LLM_API_KEY",)}
        wanted = "custom"
    else:
        wanted = os.environ.get("LLM_PROVIDER", "").strip().lower()
    order = ([wanted] if wanted in LLM_PROVIDERS else []) + [p for p in LLM_PROVIDERS if p != wanted]
    for name in order:
        key = _find_key(name)
        if key:
            cfg.update(provider=name, key=key)
            break
    if not cfg["key"] and ask_if_missing:
        try:
            from getpass import getpass
            key = getpass("No key found. Paste an API key (OpenAI, Anthropic or Mistral), or leave empty to skip: ").strip()
        except Exception:
            key = ""
        if key:
            cfg.update(provider="anthropic" if key.startswith("sk-ant-") else "openai" if key.startswith("sk-") else "mistral", key=key)
    if cfg["provider"]:
        spec = LLM_PROVIDERS[cfg["provider"]]
        cfg.update(url=spec["url"], model=os.environ.get("LLM_MODEL") or spec["model"])
        print(f"LLM ready: {cfg['provider']} / {cfg['model']}" + (f" (keys read from {env_file})" if env_file else ""))
    else:
        print("No API key: the LLM cells will be skipped.")
    return cfg


def query_llm(prompt=None, max_tokens=200, temperature=0.0, messages=None, retries=6):
    """Send one chat request to the selected LLM and return the answer text.

    `prompt` is a single user message; `messages` (list of {"role", "content"}) can be given instead for a conversation.
    Rate limits (HTTP 429) and temporary server errors are retried with a growing delay.
    """
    if not LLM["key"]:
        return "(no API key)"
    spec = LLM_PROVIDERS[LLM["provider"]]
    payload = {"model": LLM["model"], "messages": messages or [{"role": "user", "content": prompt}], spec["tokens_param"]: max_tokens}
    payload.update(spec.get("extra", {}))
    if spec["temperature"] and temperature is not None:
        payload["temperature"] = temperature
    for attempt in range(retries):
        try:
            r = requests.post(LLM["url"], headers={"Authorization": f"Bearer {LLM['key']}"}, json=payload, timeout=60)
        except requests.RequestException as e:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        if r.status_code == 429 and r.headers.get("x-ratelimit-limit-req-minute") == "0":
            return (f"Error 429: the model '{LLM['model']}' is not available with this key (request limit of 0 per minute). "
                    "Choose another model or provider (LLM_MODEL / LLM_PROVIDER).")      # permanent, retrying is useless
        if r.status_code in (429, 500, 502, 503, 529):
            wait = float(r.headers["retry-after"]) if r.headers.get("retry-after", "").replace(".", "", 1).isdigit() else 2 ** attempt
            time.sleep(wait + random.random())                       # random part avoids synchronized retries in a classroom
            continue
        if r.status_code == 400 and "temperature" in r.text and "temperature" in payload:
            payload.pop("temperature")                                # some models only accept their default temperature
            continue
        if r.status_code in (401, 403):
            return f"Error {r.status_code}: the key was rejected (invalid or expired; course keys last one day). Ask the teacher for a new .env file."
        return f"Error {r.status_code}: {r.text[:200]}"
    return "Error: the service is busy (rate limit). Wait a few seconds and run the cell again."
# --- END llm_helper ---

LLM = {"key": "", "provider": None}      # filled by setup_llm() at start-up

SYSTEM = ("You are a helpful assistant. Answer in the language of the question. "
          "When context documents are provided, base your answer on them and say when they do not contain the answer.")


class Retriever:
    def __init__(self, df, column):
        from sentence_transformers import SentenceTransformer
        self.df, self.column = df.reset_index(drop=True), column
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.emb = self.model.encode(self.df[column].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=True)

    def top(self, query, k=4):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        idx = np.argsort(self.emb @ q)[::-1][:k]
        return self.df.iloc[idx][self.column].astype(str).str[:1200].tolist()


def build_interface(retriever):
    def respond(message, history):
        messages = [{"role": "system", "content": SYSTEM}]
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        sources = retriever.top(message) if retriever else []
        if sources:
            context = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sources))
            messages.append({"role": "user", "content": f"Context documents:\n{context}\n\nQuestion: {message}"})
        else:
            messages.append({"role": "user", "content": message})
        answer = query_llm(messages=messages, max_tokens=400, temperature=0.2)
        if sources:
            answer += "\n\n---\nSources: " + " | ".join(f"[{i + 1}] {s[:80]}..." for i, s in enumerate(sources))
        return answer

    return gr.ChatInterface(respond, title="Assistant", description="LLM chat" + (" with retrieval over your documents" if retriever else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="CSV of documents to search (optional)")
    parser.add_argument("--column", default="text")
    parser.add_argument("--provider", choices=["openai", "anthropic", "mistral"], help="force a provider (default: first key found)")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    LLM = setup_llm(ask_if_missing=False)
    retriever = Retriever(pd.read_csv(args.data).dropna(subset=[args.column]), args.column) if args.data else None
    build_interface(retriever).launch(share=args.share)
