"""Make the pinned ``ragas`` (0.4.3) importable on the app's langchain 1.x stack.

Why this exists
---------------
``ragas==0.4.3`` opens its LLM module with two VertexAI imports that were
deleted from ``langchain-community`` when the Google integration moved out to
``langchain-google-genai``::

    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI

With ``langchain-community==0.4.2`` installed those two names no longer exist,
so ``import ragas`` raises ImportError before anything else runs. Everything
else ragas 0.4.3 needs works unchanged against the installed langchain-core 1.x
/ langchain-openai stack - with two caveats the caller has to handle, because
ragas 0.4.3 ships *two* metric families and only one of them still functions
with ``evaluate()``:

* the modern v2 classes under ``ragas.metrics.collections`` are not subclasses
  of ``ragas.metrics.base.Metric``, which ``evaluate()`` type-checks against,
  so :mod:`evaluation.ragas_real` resolves metrics from the legacy
  ``ragas.metrics`` registry instead; and
* the legacy registry's embeddings classes are unusable as shipped
  (``HuggingfaceEmbeddings`` is abstract, the modern ``HuggingFaceEmbeddings``
  has no ``embed_query``), so :mod:`evaluation.ragas_real` builds its own.

Downgrading the whole app to langchain-community 0.2.x (to resurrect the old
VertexAI helpers) would be a far riskier change than supplying two missing
names that ragas never actually uses unless you ask it for a VertexAI model.

Approach
--------
A narrow import hook that *only* stubs ``langchain_community`` submodules that
genuinely no longer exist. It always defers to the real finder first, so
nothing that does exist is ever shadowed, and the stubs are never instantiated
- ragas only needs the names to resolve so its module body can import.

Eval-only: nothing under ``src/`` imports this, so the production pipeline is
unaffected.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
import types

#: The imports that no longer resolve (documented for the README / runbook).
MISSING_LC_SYMBOLS = (
    "langchain_community.chat_models.vertexai.ChatVertexAI",
    "langchain_community.llms.VertexAI",
)

_PREFIX = "langchain_community."


class _StubModuleLoader(importlib.abc.Loader):
    """Creates an empty package-like module; attributes are synthesised lazily."""

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []  # allow submodules beneath a stub to resolve
        return module

    def exec_module(self, module):
        module.__getattr__ = _synthesize


def _synthesize(name):
    # module-level __getattr__ (PEP 562) - only consulted after normal lookup
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return type(name, (), {})


class _MissingLangchainCommunityFinder(importlib.abc.MetaPathFinder):
    """Stubs ``langchain_community.<x>`` only when ``<x>`` is really absent."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX) or fullname in sys.modules:
            return None
        if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
            return None  # the real module exists - never shadow it
        return importlib.machinery.ModuleSpec(
            fullname, _StubModuleLoader(), is_package=True
        )


def install_hook() -> bool:
    """Install the import hook once. Returns True if it was newly added."""
    if any(isinstance(finder, _MissingLangchainCommunityFinder)
           for finder in sys.meta_path):
        return False
    # index 0 so a missing module is stubbed before the erroring lazy loader
    sys.meta_path.insert(0, _MissingLangchainCommunityFinder())
    return True


def ensure_ragas_importable():
    """Install the hook (idempotent) and return the imported ``ragas`` module."""
    install_hook()
    import ragas

    return ragas
