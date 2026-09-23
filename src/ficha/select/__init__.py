"""Estratégias de seleção de contexto (Seção 4.1). Todas implementam ``ContextSelector``.

Implementações esperadas (ver docs/ARCHITECTURE.md):
- ``first_pages.FirstPagesSelector``  — as N primeiras páginas.
- ``keyword.KeywordSectionSelector``  — seções localizadas por palavra-chave.
- ``semantic.SemanticSelector``       — chunks por similaridade de embeddings.
- ``chunking``                         — segmentação com tamanho/sobreposição declarados.
"""
