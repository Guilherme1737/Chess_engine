# Chess Coach

Aplicativo desktop local em Python/Tkinter para capturar um tabuleiro de xadrez na tela, reconhecer automaticamente a posicao (FEN) com visao computacional (OpenCV), permitir edicao manual das casas no tabuleiro e apresentar as melhores linhas da engine no estilo **Lichess / Chess.com**.

## Funcionalidades

- **Tabuleiro Editavel (Inline e Editor Completo)**:
  - Edicao direta clicando ou arrastando pecas no tabuleiro lateral (`240x240`) ou abrindo o **Editor Completo de Tabuleiro** (`BoardEditorDialog`);
  - Opcao **Exibir tabuleiro editavel antes de analisar** para revisar/corrigir casas incertas antes de disparar a engine;
  - **Aprendizado incremental de correcoes** (`learn_from_correction`): quando voce corrige uma casa manualmente, o reconhecedor aprende o template daquela peca para o tema atual.
- **Validacao da Posicao Reconhecida (`validate_fen` / `is_suspicious`)**:
  - Detecta dois reis em xeque simultaneamente, rei ausente ou multiplos reis, quantidade impossivel de pecas/promocoes, peao na 1ª/8ª fileira, reis adjacentes e vez invertida;
  - Bloqueia analises automaticas quando a FEN reconhecida for ilegal ou suspeita (baixa confianca / muitas casas incertas).
- **Historico de Posicoes e Lances (`PositionHistory`)**:
  - Salva todas as FENs reconhecidas/editadas na sessao;
  - Exibe a sequencia dos ultimos lances em SAN (`1. e4 e5 2. Nf3 Nc6`) e permite navegar com `⏮`, `◀ Voltar`, `Avancar ▶`, `⏭` (ou `Alt+Esquerda` / `Alt+Direita`).
- **Analise UCI por Profundidade ou Limite de Tempo (`UciEngine`)**:
  - Suporte a busca por **profundidade fixa** ou **tempo limite** (ex.: analisar por `5.0 s`), com botao **Parar (`⏹`)** e exibicao em tempo real da profundidade/velocidade da engine;
  - Painel **Multi-PV (Top 3 linhas)** com icones das pecas, notacao com figurinhas (`♘f3`, `♗c4`), barra de avaliacao e **explicacao tatico-posicional curta** (`explain_move`: captura, ganha material, ameaca, garfo, xeque, desenvolvimento, controle do centro, roque, promocao).
- **Modo de Treino Completo (`TrainingSession` / `evaluate_guess`)**:
  - Aceita qualquer lance entre os melhores da engine dentro da tolerancia configuravel em centipawns (`tolerance_cp`);
  - Registra acertos, erros, precisao (`%`), sequencia atual/recorde e perda media em centipawns.
- **Configuracoes Persistentes e Exportacao**:
  - Salva automaticamente regiao, orientacao, engine, profundidade, tempo limite, modo treino e geometria da janela em `settings.json`;
  - Exporta a partida em **PGN**, copia **FEN + SAN + PV + avaliacao** e gera **Relatorio completo da sessao** em Markdown.

## Como executar

```bash
python -m pip install -r requirements.txt
python app.py
```

## Testes automatizados

```bash
python -m unittest test_app.py -v
```
