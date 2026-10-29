# Chess Coach

Aplicativo local para capturar um tabuleiro de xadrez na tela, reconhecer automaticamente a posicao (FEN) com visao computacional leve (OpenCV) e apresentar as melhores linhas da engine no estilo **Lichess / Chess.com**.

## Funcionalidades

- **Captura e Selecao do Tabuleiro**:
  - Deteccao geometrica automatica do tabuleiro (`BoardDetector`);
  - **Seletor interativo com o mouse (`Selecionar area com mouse ✂`)**: sobreposicao em tela cheia estilo *Snipping Tool* para clicar e arrastar ao redor do tabuleiro (`RegionSelectorOverlay`);
- **Reconhecimento Automatico de Pecas, Orientacao e Vez (`PieceRecognizer`)**:
  - **Calibracao em 1 clique** na posicao inicial (`Calibrar posicao inicial`), salvando os templates em `calibration_data.npz` para uso em qualquer posicao futura;
  - **Auto-orientacao**: detecta automaticamente se voce esta jogando de Brancas (`white`) ou Pretas (`black`) embaixo (`detect_orientation`);
  - **Auto-vez e deteccao do ultimo lance**: compara a transicao do tabuleiro e ataques ao rei (`infer_turn_and_last_move`) para alternar sozinho entre `w` e `b`, indicando quando e a **SUA VEZ!** e destacando em amarelo o ultimo lance jogado no preview;
- **Painel de Analise Estilo Lichess / Chess.com**:
  - **Multi-PV (Top 3 linhas)** tanto para engines UCI externas (`UciEngine`) quanto para o avaliador local embutido (`LocalChessEvaluator`);
  - **Figuras graficas das pecas (`28x28` px)** + **Notacao Algebrica com Figurinhas** (`♘f3`, `♗c4`, `♕h5+`) + linha de continuacao completa (`1. e4 e5 2. ♘f3 ♞c6 3. ♗b5`);
  - **Barra de Avaliacao (Eval Bar)** vertical ao lado do preview do tabuleiro e **setas visuais** (verde para o 1º lance, azul para o 2º lance);
- **Melhorias de UX e Modo de Treino**:
  - Opcao **Fixar janela no topo (Always on Top)** para manter o painel visivel ao lado do navegador;
  - Atalho **`F8`** para disparar `Capturar + Reconhecer + Analisar` rapidamente e botao **Copiar FEN**;
  - **Modo de Treino** aceita lances em notacao em portugues (`Cf3`, `Bc4`, `Dh5`), ingles (`Nf3`), figurinhas (`♘f3`) ou UCI (`g1f3`).

## Como executar

```bash
python -m pip install -r requirements.txt
python app.py --engine /caminho/para/stockfish
```

*(O argumento `--engine` e opcional; se omitido, o avaliador local embutido sera utilizado).*

## Testes automatizados

```bash
python -m unittest test_app.py -v
```
