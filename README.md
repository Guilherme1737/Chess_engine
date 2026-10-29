# Chess Coach

Um prototipo local para capturar um tabuleiro de xadrez na tela e analisar a posicao com uma engine UCI, como o OpenTal.

## O que ja funciona

- captura da tela inteira com `mss`;
- deteccao automatica de um tabuleiro retangular usando OpenCV;
- recorte do tabuleiro e deteccao geometrica para layouts diferentes;
- analise de FEN com qualquer engine UCI;
- modo manual: informe a regiao do tabuleiro e a FEN quando o reconhecimento automatico das pecas nao for suficiente.

O reconhecimento de pecas e deliberadamente um proximo modulo. A geometria do tabuleiro e a analise UCI ficam separadas para que um modelo de visao possa ser adicionado sem alterar a interface da engine.

## Como executar

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --engine /caminho/para/OpenTal
```

No aplicativo:

1. clique em `Capturar tela`;
2. clique em `Detectar tabuleiro` ou informe `x,y,w,h`;
3. informe a FEN da posicao, caso ainda nao tenha um modelo de reconhecimento de pecas;
4. clique em `Analisar`.

O campo da engine aceita qualquer executavel que fale UCI. Para uma experiencia sem engine instalada, o programa ainda permite validar a captura e a deteccao.

## Proximos passos tecnicos

1. adicionar um classificador de pecas treinado com imagens de diferentes temas e escalas;
2. acompanhar mudancas na tela e recalcular apenas quando a posicao mudar;
3. incluir modo de treino que esconde a melhor jogada e registra erros;
4. manter o processamento local e exibir uma indicacao clara quando a posicao estiver incerta.