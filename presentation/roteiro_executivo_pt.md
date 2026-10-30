# Roteiro — Regimes de temperatura na Amazônia e o El Niño

Público: executivos graduados, não de matemática. Tom: de colega para colega. Explicar **o que** foi feito e **por que** cada escolha, sem fórmulas.
Duração: ~15 min + perguntas. Base: `amazonia_enso_results.pdf` (10 slides) + 2 slides novos sugeridos (A e B, abaixo).

---

## 1. A mensagem em três frases

1. **Em vez de medirmos a temperatura com um índice que escolhemos, deixamos uma IA olhar os mapas diários da Amazônia (45 anos) e descobrir sozinha quais "tipos de dia" existem.**
2. **Só depois cruzamos esses tipos de dia com o El Niño: 3 dos 15 acompanham o fenômeno de forma robusta, e o mapa deles reproduz o efeito físico conhecido do El Niño, que a IA nunca viu.**
3. **É a primeira peça de um método que estamos estendendo: precipitação, umidade e outras regiões.**

## 2. O arco da história (para você se guiar)

Pergunta → Por que o método tradicional é limitado → O que a IA fez (e o que não fez) → Por que essas variáveis → Resultado → Quão confiável → Limites → Próximos passos.

---

## 3. A parte técnica, em linguagem de gente graduada

### O problema que o método resolve
A análise clássica resume a Amazônia num número (ex.: temperatura média da bacia). Isso **joga fora a geografia**: um dia quente no centro da bacia e um dia quente no norte viram o mesmo número. Nosso método guarda o **mapa inteiro**, e por isso separou dois "tipos de El Niño quente" (um centrado no meio da bacia, outro mais ao norte) que a média misturaria.

### O que a IA faz — sem matemática
- Ela recebe um mapa de temperatura por dia, **sem data, sem rótulo, sem dizer "El Niño"**.
- Treino ("auto-supervisionado"): mostramos a ela **pedaços** diferentes do mesmo mapa e ela aprende que são "a mesma coisa". Para conseguir isso, ela precisa captar o que **caracteriza** aquele dia (onde está quente, onde está frio, qual a forma do padrão). Não existe "resposta certa" fornecida por nós.
- Junto com isso ela cria **15 "protótipos"**: tipos de dia. Cada um dos 16.436 dias cai no protótipo mais parecido.
- **Quem decidiu o quê:** nós decidimos os dados, o recorte da região e *quantos* grupos (15). **A IA decidiu o que entra em cada grupo.** Os nomes ("quente central", "frio na bacia toda") são nossos, dados depois, olhando os mapas.

### Por que essas escolhas (responda antes que perguntem)

| Escolha | Por quê | Alternativa / o que vem |
|---|---|---|
| **Temperatura mínima e máxima diárias** | É contínua, bem medida e bem conectada ao El Niño na Amazônia. Usar as duas capta noite e dia: a mínima reflete nuvem e umidade (retenção de calor), a máxima, o calor diurno. O padrão mais forte (C6) aparece mais na mínima. | **Precipitação:** já estamos trabalhando na versão com chuva. É a variável com resposta ao El Niño potencialmente mais forte, mas é mais difícil: a chuva tem muitos dias "zero" e valores extremos, o que exige cuidado. **Umidade** e outras variáveis: temos versões também. |
| **Anomalias** (desvio do normal para aquela época do ano), não a temperatura crua | Sem isso a IA agruparia "dias de verão" e "dias de inverno" e acharia que descobriu algo. Pior: o El Niño tem estação preferida (pico no fim do ano), então estação e El Niño se confundiriam. Com anomalias, o 1º grupo não é "é setembro". | — |
| **Escala diária** | É a escala em que o tempo "acontece"; mensal esconderia ondas de calor. | — |
| **Dados ERA5 (reanálise), 1981–2025** | Mistura observações com um modelo físico para ter um mapa **completo e contínuo**, sem buracos (a Amazônia tem poucas estações). Pixels de ~25 km. Disponível por décadas. | Dá para estender até 1940. |
| **15 grupos** | Escolha de projeto: poucos esconde diferenças, muitos vira ilegível. Os 15 são usados de forma equilibrada (cada um em 6–8% dos dias). O resultado central não depende do número exato. | Pode-se testar outros números. |
| **Amazônia (18°S–5,5°N, 74°W–44°W)** | Região onde o impacto do El Niño é mais sensível (seca, fogo, agro, saúde). | **Outras regiões:** já temos versões (ex.: Cerrado/Centro-Sudeste do Brasil). O método não é específico da Amazônia. |

### Por que 16 mil dias não são 16 mil provas
Dias seguidos são quase o mesmo dia. O que de fato conta como "experimento" é o **episódio**: um El Niño inteiro é *uma* observação. Em 45 anos há **11 El Niños e 12 La Niñas**. Por isso testamos tudo num nível mensal e de forma conservadora, e corrigimos pelo fato de olharmos 15 padrões ao mesmo tempo. Resultado: dos 15, só **3** ficam de pé. Isso é uma força, não uma fraqueza: o método não encontra El Niño em tudo.

---

## 4. Slide a slide

**1. Título (30 s).** "A pergunta: o El Niño deixa uma impressão digital reconhecível na temperatura da Amazônia? E dá para encontrá-la sem dizer à IA o que procurar?"

**[A — slide novo, 1,5 min] Como a IA chega nos padrões.** Diagrama: mapas diários → IA → 15 "tipos de dia" → só então comparamos com El Niño. Dizer: "a ordem importa: primeiro a IA agrupa, depois a gente compara".

**2. O que fizemos (1 min).** Percorra os 4 passos. Aqui conecte com o problema: "o método clássico resume tudo num número; nós mantemos o mapa".

**[B — slide novo, 1,5 min] Por que essas escolhas e o que vem depois.** Use a tabela da seção 3 (versão enxuta: variável, dado, região, 15 grupos). Linha de baixo: "Em andamento: precipitação · umidade · outras regiões".

**3. Os 15 padrões (1 min).** Mapas do "dia típico" de cada grupo, do mais frio ao mais quente. "Veja que há tanto padrões fortes e amplos (C6, C3) como padrões sutis, com dipolos, como o C8: frio no sudoeste, quente no norte. Nenhum foi desenhado por nós."

**4. Três padrões acompanham o El Niño (1 min).** "Para cada tipo de dia perguntamos: ele aparece mais quando o El Niño está mais forte? **Só 3 de 15 passam** no teste conservador: dois quentes (C6, C0) e um frio (C3, que aparece mais na La Niña)."

**5. C6: 7× mais comum (1 min).** "Em meses de El Niño, 21% dos dias são C6; em La Niña, 3%. A média geral é 9%." Se precisar: "contando **por evento**, a média é 20% mas varia de 0% a 71%, puxada pelos três mais fortes e recentes."

**6. Os mapas lembram o El Niño (1,5 min) — o argumento mais forte.** "À direita, o efeito do El Niño calculado *direto dos dados*. À esquerda, os padrões que a IA criou sem ver esse mapa. As duas coisas se parecem: o padrão quente tem a mesma estrutura espacial. **Isso é uma validação independente.**"

**7. Série no tempo (1 min).** Troque o título por "C6 sobe nos El Niños fortes". "1997–98, 2015–16, 2023–24. Nos fracos, quase não aparece."

**8. Padrões quentes mais comuns (1 min).** "Os padrões quentes ganharam espaço em 40 anos e os frios perderam. C6: +5 pontos percentuais por década. Isso pode ser El Niño *mais* aquecimento de fundo; com 11 El Niños não separamos as duas coisas."

**9. Vale nas duas metades (45 s).** "Dividimos o período em dois, 1981–2002 e 2003–2025, e refizemos tudo. As duas metades concordam (0,83)."

**10. Conclusões e próximos passos (1 min).** Três conclusões da seção 1 + roadmap: chuva, umidade, outras regiões, antes de 1980. Fechar com uso: os "tipos de dia" viram uma **linguagem simples** para cruzar com risco de fogo, safra, saúde.

---

## 5. Perguntas prováveis

- **"A IA decidiu sozinha?"** — O agrupamento, sim. Nós definimos dados, região e número de grupos; nada sobre El Niño.
- **"Por que não usar um índice, que é mais simples?"** — Índice resume e perde a geografia. Aqui o mapa mostrou dois tipos de calor ligados ao El Niño que a média misturaria.
- **"Isso prevê o El Niño ou a temperatura?"** — Não. Mostra associação no mesmo mês. Não conseguimos demonstrar antecedência com segurança (poucos episódios).
- **"É causa?"** — É associação robusta e coerente com a física conhecida. Não é prova causal isolada.
- **"Todo El Niño faz isso?"** — Não igual. Os fortes e recentes mais; os fracos, pouco.
- **"É aquecimento global?"** — Parte da tendência pode ser. Os dados não separam.
- **"Por que só temperatura?"** — Primeiro passo. Chuva (já em andamento) provavelmente tem resposta ainda mais forte, mas exige tratamento próprio.
- **"Dá para usar em outra região?"** — Sim, já temos versões para outras regiões.
- **"Como sei que não é acaso?"** — Três verificações: (1) só 3 dos 15 passam num teste que corrige por olharmos muitos; (2) o mapa bate com o efeito físico conhecido; (3) repete nas duas metades do período.

## 6. Ajustes sugeridos no deck

1. **Adicionar os slides A e B** (como a IA aprende; escolhas e próximos passos).
2. Traduzir para português.
3. Slide 7: título "C6 sobe nos El Niños fortes".
4. Rodapés (slides 4 e 9) com "p = 0,013", "IC 95%", "r": diminuir ou trocar por "confirmado por teste estatístico".
5. Slide 3: rodapé "5,6–7,6% dos dias" refere-se só aos anos de treino.
6. Opcional: slide com um ponto por El Niño (variação entre eventos).

## 7. Só para você

- Os números usam o índice ONI **corrigido** (`src/el_nino.py` tinha mês deslocado). Por isso diferem do rascunho do artigo; conclusões iguais (C6, C0, C3).
- **Confirmar antes de falar:** o status exato das versões com precipitação, umidade e outras regiões (resultados prontos ou só em andamento?). No roteiro escrevi "em andamento / já temos versões", como você disse, mas só os resultados de temperatura estão no deck.
