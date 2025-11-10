# Roteiro de Vídeo (até 4 minutos)

**0:00 – 0:15 • Abertura**
- Apresentação rápida do projeto “Rota Inteligente — Sabor Express (Santana/SP)”.
- Objetivo: reduzir tempo e custo das entregas com IA clássica.

**0:15 – 0:45 • Problema**
- Rotas manuais causam atrasos e gasto de combustível.
- Mostrar no mapa (screenshot) a região de Santana / Av. Braz Leme e a pizzaria de referência.

**0:45 – 1:40 • Solução (Visão Geral)**
- Cidade como **grafo** com pesos = distância haversine.
- **K-Means (k=3)** para criar **zonas** de entrega.
- Em cada zona: ordem de visita **Greedy (Nearest Neighbor)** e trechos entre pares com **A***.
- Métricas de eficiência e visualização das rotas.

**1:40 – 2:40 • Funcionamento do Código**
- Explicar `data/points.csv` (R0 + 10 pontos com lat/lon).
- Rodar `python src/main.py`.
- Mostrar `outputs/rotas_clusters.png` e `outputs/metrics.json`.
- Comentar escolhas: A* é ótimo entre pares; greedy aproxima TSP rapidamente; K-Means simples e eficaz.

**2:40 – 3:20 • Resultados**
- Fale dos km totais e por cluster (mostrar `metrics.json` na tela).
- Benefícios: balanceamento entre entregadores, menos sobreposição de rotas.

**3:20 – 3:50 • Limitações e Próximos Passos**
- Não usa tráfego em tempo real nem regras de trânsito.
- Melhorias: API de mapas, tuning automático de K, algoritmos genéticos / RL para dinâmica.

**3:50 – 4:00 • Fechamento**
- Conclusão e convite para ver o repositório / documentação.
