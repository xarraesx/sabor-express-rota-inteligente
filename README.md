# Rota Inteligente — Sabor Express (Santana/SP)

## Problema
A Sabor Express enfrenta atrasos e altos custos por **rotas manuais**. O objetivo é **otimizar entregas** na região de Santana (SP) usando **IA clássica**.

## Abordagem
1. **Grafo completo** com nós (pontos) e arestas ponderadas por distância **haversine** (km).
2. **K-Means (k=3)** para agrupar pedidos próximos em **zonas**.
3. **A*** como buscador de caminho entre pares; para visitar todos os pontos do cluster, usamos **heurística gulosa** (nearest neighbor) para ordenar as visitas e **A*** para cada trecho.
4. Métricas: km por cluster e total.

## Algoritmos
- **A*** (heurístico: distância haversine ao alvo) — encontra menor caminho entre dois nós.
- **K-Means** (não supervisionado) — cria zonas de entrega.
- **Greedy NN** — ordem de visita em cada cluster (aproxima TSP).
- **Grafo** — completo; pesos = distâncias geográficas.

## Resultados (executando este repositório)
- Total percorrido (soma dos clusters): ver `outputs/metrics.json`.
- Visual das rotas por cluster: `outputs/rotas_clusters.png`.

## Estrutura
```
/data/points.csv        # R0 + 10 pontos (lat, lon)
/src/main.py            # Execução completa
/outputs/*.png|json     # Gráficos e métricas
/docs/video_pitch.md    # Roteiro de 4 minutos
README.md
```


