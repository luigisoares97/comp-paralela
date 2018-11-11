(1) Tempos de execução:

    (a) CUDA: 0m1.626s
    (b) Sequencial: 0m0.324s
    (c) OpenMP: 0m0.314s
    (d) OpenMP - GPU: 0m1.688s
    (e) CUDA - Global: 0m1.723s

(2) Nvprof:

    (a) CUDA:
        (a.1) CUDA memcpy HtoD: 465.46ms
        (a.2) sum_cuda: 21.560ms
    (b) OpenMP - GPU:
        (b.1) CUDA memcpy HtoD: 463.25ms
        (b.2) sum_cuda: 8.1613ms
    (c) CUDA - Global:
        (c.1) CUDA memcpy HtoD: 469.75ms
        (c.2) sum_cuda: 31.788ms

(3) Explicação:

    Os resultados mostram um tempo de execução maior para aqueles códigos relacionados a GPU. Isso ocorre devido a complexidade do algoritmo (linear). Por ser bastante simples, o custo associado a cópia das informações da memória principal para a memória da GPU não compensa.
    Em relação ao uso de memória global ou compartilhada (por cada grupo), no CUDA, o que pode-se concluir é que há um prejuízo na utilização da memória global. Isso ocorre, porque o acesso a um dado presenta na memória global é mais custoso. A memória compartilhada é específica para cada grupo de threads, sendo assim o custo de acesso é, logicamente, menor.
