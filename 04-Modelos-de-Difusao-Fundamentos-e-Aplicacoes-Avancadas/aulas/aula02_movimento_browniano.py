# ==========================================================
# PROJETO 04 - MODELOS DE DIFUSAO
# AULA 02 - INTUICAO DOS MODELOS
# SIMULACAO DO MOVIMENTO BROWNIANO EM 3D
# ==========================================================

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ==========================================================
# 1 - CONFIGURACAO
# ==========================================================

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = BASE / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

GIF = RESULTADOS / "aula02_movimento_browniano.gif"
PNG = RESULTADOS / "aula02_trajetoria_3d.png"
CSV = RESULTADOS / "aula02_trajetoria_particula.csv"
JSON = RESULTADOS / "aula02_parametros.json"

n_steps = 500
n_particles = 100

particle_index = 10

np.random.seed(42)

print("=" * 65)
print("AULA 02 - MOVIMENTO BROWNIANO EM 3D")
print("=" * 65)


# ==========================================================
# 2 - CRIAR ARRAYS INICIAIS
# ==========================================================

print("\n1 - CRIANDO PARTICULAS")

x = np.zeros((n_steps, n_particles))
y = np.zeros((n_steps, n_particles))
z = np.zeros((n_steps, n_particles))

print("Particulas:", n_particles)
print("Passos:", n_steps)
print("Dimensoes: X, Y, Z")


# ==========================================================
# 3 - SIMULAR MOVIMENTO BROWNIANO
# ==========================================================

print("\n2 - ADICIONANDO RUIDO")

for i in range(1, n_steps):

    x[i] = (
        x[i - 1]
        + np.random.normal(0, 0.2, n_particles)
    )

    y[i] = (
        y[i - 1]
        + np.random.normal(0, 0.2, n_particles)
    )

    z[i] = (
        z[i - 1]
        + np.random.normal(0, 0.2, n_particles)
    )

print("Movimento browniano calculado: OK")


# ==========================================================
# 4 - CRIAR FIGURA EM 3D
# ==========================================================

print("\n3 - CRIANDO VISUALIZACAO 3D")

fig = plt.figure(figsize=(8, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)

sc = ax.scatter(
    x[0],
    y[0],
    z[0],
    c="blue",
    s=20,
    alpha=0.6
)

highlight = ax.scatter(
    [x[0, particle_index]],
    [y[0, particle_index]],
    [z[0, particle_index]],
    s=100,
    c="lime",
    edgecolors="black",
    label=f"Particula {particle_index}"
)

traj_line, = ax.plot(
    [],
    [],
    [],
    color="black",
    lw=1,
    alpha=0.7,
    label="Trajetoria"
)

start_point = ax.scatter(
    [x[0, particle_index]],
    [y[0, particle_index]],
    [z[0, particle_index]],
    color="green",
    s=80,
    label="Inicio"
)

end_point = ax.scatter(
    [],
    [],
    [],
    color="red",
    s=80,
    label="Fim"
)

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-20, 20)

ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
ax.set_zlabel("Eixo Z")

ax.legend(loc="upper right")


# ==========================================================
# 5 - FUNCAO DE ATUALIZACAO DA ANIMACAO
# ==========================================================

def update(frame):

    sc._offsets3d = (
        x[frame],
        y[frame],
        z[frame]
    )

    highlight._offsets3d = (
        [x[frame, particle_index]],
        [y[frame, particle_index]],
        [z[frame, particle_index]]
    )

    # Trajetoria percorrida ate o quadro atual
    traj_line.set_data(
        x[:frame + 1, particle_index],
        y[:frame + 1, particle_index]
    )

    traj_line.set_3d_properties(
        z[:frame + 1, particle_index]
    )

    if frame == n_steps - 1:

        end_point._offsets3d = (
            [x[frame, particle_index]],
            [y[frame, particle_index]],
            [z[frame, particle_index]]
        )

    else:

        end_point._offsets3d = (
            [],
            [],
            []
        )

    ax.view_init(
        elev=30,
        azim=frame / 2
    )

    ax.set_title(
        f"Movimento Browniano 3D - Passo {frame + 1}/{n_steps}"
    )

    return (
        sc,
        highlight,
        traj_line,
        start_point,
        end_point
    )


# ==========================================================
# 6 - GERAR GIF ANIMADO
# ==========================================================

print("\n4 - GERANDO ANIMACAO")
print("Total: 500 quadros")
print("O processamento pode ser demorado.")

ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_steps,
    interval=50,
    blit=False
)

ani.save(
    GIF,
    writer=animation.PillowWriter(fps=20),
    dpi=75,
    progress_callback=lambda atual, total: (
        print(
            f"Quadros processados: {atual + 1}/{total}",
            flush=True
        )
        if (atual + 1) % 100 == 0
        else None
    )
)

print("GIF gerado: OK")


# ==========================================================
# 7 - SALVAR IMAGEM FINAL
# ==========================================================

update(n_steps - 1)

fig.savefig(
    PNG,
    dpi=140,
    bbox_inches="tight"
)

plt.close(fig)

print("Imagem PNG gerada: OK")


# ==========================================================
# 8 - SALVAR TRAJETORIA DA PARTICULA 10
# ==========================================================

trajetoria = np.column_stack([
    np.arange(n_steps),
    x[:, particle_index],
    y[:, particle_index],
    z[:, particle_index]
])

np.savetxt(
    CSV,
    trajetoria,
    delimiter=",",
    header="passo,x,y,z",
    comments="",
    fmt=["%d", "%.6f", "%.6f", "%.6f"]
)

with JSON.open("w", encoding="utf-8") as arquivo:

    json.dump(
        {
            "aula": 2,
            "tema": "Movimento Browniano",
            "passos": n_steps,
            "particulas": n_particles,
            "particula_destacada": particle_index,
            "desvio_padrao_ruido": 0.2,
            "seed": 42
        },
        arquivo,
        indent=4
    )


# ==========================================================
# 9 - VALIDACAO
# ==========================================================

print("\n" + "=" * 65)
print("VALIDACAO FINAL")
print("=" * 65)

assert x.shape == (500, 100)
assert y.shape == (500, 100)
assert z.shape == (500, 100)

assert np.all(x[0] == 0)
assert np.all(y[0] == 0)
assert np.all(z[0] == 0)

assert np.isfinite(x).all()
assert np.isfinite(y).all()
assert np.isfinite(z).all()

assert GIF.exists() and GIF.stat().st_size > 0
assert PNG.exists() and PNG.stat().st_size > 0
assert CSV.exists() and CSV.stat().st_size > 0
assert JSON.exists() and JSON.stat().st_size > 0

print("NumPy: OK")
print("100 particulas: OK")
print("500 passos: OK")
print("Movimento nas tres dimensoes: OK")
print("Particula 10 destacada: OK")
print("Animacao GIF: OK")
print("Imagem PNG: OK")
print("Trajetoria CSV: OK")
print("Parametros JSON: OK")

print("\nAULA 02 EXECUTADA COM SUCESSO!")

print("\nGIF:", GIF)
print("PNG:", PNG)
